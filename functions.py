import numpy as np
import torch
print("pytorch verison =", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
#import netCDF4 as nc
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data
from torch.optim import Adam
import math
device = "cuda" if torch.cuda.is_available() else "cpu"

#Parameters
path_outputs = "./outputs/"
lead = 1
delta_t =0.01

batch_size = 256
lamda_reg =0.2
wavenum_init=0 #10
wavenum_init_ydir=0 #10

def mse_loss(output, target, wavenum_init,lamda_reg):
    loss1 = F.mse_loss(output,target) 
    return loss1

def spectral_loss_ashesh(output, target, wavenum_init,lamda_reg):
    loss1 = F.mse_loss(output,target)
    out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
    target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2)
    loss2 = torch.mean(torch.abs(out_fft[:,0,wavenum_init:]-target_fft[:,0,wavenum_init:]))
    loss3 = torch.mean(torch.abs(out_fft[:,1,wavenum_init:]-target_fft[:,1,wavenum_init:]))
    loss4 = torch.mean(torch.abs(out_fft[:,2,wavenum_init:]-target_fft[:,2,wavenum_init:]))

   # loss = (1-lamda_reg)*loss1 + 0.33*lamda_reg*loss2 + 0.33*lamda_reg*loss2_ydir + 0.33*LC_loss
    loss = 0.25*(1-lamda_reg)*loss1 + 0.25*(lamda_reg)*loss2 + 0.25*(lamda_reg)*loss3 + 0.25*(lamda_reg)*loss4
    return loss


def RK4step(net,input_batch):
    output_1 = net(input_batch.cuda())
    output_2= net(input_batch.cuda()+0.5*output_1)
    output_3 = net(input_batch.cuda()+0.5*output_2)
    output_4 = net(input_batch.cuda()+output_3)
    return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6

def Eulerstep(net,input_batch):
    output_1 = net(input_batch.cuda())
    return input_batch.cuda() + delta_t*(output_1)
 
def PECstep(net,input_batch):
    output_1 = net(input_batch.cuda()) + input_batch.cuda()
    return input_batch.cuda() + delta_t*0.5*(net(input_batch.cuda())+net(output_1))

def directstep(net,input_batch):
    output_1 = net(input_batch.cuda())
    return output_1

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device) # x_0 + noise = x_noisy 
    noise_pred = model(x_noisy, t) # x_noisy ---> noise
    return F.l1_loss(noise, noise_pred)

def get_loss_cond(model, x_0, t, label_batch):  #???
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, x_0, t)
    return  mse_loss((x_noisy-noise_pred), label_batch , wavenum_init, lamda_reg)


def get_loss_cond_egnn(model, x_0, t, label_batch, is_gen_step = False):  #???
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)

    x_0 = x_0.squeeze().to(t.device)
    x_noisy = x_noisy.squeeze().to(t.device)

    n_frames, n_atoms, n_features = x_0.shape
    # input = noise + x_0
    x_coord, x_force_speed = torch.split(x_noisy, [3, 6], dim=-1)
    # distance mtx and masks
    dist_mtx = calc_distance(x_coord).to(t.device)
    mask = torch.ones((n_frames, n_atoms)).to(t.device)
    mask2d = dist_mtx < 0.9
    # currently [x_noisy, x_0] are passed to atom_feat in the model
    feat_noise_pred, coord_noise_pred = model(x_force_speed, x_coord, t.view(-1, 1), 
                                                adj_mat=mask2d, mask=mask, mask2d=mask2d, condition=x_0)

    noise_pred = feat_noise_pred.unsqueeze(1)
    #noise_pred = torch.cat((coord_noise_pred, feat_noise_pred), dim=-1).unsqueeze(1)     
    x_noisy = x_noisy.unsqueeze(1)   

    if is_gen_step:
        return x_noisy

    # noise_pred = model(x_noisy, x_0, t) # currently not implemented conditional diffusion (is it really conditional?)
    # (x_0+noise-noise_pred) - x_0 = x_1 - x_0
    # mse((x_1-x_0), (noise-noise_pred))
    return  mse_loss((x_noisy-label_batch), noise_pred, wavenum_init, lamda_reg)


def sample_from_egnn(model, x_noisy, cond, t):
    # squeeze tensors before pass to the model
    x_noisy = x_noisy.squeeze(1)
    cond = cond.squeeze(1)
    
    n_frames, n_atoms, n_features = x_noisy.shape
    x_coord, x_force_speed = torch.split(x_noisy, [3, 6], dim=-1)
    dist_mtx = calc_distance(x_coord).to(t.device)
    mask = torch.ones((n_frames, n_atoms)).to(t.device)
    mask2d = dist_mtx < 0.9

    feat_noise_pred, coord_noise_pred = model(x_force_speed, x_coord, t.view(-1, 1), 
                                                adj_mat=mask2d, mask=mask, mask2d=mask2d, condition=cond)

    noise_pred = feat_noise_pred.unsqueeze(1)
    
    return x_noisy - noise_pred



@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    if (t==0):
        return model_mean
    else: 
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

 
def forward_diffusion_sample(x_0, t, device="cuda"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 64, 64, 64, 64)
        up_channels = (64, 64, 64, 64, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding='same')

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])


        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

def normalize_md(x):
    #x: [5000,64,9]
    #x_out = np.zeros_like(x)
    d1,d2,d3 = np.shape(x)
    mean_list = [np.average(x[:,:,i]) for i in range(d3)]
    std_list = [np.std(x[:,:,i]) for i in range(d3)]
    for i in range(d3):
        for index_1 in range(d1):
            for index_2 in range(d2):
                 x[index_1,index_2,i] = (x[index_1,index_2,i]-mean_list[i])/(std_list[i])
    #print(np.amax(x), np.amin(x), mean_list, std_list)
    return x, mean_list, std_list


def denormalize_md(x, mean_list, std_list):
    #x: [5000,1,64,9]
    d1,d2,d3,d4=np.shape(x)
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                for l in range(d4):
                    x[i,j,k,l] = x[i,j,k,l]*std_list[l]+mean_list[l]
    return x


def denormalize_md_pred(x, mean_list, std_list):
    #(1000, 100, 20, 1, 64, 9)
    d0, d1,d2,_,d3,d4=np.shape(x)
    for ii in range(d0):
        for i in range(d1):
            for j in range(d2):
                for k in range(d3):
                    for l in range(d4):
                        x[ii, i,j,0,k,l] = x[ii,i,j,0,k,l]*std_list[l]+mean_list[l]
    return x


def calc_distance(x:torch.Tensor):
    n_frames, n_atoms, n_features = x.shape
    assert n_features == 3,  "The last dimension should be 3"

    dist_mtx = torch.zeros((n_frames, n_atoms, n_atoms))
    x2 = torch.sum(torch.square(x), dim=-1)
    y2 = torch.sum(torch.square(x), dim=-1)
    xy = torch.matmul(x, x.transpose(-1, -2))
    dist_mtx = torch.sqrt(torch.maximum(x2[:, :, None] + y2[:, None, :] - 2 * xy, torch.tensor(1e-6)))

    return dist_mtx


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
