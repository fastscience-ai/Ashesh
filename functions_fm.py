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


def sample_from_egnn(model, x_noisy, cond, t, max_timestep):
    """
    x_noisy: noisy input at timestep t (Tensor)
    cond: conditioning data (x_0) (Tensor)
    t: integer timestep in [0, max_timestep]
    max_timestep: the maximum timestep 
    """
    device = x_noisy.device

    # Squeeze tensors before passing to the model
    x_noisy = x_noisy.squeeze(1)
    cond = cond.squeeze(1)
    
    n_frames, n_atoms, n_features = x_noisy.shape
    x_coord, x_force_speed = torch.split(x_noisy, [3, 6], dim=-1)
    dist_mtx = calc_distance(x_coord).to(device)
    mask = torch.ones((n_frames, n_atoms), device=device)
    mask2d = dist_mtx < 0.9

    # Scale t for model input
    t_scaled = t.float() / max_timestep  # Now t_scaled is in [0, 1]
    t_input = t_scaled.view(-1, 1).to(device)

    # Model prediction
    feat_noise_pred, coord_noise_pred = model(
        x_force_speed, x_coord, t_input,
        adj_mat=mask2d, mask=mask, mask2d=mask2d, 
        condition=cond
    )

    # Compute the predicted x_t
    eps_ = 1e-6
    t_scaled = t_scaled.clamp(eps_, 1 - eps_)
    t_scaled = t_scaled.view(-1, *([1] * (x_noisy.dim() - 1)))  # [batch_size, 1, 1, ...]

    std = torch.sqrt(t_scaled * (1 - t_scaled) + eps_)

    # Update x_noisy using the predicted noise
    x_prev = x_noisy - std * feat_noise_pred

    # Reshape to add back the singleton dimension
    x_prev = x_prev.unsqueeze(1)  # [batch_size, 1, n_atoms, n_features]

    return x_prev



def get_loss_cond_egnn(model, x_0, t, label_batch, max_timestep, is_gen_step=False):
    """
    x_0: previous state (Tensor)
    label_batch: next state (x_1) (Tensor)
    t: integer timestep in [0, max_timestep]
    max_timestep: the maximum timestep (e.g., 100)
    is_gen_step: boolean flag for generation step
    """
    device = x_0.device

    # Generate x_t using the stochastic interpolant forward function
    x_t = stochastic_interpolant_forward(x_0, label_batch, t, max_timestep)

    x_0 = x_0.squeeze().to(device)
    x_t = x_t.squeeze().to(device)
    n_frames, n_atoms, n_features = x_0.shape

    # Split x_t into coordinates and features
    x_coord, x_force_speed = torch.split(x_t, [3, 6], dim=-1)

    # Compute distance matrix and masks
    dist_mtx = calc_distance(x_coord).to(device)
    mask = torch.ones((n_frames, n_atoms), device=device)
    mask2d = dist_mtx < 0.9

    # Scale t for model input
    t_scaled = t.float() / max_timestep  # Now t_scaled is in [0, 1]
    t_input = t_scaled.view(-1, 1).to(device)

    # Model prediction
    feat_noise_pred, coord_noise_pred = model(
        x_force_speed, x_coord, t_input,
        adj_mat=mask2d, mask=mask, mask2d=mask2d, 
        condition=x_0
    )

    # Compute the actual noise epsilon
    with torch.no_grad():
        eps_ = 1e-6
        t_scaled = t_scaled.clamp(eps_, 1 - eps_)
        t_scaled = t_scaled.view(-1, *([1] * (x_0.dim() - 1)))  # [batch_size, 1, 1, ...]

        mean = (1 - t_scaled) * x_0 + t_scaled * label_batch
        std = torch.sqrt(t_scaled * (1 - t_scaled) + eps_)
        epsilon = (x_t - mean) / std

    # Reshape tensors
    noise_pred = feat_noise_pred.unsqueeze(1)
    epsilon = epsilon.unsqueeze(1)

    if is_gen_step:
        return x_t

    # Compute loss between predicted noise and actual noise
    loss = mse_loss(noise_pred, epsilon, None, None)

    return loss


def stochastic_interpolant_forward(x0, x1, t, max_timestep):
    """
    x0: previous state (Tensor)
    x1: next state (Tensor)
    t: integer timestep in [0, max_timestep]
    max_timestep: the maximum timestep (e.g., 100)
    """
    device = x0.device
    dtype = x0.dtype

    # Scale t to [0, 1]
    t_scaled = t / max_timestep  # Now t_scaled is in [0, 1]

    # Ensure t_scaled is within [0, 1]
    eps_ = 1e-6
    t_scaled = t_scaled.clamp(eps_, 1 - eps_)

    if not torch.is_tensor(t_scaled):
        t_scaled = torch.tensor(t_scaled, device=device, dtype=dtype)
    if t_scaled.dim() == 0:
        t_scaled = t_scaled.expand(x0.size(0))
    t_scaled = t_scaled.view(-1, *([1] * (x0.dim() - 1)))  # [batch_size, 1, 1, ...]

    mean = (1 - t_scaled) * x0 + t_scaled * x1
    std = torch.sqrt(t_scaled * (1 - t_scaled))
    epsilon = torch.randn_like(x0)

    x_t = mean + std * epsilon
    return x_t




#================================================================================================



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


# # Define beta schedule
# T = 300
# betas = linear_beta_schedule(timesteps=T)

# # Pre-calculate different terms for closed form
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, axis=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
