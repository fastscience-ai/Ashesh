import numpy as np
import torch
print("pytorch verison =", torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
#import netCDF4 as nc
# from data_loader_one_step_UVS import load_test_data
# from data_loader_one_step_UVS import load_train_data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable
from .args import get_parser

parser = get_parser()
args = parser.parse_args([])

import math
device = "cuda" if torch.cuda.is_available() else "cpu"

#Parameters
path_outputs = "./outputs/"
lead = 1
delta_t =0.01

#batch_size = 64
lamda_reg =0.2
cutoff = 10.0
wavenum_init=0 #10
wavenum_init_ydir=0 #10

#cell_vector = torch.tensor([[[21.04, 0.0, 0.0], [0.0, 21.04, 0.0], [0.0, 0.0, 21.04]]], device=device)
#cell_vector = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device)

def model_pbc_call(model, 
        X_t_orig: torch.Tensor, # X(K,0) or X(K,t)
        X_noisy: torch.Tensor,  # noise + X_0
        X_0: torch.Tensor,      # X_0
        t: torch.Tensor,
        temp: torch.Tensor,
        lattice:torch.Tensor):
        #ca):: str
    
    # Not implimented yet

    device = X_t.device
    
    X_noisy_wrappedd = pbc_coord(X_noisy, lattice)
    pred = model(X_noisy_wrappeedd, X_0, t, temps)

    dX = pred[:, 0:3]

    V_pred = torch.zeros_like(X_t_orig, device=device)
    F_pred = torch.zeros_like(X_t_orig, device=device)
    
    # dx = X(K+1, 0) - X(K, 0) x_t_orig : pred pbc X
    # dx = X(K+1, 0) - X(K, t) x_t_orig : ?

    # X(K+1,0) = X(K,0):X_t_orig + dX
    # X(K+1,0) = X(K,t):X_t_orig + dX 
    
    # dX = pred
    
    new_coord = X_t_orig + dX

    return new_coord, dX, V_pred, F_pred
    #return new_coord

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

def get_loss_cond(model, x_0, t, label_batch, temps=None):  #???
    x_noisy, noise = forward_diffusion_sample(label_batch, t, device)
    x_noisy = x_noisy.unsqueeze(1)
    noise = noise.unsqueeze(1)
    # if temps is not None:
    #     temps = temps.reshape(*temps.shape, 1, 1, 1).expand(*temps.shape, 1, 64, 1)
    #     x_0 = torch.cat([x_0, temps], dim=-1)
    noise_pred = model(x_noisy, x_0, t, temps)
    return mse_loss((x_noisy-noise_pred), label_batch , wavenum_init, lamda_reg)
    # mse_loss(noise, noise_pred , wavenum_init, lamda_reg)


def get_loss_cond_orig(model, x_0, t, label_batch, temps=None):  #???
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    x_noisy = x_noisy.unsqueeze(1)
    noise = noise.unsqueeze(1)
    # if temps is not None:
    #     temps = temps.reshape(*temps.shape, 1, 1, 1).expand(*temps.shape, 1, 64, 1)
    #     x_0 = torch.cat([x_0, temps], dim=-1)
    noise_pred = model(x_noisy, x_0, t, temps)
    out = x_noisy-noise_pred

    pos_loss = mse_loss(out[..., :3], label_batch[..., :3], wavenum_init, lamda_reg)
    vel_loss = mse_loss(out[..., -3:], label_batch[..., -3:], wavenum_init, lamda_reg)
    force_loss = mse_loss(out[..., 3:-3], label_batch[..., 3:-3], wavenum_init, lamda_reg)
    
    return pos_loss + vel_loss * 10 + force_loss
    
    #mse_loss((x_noisy-noise_pred), label_batch , wavenum_init, lamda_reg)


def get_loss_cond_diff(model, x_0, t, label_batch, temps=None):  #???
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    x_noisy = x_noisy.unsqueeze(1)
    noise = noise.unsqueeze(1)

    noise_pred = model(x_noisy, x_0, t, temps)
    difference = label_batch - x_0

    pos_loss = mse_loss(difference[..., :3], noise_pred[..., :3], wavenum_init, lamda_reg)
    vel_loss = mse_loss(difference[..., -3:], noise_pred[..., -3:], wavenum_init, lamda_reg)
    force_loss = mse_loss(difference[..., 3:-3], noise_pred[..., 3:-3], wavenum_init, lamda_reg)
    
    return pos_loss + vel_loss * 10 + force_loss


def get_loss_cond_rev(model, x_0, t, label_batch, temps=None):  
    # add noise to GT data
    x_noisy, noise = forward_diffusion_sample(label_batch, t, device)
    x_noisy = x_noisy.unsqueeze(1)
    noise = noise.unsqueeze(1)
    # if temps is not None:
    #     temps = temps.reshape(*temps.shape, 1, 1, 1).expand(*temps.shape, 1, 64, 1)
    #     x_0 = torch.cat([x_0, temps], dim=-1)
    # and make the model to predict the noise wirh [x_K, noised_x_K+1]
    noise_pred = model(x_noisy, x_0, t, temps)
    return mse_loss(noise_pred, noise, wavenum_init, lamda_reg)


def get_loss_cond_direct(model, x_0, t, label_batch, temps=None):  
    x_noisy, noise = forward_diffusion_sample(label_batch, t, device)
    x_noisy = x_noisy.unsqueeze(1)

    output = model(x_noisy, x_0, t, temps)
    return mse_loss(output, label_batch, wavenum_init, lamda_reg)


def get_loss_cond_pos(model, x_0, t, label_batch, temps=None):  #???
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    x_noisy = x_noisy.unsqueeze(1)
    noise = noise.unsqueeze(1)
    if temps is not None:
        temps = temps.reshape(*temps.shape, 1, 1, 1).expand(*temps.shape, 1, 64, 1)
        x_0 = torch.cat([x_0, temps], dim=-1)
    noise_pred = model(x_noisy, x_0, t)

    pred_pos = x_noisy[..., :3] - noise_pred[..., :3]
    x_0_pos = x_0[..., :3]
    vel_pred = (pred_pos - x_0_pos) / 2.5
    label_vel = label_batch[..., -3:]
    # print('pred_pos shape: ', pred_pos.shape)
    # print("x_0 shape: ", x_0.shape)
    # print("label_vel shape: ", label_vel.shape)

    vel_loss = mse_loss(vel_pred, label_vel, wavenum_init, lamda_reg)
    pos_loss = mse_loss(pred_pos, label_batch[..., :3], wavenum_init, lamda_reg)

    return pos_loss + vel_loss * 0.5



def RK4_sampler(model, x_noisy, cond, tt):
    output_1 = model(x_noisy, cond, tt)
    cond[..., :9] = cond[..., :9] + 0.5*output_1
    output_2 = model(x_noisy, cond, tt)
    cond[..., :9] = cond[..., :9] + 0.5*output_2
    output_3 = model(x_noisy, cond, tt)
    cond[..., :9] = cond[..., :9] + output_3
    output_4 = model(x_noisy, cond, tt)
    return (output_1+2*output_2+2*output_3+output_4)/6


def get_loss_cond_egnn(model, x_0, t, label_batch, is_gen_step = False, cutoff = cutoff):  #???

    x_0 = x_0.squeeze().to(t.device)
    label_batch = label_batch.squeeze().to(t.device)

    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    x_noisy = x_noisy.squeeze().to(t.device)
    noise = noise.squeeze().to(t.device)
    # print('x_noisy shape: ', x_noisy.shape)
    # print('noise shape: ', noise.shape)

    n_frames, n_atoms, n_features = x_0.shape
    x_coord, x_force_speed = torch.split(x_noisy, [3, 6], dim=-1)
    #x_noised_coord, x_noised_force_speed = torch.split(x_noisy, [3, 6], dim=-1)

    # coords = torch.cat((x_coord, x_noised_coord), dim=-1)
    # force_speeds = torch.cat((x_force_speed, x_noised_force_speed), dim=-1)
    # print("coord and force_speed shape: ", coords.shape, force_speeds.shape)
    # print('x_coord shape: ', x_coord.shape)
    # print('x_force_speed shape: ', x_force_speed.shape)

    # distance mtx and masks
    # dist_mtx, disp_mtx, min_idx = compute_min_distance_pbc_single_cell(x_coord, x_coord, cell_vector, cell_vector, cutoff)
    # print('dist_mtx shape: ', dist_mtx.shape)
    mask = torch.ones((n_frames, n_atoms)).to(t.device)
    # mask2d = dist_mtx < cutoff
    # currently [x_noisy, x_0] are passed to atom_feat in the model
    feat_noise_pred, coord_noise_pred = model(\
                x_force_speed, x_coord, \
                None, None, t.view(-1, 1), 
                adj_mat=None, mask=mask, mask2d=None, condition=x_0)

    noise_pred = feat_noise_pred.unsqueeze(1)
    label_batch = label_batch.unsqueeze(1)
    #noise_pred = torch.cat((coord_noise_pred, feat_noise_pred), dim=-1).unsqueeze(1) 
    x_noisy = x_noisy.unsqueeze(1)   
    noise = noise.unsqueeze(1)
    # print('x_noisy shape: ', x_noisy.shape)
    # print('noise_pred shape: ', noise_pred.shape)
    # print("gt frame shape: ", label_batch.shape)

    if is_gen_step:
        return x_noisy

    # mse_loss((x_noisy-noise_pred), label_batch , wavenum_init, lamda_reg)
    # print('noise_pred shape: ', noise_pred.shape)
    # print('label_batch shape: ', label_batch.shape)
    # print("x_noisy shape: ", x_noisy.shape)

    return mse_loss(noise, noise_pred , wavenum_init, lamda_reg)
    # mse_loss((x_noisy-noise_pred), label_batch, wavenum_init, lamda_reg)


def sample_from_egnn(model, x_noisy, cond, t, cutoff = cutoff):
    # squeeze tensors before pass to the model

    x_noisy = x_noisy.squeeze(1)
    cond = cond.squeeze(1)
    ### x_noisy[:, :, :3] = pbc_coord(x_noisy[:, :, :3], cell_vector)
    ### cond[:, :, :3] = pbc_coord(cond[:, :, :3], cell_vector)
    
    n_frames, n_atoms, n_features = x_noisy.shape
    x_coord, x_force_speed = torch.split(x_noisy, [3, 6], dim=-1)
    #dist_mtx = calc_distance(x_coord).to(t.device)
    # dist_mtx, disp_mtx, min_idx = compute_min_distance_pbc_single_cell(x_coord, x_coord, cell_vector, cell_vector, cutoff)
    mask = torch.ones((n_frames, n_atoms)).to(t.device)
    # mask2d = dist_mtx < cutoff

    feat_noise_pred, coord_noise_pred = model(\
                x_force_speed, x_coord, \
                None, None, t.view(-1, 1), 
                adj_mat=None, mask=mask, mask2d=None, condition=cond)
    x_noisy = x_noisy.unsqueeze(1)
    noise_pred = feat_noise_pred.unsqueeze(1)
    # print('x_noisy shape: ', x_noisy.shape)
    # print('noise_pred shape: ', noise_pred.shape)

    #noise_pred = torch.cat((coord_noise_pred, feat_noise_pred), dim=-1).unsqueeze(1)    
    
    return noise_pred #x_noisy-noise_pred # noise_pred


def pbc_coord(coord: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """
    Apply periodic boundary condition to the coordinates.
    Args:
        coord (torch.Tensor): The coordinates to apply PBC to. The shape is (batch_size, num_atoms, 3).
        lattice (torch.Tensor): The lattice vectors. The shape is (batch_size, 3, 3).
    Returns:
        torch.Tensor: The coordinates after applying PBC. The shape is (batch_size, num_atoms, 3).
    """
    # Calculate the fractional coordinates
    fractional_coord = torch.einsum('bji,bni->bnj', torch.inverse(lattice), coord)
    # Apply PBC
    fractional_coord = fractional_coord - torch.round(fractional_coord)
    # Convert back to Cartesian coordinates
    coord = torch.einsum('bji,bnj->bni', lattice, fractional_coord)
    return coord


def compute_min_distance_pbc_single_cell(
                    coord1: torch.Tensor,  # (batch, num_atom, 3)
                    coord2: torch.Tensor,  # (batch, num_atom, 3)
                    lattice1: torch.Tensor,  # (batch, 3, 3) lattice vectors for coord1 (GT)
                    lattice2: torch.Tensor,  # (batch, 3, 3) optional lattice for coord2 (Pred)
                    mask: torch.Tensor,
                    cutoff: float = cutoff,
                    eps: float = 1e-6,
                    return_disp: bool = False,
                    num_image_cell: int = 1) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    
    if lattice1 is None and lattice2 is None:
        return comput_disp(coord1, coord2, mask, cutoff, eps)


    coord1 = pbc_coord(coord1, lattice1)
    coord2 = pbc_coord(coord2, lattice2)

    batch_size, num_atoms, _ = coord1.shape
    # Normalize the coord (move into the unit cell)
    # if self.max_neighbor is not None:
    # Step 1: Generate relative cell displacement vectors
    # num_image_cell:1 -> (3x3x3) shift vectors
    # num_image_cell:3 -> (7x7x7) shift vectors

    shifts = torch.stack(torch.meshgrid(
        torch.arange(-num_image_cell, num_image_cell + 1),  # Neighbor cells in x
        torch.arange(-num_image_cell, num_image_cell + 1),  # Neighbor cells in y
        torch.arange(-num_image_cell, num_image_cell + 1),   # Neighbor cells in z
    indexing="ij"), dim=-1).reshape(-1, 3).type(coord1.dtype).to(coord1.device)  
    # shift = [0,0,0] index
    r_000_index = torch.where(torch.all(shifts == 0, dim=-1))[0].item()
    num_images = shifts.shape[0]
    # Apply lattice vectors to get Cartesian displacements for the 27 neighbor cells
    # The shape should be (batch_size, 27, 3), where lattice has shape (batch_size, 3, 3) and shifts has shape (27, 3)
    # print("shifts shape: ", shifts.shape)
    r_vector_1 = torch.einsum('nj, bji->bni', shifts, lattice1)  # (batch_size, 27, 3)
    r_vector_2 = torch.einsum('nj, bji->bni', shifts, lattice2)
    # print("r_vector shape: ", r_vector.shape)
        
    # Step 2: calculate distance btw original(coord1) and noised coord pbc (coord2_ext)
    # coord1_ext = coord1.unsqueeze(2) + r_vector_1.unsqueeze(1)  # (batch_size, num_atoms, n_img, 3)
    coord1_ext = coord1.unsqueeze(2) + r_vector_1.unsqueeze(1)  # (batch_size, num_atoms, n_img, 3)
    disp_1 = coord1_ext.unsqueeze(2) - coord1.unsqueeze(1).unsqueeze(3)
    coord2_ext = coord2.unsqueeze(2) + r_vector_2.unsqueeze(1)  # (batch_size, num_atoms, n_img, 3)
    disp_2 = coord2_ext.unsqueeze(2) - coord2.unsqueeze(1).unsqueeze(3)
    
    
    self_min_disp = torch.sqrt(torch.sum(disp_1 ** 2, dim=-1))  # (batch_size, num_atoms, num_atoms, n_img)
    
    #print("self min disp shape: ", self_min_disp.shape)

    min_dist, min_idx = torch.min(self_min_disp, dim=-1)  # (batch_size, num_atoms, num_atoms)
    min_disp = torch.gather(disp_1, 3,  min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3))
    
    #return disp
    return min_dist, min_disp, min_idx



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
    x_0 = x_0.squeeze(1)
    # print('x_0 shape in forward sample : ', x_0.shape)
    # print('cell vector shape in forward sample : ', cell_vector.shape)
    
    ### x_0[:, :, :3] = pbc_coord(x_0[:, :, :3], cell_vector.to(x_0.device))
    # print('x_0 shape in forward sample after pbc: ', x_0.shape)
    noise = torch.randn_like(x_0)  # sigma set to 0.25
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # x_0 = x_0.unsqueeze(1)
    # noise = noise.unsqueeze(1)  

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


def normalize_md_rev(x):
    means = np.mean(x, axis=0, keepdims=True)
    sigmas = np.std(x, axis=0, keepdims=True)

    # normalize pos
    x[..., :3] = (x[..., :3] - means[..., :3]) / sigmas[..., :3]
    means[..., 3:] = 0.
    sigmas[..., 3:] = 1.

    return x, means, sigmas


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
    d1,d2,_,d3,d4=np.shape(x)
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                for l in range(d4):
                    x[i,j,0,k,l] = x[i,j,0,k,l]*std_list[l]+mean_list[l]
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
T = args.timesteps
betas = linear_beta_schedule(timesteps=T)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
