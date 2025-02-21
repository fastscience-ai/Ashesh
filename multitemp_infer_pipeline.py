#%%
import os
import torch
import numpy as np

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.args import get_parser
from utils.functions import *
from utils.data_loader import *
from utils.sampler import *
from models.transformer_multiframe import *
from torch.utils.data import DataLoader, TensorDataset

#%%

parser = get_parser()
args = parser.parse_args([])
print(args.root, args.temperature, args.temperature[0])

# dataset selection
dset_generator = None
temp_ = args.temperature
use_temp_embed = len(temp_) >= 1

dset_generator = argon_dataset_rev

# sampling strategy
loss_calculator = None
next_frame_sampler = None

# Manul sampler selection
args.how_to_sample = "one_step"

if args.how_to_sample == "one_step":
    loss_calculator = get_loss_cond
    next_frame_sampler = sample_one_step
elif args.how_to_sample == "next_frame":
    loss_calculator = get_loss_cond_rev
    next_frame_sampler = sample_next_frame
elif args.how_to_sample == "direct":
    loss_calculator = get_loss_cond_direct
    next_frame_sampler = None
else:
    raise ValueError("Invalid choice of sampling method! Choose one_step or next_frame")

print("sampler", next_frame_sampler," temperature : ", temp_)

tr_x, te_x, tr_y, te_y, \
mean_list_x, std_list_x, mean_list_y, std_list_y, \
TRAIN_SIZE, TEST_SIZE, tr_ts, te_ts \
     = dset_generator(args)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(num_atoms=64, input_size=9, dim=64,
                 depth=3, heads=4, mlp_dim=256, k=64, in_channels=1,
                 use_temp_embed=use_temp_embed)
model = model.to(device)

num_epochs = args.num_epochs
exp_name = "md_ep2501_one_step_noise_\
    single_temp_[300, 500, 1000]K_T300_lr0.001"
num_epochs = 2501

try : 
    model.load_state_dict(torch.load(os.path.join(f'./{args.result_path}/{exp_name}/{num_epochs}'+'.pt')))
    print("model successfully loaded")
except:
    print("model not loaded")
    raise Exception("model not loaded")


#%%
Nens = 1
Nsteps = 2000 # number of experiments
t_to_simulate = args.t_to_simulate

if t_to_simulate:
    # find the temperature index
    temp_diff = te_ts - t_to_simulate
    temp_idx = np.argmin(np.abs(temp_diff))
    temp_cond = te_ts[temp_idx].unsqueeze(0).float().to(device)
else:
    temp_cond = te_ts[0].unsqueeze(0).float().to(device)

pred = np.zeros([Nsteps,Nens,1,64,9]) # 64 atoms 9 features

with torch.no_grad():
    for k in range(0, Nsteps):
        print('time step',k)   
        if (k==0):
            for ens in range (0, Nens):
                if args.how_to_sample == "one_step":
                    #tt = torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
                    tt =  torch.tensor([T-1], device=device).long() # 이제 randn이니까?
                else:
                    tt =  torch.tensor([T-1], device=device).long()
                x_0 = te_x[0,:,:,:].reshape([1,1,64,9]).float().to(device)
                # generate from pure noise : 
                x_noisy = torch.randn_like(x_0)
                #print(x_noisy.device, x_0.device, tt.device)
                cond = x_0
                # sample
                if next_frame_sampler:
                    u = next_frame_sampler(model, x_noisy, x_0, tt, temp_cond)
                else:
                    u = model(x_noisy, x_0, tt, temp_cond)
                pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
        else:
            mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,1,64,9])).float().to(device) 
            # choose random ensemble? error 누적될수도
            single_traj = torch.from_numpy(pred [k-1,0,:,:,:].reshape([1,1,64,9])).float().to(device)
            for ens in range (0, Nens):
                if args.how_to_sample == "one_step":
                    # tt = torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
                    tt =  torch.tensor([T-1], device=device).long()
                else:
                    tt =  torch.tensor([T-1], device=device).long()
                # generate from pure noise : 
                x_noisy = torch.randn_like(x_0)
                cond = single_traj
                # sample
                if next_frame_sampler:
                    u = next_frame_sampler(model, x_noisy, single_traj, tt, temp_cond)
                else:
                    u = model(x_noisy, single_traj, tt, temp_cond)
                pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())

#%%
print(pred.shape, te_y.shape, np.array(mean_list_y).shape) #(100, 20, 1, 64, 9) torch.Size([100, 1, 64, 9])
#Denormalize
print(pred.shape, te_y.shape) #(100, 20, 1, 64, 9) torch.Size([100, 1, 64, 9])
pred_denorm = denormalize_md_pred(pred, mean_list_y, std_list_y)
te_y_denorm = denormalize_md(te_y, mean_list_y, std_list_y)
np.savez(os.path.join(args.result_path, f'./{exp_name}_{args.how_to_sample}_{temp_}K'),pred,te_y_denorm[:Nsteps])
print('Saved Predictions')
# %%
