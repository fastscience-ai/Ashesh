#%%
import os
import wandb
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from args import get_parser
from utils.functions import *
from utils.data_loader import *
from models.transformer import *

#%%

parser = get_parser()
args = parser.parse_args([])
print(args.root)

# dataset selection
dset_generator = None
tmep_ = args.temperature
temp_list = args.t_selection
use_temp_embed = len(temp_list) >= 1
if len(temp_list) == 0:
    dset_generator = argon_dataset
elif len(temp_list) >= 1:
    dset_generator = argon_dataset_mixed_temp
else:
    raise ValueError("Invalid temperature range")

tr_x, te_x, tr_y, te_y, \
mean_list_x, std_list_x, mean_list_y, std_list_y, \
TRAIN_SIZE, TEST_SIZE, tr_ts, te_ts \
     = dset_generator(args)
#%%
print(te_x.shape)

#%%


#torch.cuda.set_device(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# options for the model and training
root = args.root
dataset_path = args.dataset_path
batch_size = args.batch_size
save_interval = args.save_interval # how often to save the model
num_epochs = args.num_epochs
learning_rate = args.learning_rate # 0.00001

lr_milestones = None
if num_epochs > 10:
    lr_milestones = [num_epochs//3, num_epochs//3*2]

exp_name = f"main_md_ep{num_epochs}_one_step_difference_\
single_temp_{args.temperature}K_T{args.timesteps}_lr{learning_rate}_debug"
print("exp name : ", exp_name)
wandb.init(project='diffusionMD')
wandb.run.name = exp_name

model_save_path = os.path.join('./', args.result_path, exp_name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


#%%
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
# psi_test_label_Tr = y_tensor_norm.detach().cpu().numpy()

#%%
def sample_one_step(model, x_noisy, condition, t):

    with torch.no_grad():
        x_noisy = x_noisy.to(device)
        condition = condition.to(device)
        t = t.to(device)

        noise_pred = sample_from_egnn(model, x_noisy, condition, t)

        return noise_pred

def sample_next_frame(model, x_noisy, condition, tt):

    x_noisy = x_noisy.to(device)
    condition = condition.to(device)
    n_frames = x_noisy.shape[0]
    tt = tt.to(device)

    x_prev = x_noisy
    
    
    for t in range(tt[0].item())[::-1]:
        t_tensor = torch.tensor([t], device=device).long()
        t_tensor = t_tensor.repeat(n_frames)
        noise_pred = model(x_noisy, condition, t_tensor)
        
        # Calculate the mean
        pred_mean = sqrt_recip_alphas[t] * \
                    (x_prev - betas[t] / sqrt_one_minus_alphas_cumprod[t] * noise_pred)
        
        # Add noise only for t > 0
        if t > 0:
            posterior_variance = betas[t] * (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t])
            noise = torch.randn_like(x_prev) * 0.5
            x_prev = pred_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = pred_mean

    return x_prev
#%%
# v_0 = tr_x[0, 0, 0][3:6]
# x_0 = tr_x[0, 0, 0][:3]
# x_1 = tr_x[1, 0, 0][:3]
# print(x_0, v_0)

# x_noisy, noise = forward_diffusion_sample(tr_x[0:10], torch.tensor([299]), device)
# print(x_noisy.shape)
# print(x_noisy[0, 0, :3])
# print(torch.mean(x_noisy[:, 0, :3]), torch.mean(tr_x[:10][..., :3]))
# print(torch.mean(x_noisy[:, 0, 3:6]), torch.mean(tr_x[:10][..., 3:6]))
# print(torch.mean(x_noisy[:, 0, -3:]), torch.mean(tr_x[:10][..., -3:]))

# print(torch.std(x_noisy[:, 0, :3]), torch.std(tr_x[:10][..., :3]))
# print(torch.std(x_noisy[:, 0, 3:6]), torch.std(tr_x[:10][..., 3:6]))
# print(torch.std(x_noisy[:, 0, -3:]), torch.std(tr_x[:10][..., -3:]))
# #%%
# print(torch.tensor([200]*10))

#%%

#model = SimpleUnet()
model = Transformer(num_atoms=64, input_size=9, dim=64,
                 depth=3, heads=4, mlp_dim=256, k=64, in_channels=1,
                 use_temp_embed=use_temp_embed)
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = None
if lr_milestones:
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

#%%
# iter_cnt_total = 0
# for epoch in range(0, num_epochs):  # loop over the dataset multiple times
#     running_loss = 0.0
#     mean_loss = 0.0
#     iter_cnt_per_epoch = 0

#     trainN=tr_x.shape[0]
    
#     for step in range(0,trainN-batch_size,batch_size):
#         # get the inputs; data is a list of [inputs, labels]
#         indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
#         input_batch, label_batch = tr_x[indices,:,:,:], tr_y[indices,:,:,:]
#         temp_batch = tr_ts[indices]

#         t = torch.randint(1, T, (1,), device=device).long() 
#         t = t.repeat(batch_size)
#         #print(t)

#         x_0 = input_batch.float().to(device)
#         label_batch = label_batch.float().to(device)
        
#         # zero the parameter gradients
#         optimizer.zero_grad()

#         if use_temp_embed:
#             temps = temp_batch.float().to(device)        

#         loss = get_loss_cond_diff(model, input_batch.float().cuda(), t,
#                              label_batch.float().cuda(), temps)
        
#         # TODO : Implement recursive sampling logic in functions.py
#         # recursive sampling 
#         # x_noisy, noise = forward_diffusion_sample(x_0, t, device)
#         # t = torch.randint(0, T, (batch_size,), device=device).long()
#         # x_noisy = x_noisy.unsqueeze(1)
#         # noise = noise.unsqueeze(1)
#         # u = sample_next_frame(model, x_noisy, x_0, t)
#         # loss =  F.mse_loss(u,label_batch) 

#         loss.backward()
#         optimizer.step()
#         wandb.log({'loss': loss}, step=iter_cnt_total)
#         print('Epoch',epoch, 'Step',step, 'Loss',loss)
#         mean_loss += loss.item()
#         iter_cnt_per_epoch += 1
#         iter_cnt_total += 1

#         # remove gradient info
#         # del x_0, label_batch, x_noisy, noise, u, loss
#     if scheduler:
#         scheduler.step()
#     mean_loss /= iter_cnt_per_epoch
#     print('Epoch',epoch, 'Mean Loss',mean_loss)
#     wandb.log({'mean_loss': mean_loss}, step=iter_cnt_total)
#     if epoch % save_interval == 0:
#         torch.save(model.state_dict(), os.path.join(model_save_path, str(epoch+1)+'.pt'))
#         print('Model saved')

#%%

# load weights to the model
try : 
    model.load_state_dict(torch.load(os.path.join(f'./{args.result_path}/{exp_name}/{num_epochs}'+'.pt')))
    print("model successfully loaded")
except:
    print("model not loaded")
    raise Exception("model not loaded")


#Inference
Nens = 20
Nsteps = 1000 # number of experiments
t_to_simulate = args.t_to_simulate

pred = np.zeros([Nsteps,Nens,1,64,9]) # 64 atoms 9 features
with torch.no_grad():
    temp = None
    idx_to_start = None
    if use_temp_embed:
        temp = torch.tensor([t_to_simulate], device=device).float() / 300. # normalize
        temp = temp.reshape(*temp.shape, 1, 1, 1).expand(*temp.shape, 1, 64, 1)
        eps = 0.8
        idx_to_start = torch.where(torch.abs(te_ts.float() - (t_to_simulate / 300.)) < eps)[0][0]
    
    for k in range(0, Nsteps):
        print('time step',k)   
        if (k==0):
            for ens in range (0, Nens):
                #tt =  torch.tensor([1], device=device).long()
                tt = torch.randint(0, T//10, (1,), device=device).long() # diffusion time step--> random tt 
                x_0 = te_x[idx_to_start,:,:,:].reshape([1,1,64,9]).float().to(device)
                x_noisy, noise = forward_diffusion_sample(x_0, tt, device)
                x_noisy = x_noisy.unsqueeze(1)
                noise = noise.unsqueeze(1)
                #print(x_noisy.device, x_0.device, tt.device)
                if temp is not None:
                    x_0 = torch.cat([x_0, temp], dim=-1)
                #u=x_0[..., :9] + RK4_sampler(model, x_noisy, x_0, tt)
                # model(x_noisy, x_0, tt) # * betas[tt.item()] / sqrt_one_minus_alphas_cumprod[tt.item()]
                u = sample_next_frame(model, x_noisy, x_0, tt)
                pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
        else:
            mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,1,64,9])).float().to(device) 
            # single traj prediction : just change to mean -> single
            single_traj = torch.from_numpy(pred [k-1,0,:,:,:].reshape([1,1,64,9])).float().to(device)
            # choose random ensemble? error 누적될수도
            for ens in range (0, Nens):
                tt =  torch.randint(0, T//10, (1,), device=device).long()
                x_noisy, noise = forward_diffusion_sample(mean_traj, tt, device)
                if temp is not None :
                    cond = torch.cat([single_traj, temp], dim=-1)
                x_noisy = x_noisy.unsqueeze(1)
                noise = noise.unsqueeze(1)
                # u=mean_traj[..., :9] + RK4_sampler(model, x_noisy, cond, tt)
                # model(x_noisy, cond, tt) # * betas[tt.item()] / sqrt_one_minus_alphas_cumprod[tt.item()]
                u = sample_next_frame(model, x_noisy, mean_traj, tt)
                pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
#%%
print(pred.shape, te_y.shape, np.array(mean_list_y).shape) #(100, 20, 1, 64, 9) torch.Size([100, 1, 64, 9])
#Denormalize
print(pred.shape, te_y.shape) #(100, 20, 1, 64, 9) torch.Size([100, 1, 64, 9])
pred_denorm = denormalize_md_pred(pred, mean_list_y, std_list_y)
te_y_denorm = denormalize_md(te_y, mean_list_y, std_list_y)
np.savez(os.path.join(args.result_path, f'./{exp_name}'),pred,te_y_denorm[:Nsteps])
print('Saved Predictions')

# %%

