#%%
import os
import wandb
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from models.transformer import *
from models.EGNN import *
from args import get_parser
from utils.data_loader import argon_dataset
from utils.functions import *


#%%

parser = get_parser()
args = parser.parse_args([])
print(args.root)

tr_x, te_x, tr_y, te_y, mean_list_x, std_list_x, mean_list_y, std_list_y, TRAIN_SIZE, TEST_SIZE\
     = argon_dataset(args)


#%%

device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"

root = args.root
dataset_path = args.dataset_path
batch_size = args.batch_size
temp_ = args.temperature

save_interval = args.save_interval # how often to save the model
num_epochs = args.num_epochs
learning_rate = args.learning_rate
D_TIME_STEP = args.timesteps

# Initialize wandb
# wandb.init(project='diffusionMD')
# t250_300K_pbc_less_force_2
exp_name = f'egnn_3lr_1e-4_t{args.timesteps}_{temp_}K_pbc_less_force_2'
print("exp name : ", exp_name)
# wandb.run.name = exp_name


# Define beta schedule
betas = linear_beta_schedule(timesteps=D_TIME_STEP)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
# psi_test_label_Tr = y_tensor_norm.detach().cpu().numpy()

#%%
print(alphas_cumprod_prev)
#%%

model = EGNN(in_dim=64,
            out_dim=9, # coord, force, vel output
            h_dim=128,
            num_layer=3,
            num_timesteps=D_TIME_STEP,
            update_coord='last',
            use_attention=True,
            num_head=4,
            use_condition=True,
            temperature=args.temperature,)
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, amsgrad=True)

# Training

# iter_cnt_total = 0
# for epoch in range(0, num_epochs):  # loop over the dataset multiple times
#     running_loss = 0.0
#     mean_loss = 0.0
#     iter_cnt_per_epoch = 0

#     for step in range(0,TRAIN_SIZE-batch_size,batch_size):
#         # get the inputs; data is a list of [inputs, labels]
#         indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
#         input_batch, label_batch = tr_x[indices,:,:,:], tr_y[indices,:,:,:]
        
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         t = torch.randint(0, T, (batch_size,), device=device).long()
#         loss = get_loss_cond_egnn(model, input_batch.float().cuda(), t, label_batch.float().cuda())
#         loss.backward()
#         optimizer.step()

#         wandb.log({'loss': loss}, step=iter_cnt_total)
#         indices = np.random.permutation(np.arange(start=0, stop=batch_size))
#         input_batch, label_batch = tr_x[indices,:,:,:], tr_y[indices,:,:,:]
#         val_loss = get_loss_cond_egnn(model, input_batch.float().cuda(), t, label_batch.float().cuda())
#         wandb.log({'val_loss': val_loss}, step=iter_cnt_total)
#         wandb.log({'epoch': epoch}, step=iter_cnt_total)

#         mean_loss += loss.item()
#         iter_cnt_per_epoch += 1
#         iter_cnt_total += 1
        
#     mean_loss /= iter_cnt_per_epoch
#     #print('Epoch',epoch, 'Mean Loss',mean_loss)
#     wandb.log({'mean_loss': mean_loss}, step=iter_cnt_total)
#     if epoch % save_interval == 0:
#         torch.save(model.state_dict(), os.path.join(result_path, './Diffusion_MD_trial_'+str(exp_name)+'_'+str(num_epochs)+'.pt'))
#         print('Model saved')

#%%
# laod weights
model_name = 'Diffusion_MD_trial_'+str(exp_name)+'_'+str(num_epochs)
model.load_state_dict(torch.load(os.path.join(args.result_path, model_name+'.pt')))
#model.load_state_dict(torch.load(os.path.join(result_path, './Diffusion_MD_trial_'+str(args.exp_name)+'_'+str(num_epochs)+'.pt')))
print("Loading successful")

#%%

#Inference
#Sample Nens trajectories, get the mean of the trajectories 
#device = "cpu"
T = D_TIME_STEP
Nens = 1
Nframes = 15000
Nsteps = 50 # number of experiments
d1,d2,d3,d4 = np.shape(te_x)
# pred = np.zeros([Nframes,Nsteps,Nens,1,64,9]) # 64 atoms 9 features
pred = np.zeros([Nsteps,Nens,1,64,9]) 
print(pred.shape)

#%%
def sample_one_step(model, x_noisy, condition, t):

    with torch.no_grad():
        x_noisy = x_noisy.to(device)
        condition = condition.to(device)
        t = t.to(device)

        noise_pred = sample_from_egnn(model, x_noisy, condition, t)

        return noise_pred

def sample_next_frame(model, x_noisy, condition):
    x_prev = x_noisy

    with torch.no_grad():
        for t in range(T)[::-1]:
            t_tensor = torch.tensor([t], device=device).long()
            noise_pred = sample_one_step(model, x_prev, condition, t_tensor)
            
            # Calculate the mean
            pred_mean = sqrt_recip_alphas[t] * \
                       (x_prev - betas[t] / sqrt_one_minus_alphas_cumprod[t] * noise_pred)
            
            # Add noise only for t > 0
            if t > 0:
                posterior_variance = betas[t] * (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t])
                noise = torch.randn_like(x_prev)
                x_prev = pred_mean + torch.sqrt(posterior_variance) * noise
            else:
                x_prev = pred_mean

    return x_prev

#%%
print(betas)
print(len(sqrt_alphas_cumprod))
#%%
t_temp = torch.tensor([10], device=device).long()
x_noisy, noise = forward_diffusion_sample(te_x[0,:,:,:].reshape([1,1,64,9]).float(), t_temp, device)
print(x_noisy.shape, noise.shape)

#%%
u = sample_from_egnn(model, x_noisy, te_x[0,:,:,:].reshape([1,1,64,9]).float().to(device), t_temp)
u_test = sample_next_frame(model, x_noisy, te_x[0,:,:,:].reshape([1,1,64,9]).float().to(device))
print(u_test.shape, u.shape)

#%%
for k in range(0, Nsteps):
    print('time step',k)
    if (k==0):
        for ens in range (0, Nens):
            # tt =  torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
            tt = torch.tensor([T-1], device=device).long()
            print("tt : ", tt)
            x_noisy, noise = forward_diffusion_sample(te_x[0,:,:,:].reshape([1,1,64,9]).float(), tt, device)
            u = sample_next_frame(model, x_noisy, te_x[0,:,:,:].reshape([1,1,64,9]).float().to(device))
            #u=sample_from_egnn(model, x_noisy, te_x[0,:,:,:].reshape([1,1,64,9]).float().to(device), tt)
            pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
    else:
        print(pred[k-1,:,:,:,:].shape)
        mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,1,64,9])).float()
        print(mean_traj.shape)
        for ens in range (0, Nens):
            # tt =  torch.randint(0, T, (1,), device=device).long()
            tt = torch.tensor([T-1], device=device).long()
            x_noisy, noise = forward_diffusion_sample(mean_traj, tt, device)
            #u=x_noisy - model(x_noisy, tt)
            u = sample_next_frame(model, x_noisy, mean_traj.reshape([1,1,64,9]).float().to(device))
            #u=sample_from_egnn(model, x_noisy, mean_traj.reshape([1,1,64,9]).float().to(device), tt)
            pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())



#%%
print(mean_list_y, std_list_y)
#Denormalize
print("pred and te_y shape : ", pred.shape, te_y.shape) # (1000, 100, 20, 1, 64, 9) torch.Size([1000, 1, 64, 9])
print("mean traj shape : ", np.mean(pred,1).shape)
mean_traj = torch.from_numpy(np.mean(pred,1).reshape([-1,1,1,64,9])).float()
# positions
#mean_traj[..., :3] = mean_traj[..., :3] * 21.04 + np.array(mean_list_y)[:3]
# forces and velocities
#mean_traj[..., 3:] = mean_traj[..., 3:] * np.array(std_list_y)[3:] + np.array(mean_list_y)[3:]
# pred_denorm = mean_traj * np.array(std_list_y) + np.array(mean_list_y)
pred_denorm = np.squeeze(mean_traj, 1)
print(pred_denorm.shape, te_y.shape, np.array(mean_list_y).shape, np.array(std_list_y).shape)
pred_denorm = denormalize_md(pred_denorm, mean_list_y, std_list_y)
te_y_denorm = denormalize_md(te_y, mean_list_y, std_list_y)

#%% 

np.savez(os.path.join(args.result_path, './'+model_name+'_'+str(Nsteps)+'_short.npz'), pred = pred_denorm, GT = te_y_denorm)
np.savez(os.path.join(args.result_path, './'+str(args.exp_name)+'.npz'), pred = pred, GT = te_y_denorm)
print('Saved Predictions')

#%%
