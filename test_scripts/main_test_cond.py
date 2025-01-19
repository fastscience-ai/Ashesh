import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import *
from transformer import *
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data
#torch.cuda.set_device(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# options for the model and training
dataset_path = "dataset"
result_path = "results"

save_interval = 10 # how often to save the model
num_epochs = 500000
learning_rate = 1e-3 # 0.00001

#data load
#(50000, 64, 9)
x_tensor=np.load(os.path.join(dataset_path, "input.npy"))
s = np.shape(x_tensor)
x_tensor = np.reshape(x_tensor, (s[0]*s[1],s[2], s[3]))

d1,d2,d3 = np.shape(x_tensor)
x_tensor_norm, mean_list_x, std_list_x = normalize_md(x_tensor)
x_tensor_norm=np.reshape(x_tensor_norm, [d1,1,d2,d3])
x_tensor_norm=torch.from_numpy(x_tensor_norm)
y_tensor=np.load(os.path.join(dataset_path, "output.npy"))
s = np.shape(y_tensor)
y_tensor = np.reshape(y_tensor, (s[0]*s[1],s[2], s[3]))
d1,d2,d3 = np.shape(y_tensor)
y_tensor_norm, mean_list_y, std_list_y = normalize_md(y_tensor)
y_tensor_norm=np.reshape(y_tensor_norm, [d1,1,d2,d3]) 
y_tensor_norm=torch.from_numpy(y_tensor_norm)

tr_x, te_x =  x_tensor_norm[:-1000], x_tensor_norm[-1000:]
tr_y, te_y =  y_tensor_norm[:-1000], y_tensor_norm[-1000:]
print(tr_x.shape, tr_y.shape, te_x.shape, te_y.shape) #[batch_size, channel, n_atom, atom_feature_size]
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
psi_test_label_Tr = y_tensor_norm.detach().cpu().numpy()

#model = SimpleUnet()
model = Transformer(num_atoms=64, input_size=9, dim=64,
                 depth=3, heads=4, mlp_dim=512, k=64, in_channels=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load("./results/Diffusion_MD_trial_5000.pt"))
model.to(device)


print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)
print(np.shape(te_x), np.shape(te_y)) #torch.Size([1000, 1, 64, 9]) torch.Size([1000, 1, 64, 9])

#Inference
T = 300
Nens = 20
Nsteps = 100 # number of experiments
d1,d2,d3,d4 = np.shape(te_x)
pred = np.zeros([d1,Nsteps,Nens,1,64,9]) # 64 atoms 9 features
for i in range(d1):
    for k in range(0, Nsteps):
        print(i, k ,'time step',k)
        if (k==0):
            for ens in range (0, Nens):
                tt =  torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
                x_noisy, noise = forward_diffusion_sample(te_x[i,:,:,:].reshape([1,1,64,9]).float(), tt, device)
                predicted_noisy =  model(x_noisy,te_x[i,:,:,:].reshape([1,1,64,9]).float().to(device),  tt)
                u=x_noisy - predicted_noisy
                pred[i, k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
        else:
            mean_traj = torch.from_numpy(np.mean(pred [i, k-1,:,:,:,:],0).reshape([1,1,64,9])).float()
            for ens in range (0, Nens):
                tt =  torch.randint(0, T, (1,), device=device).long()
                x_noisy, noise = forward_diffusion_sample(mean_traj, tt, device)
                #u=x_noisy - model(x_noisy, tt)
                predicted_noisy =  model(x_noisy, te_x[i,:,:,:].reshape([1,1,64,9]).float().to(device),  tt)
                u=x_noisy - predicted_noisy
                pred[i,k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())


#Denormalize
print(pred.shape, te_y.shape) # (1000, 100, 20, 1, 64, 9) torch.Size([1000, 1, 64, 9])

pred_denorm = denormalize_md_pred(pred, mean_list_y, std_list_y)
te_x_denorm = denormalize_md(te_x, mean_list_x, std_list_x)
te_y_denorm = denormalize_md(te_y, mean_list_y, std_list_y)
np.savez(os.path.join(result_path, './predicted_diffusion_cond'),pred,te_y_denorm[:Nsteps], te_x_denorm[:Nsteps])
print('Saved Predictions')
