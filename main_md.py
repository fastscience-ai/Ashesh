import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import *
from transformer import *
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data
import wandb
#torch.cuda.set_device(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb.init(project='diffusionMD')

# options for the model and training
dataset_path = "dataset"
result_path = "results"

save_interval = 10 # how often to save the model
num_epochs = 500000
learning_rate = 1e-3 # 0.00001

#data load
#(50000, 64, 9)
x_tensor=np.load(os.path.join(dataset_path, "input.npy"))
d1,d2,d3 = np.shape(x_tensor)
x_tensor_norm, mean_list_x, std_list_x = normalize_md(x_tensor)
x_tensor_norm=np.reshape(x_tensor_norm, [d1,1,d2,d3])
x_tensor_norm=torch.from_numpy(x_tensor_norm)
y_tensor=np.load(os.path.join(dataset_path, "output.npy"))
d1,d2,d3 = np.shape(y_tensor)
y_tensor_norm, mean_list_y, std_list_y = normalize_md(y_tensor)
y_tensor_norm=np.reshape(y_tensor_norm, [d1,1,d2,d3]) 
y_tensor_norm=torch.from_numpy(y_tensor_norm)

tr_x, te_x =  x_tensor_norm[:-100], x_tensor_norm[-100:]
tr_y, te_y =  y_tensor_norm[:-100], y_tensor_norm[-100:]
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
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)


iter_cnt_total = 0
for epoch in range(0, num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    mean_loss = 0.0
    iter_cnt_per_epoch = 0
    for k in ["file_1"]:
        print('File index',k)
        trainN=5000
        
        
        for step in range(0,trainN-batch_size,batch_size):
            # get the inputs; data is a list of [inputs, labels]
            indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
            input_batch, label_batch = tr_x[indices,:,:,:], tr_y[indices,:,:,:]
            # zero the parameter gradients
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch_size,), device=device).long()
            loss = get_loss_cond(model, input_batch.float().cuda(), t, label_batch.float().cuda())
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss}, step=iter_cnt_total)
            print('Epoch',epoch, 'Step',step, 'Loss',loss)
            mean_loss += loss.item()
            iter_cnt_per_epoch += 1
            iter_cnt_total += 1
        
    mean_loss /= iter_cnt_per_epoch
    print('Epoch',epoch, 'Mean Loss',mean_loss)
    wandb.log({'mean_loss': mean_loss}, step=iter_cnt_total)
    if epoch % save_interval == 0:
        torch.save(model.state_dict(), os.path.join(result_path, './Diffusion_MD_trial'+str(num_epochs)+'.pt'))
        print('Model saved')

#Inference
Nens = 20
Nsteps = 100 # number of experiments
pred = np.zeros([Nsteps,Nens,1,64,9]) # 64 atoms 9 features
for k in range(0, Nsteps):
    print('time step',k)   
    if (k==0):
        for ens in range (0, Nens):
            tt =  torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
            x_noisy, noise = forward_diffusion_sample(te_x[0,:,:,:].reshape([1,1,64,9]).float(), tt, device)
            u=x_noisy - model(x_noisy, tt)
            pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
    else:
        mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,1,64,9])).float() 
        for ens in range (0, Nens):
            tt =  torch.randint(0, T, (1,), device=device).long()
            x_noisy, noise = forward_diffusion_sample(mean_traj, tt, device)
            u=x_noisy - model(x_noisy, tt)
            pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())

#Denormalize
print(pred.shape, te_y.shape) #(100, 20, 1, 64, 9) torch.Size([100, 1, 64, 9])
pred_denorm = denormalize_md_pred(pred, mean_list_y, std_list_y)
te_y_denorm = denormalize_md(te_y, mean_list_y, std_list_y)
np.savez(os.path.join(result_path, './predicted_diffusion_md_'+str(lamda_reg)),pred,te_y_denorm[:Nsteps])
print('Saved Predictions')
