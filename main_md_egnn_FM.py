#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir('/scratch/x2895a03/research/md-diffusion/Ashesh/')
from functions_fm import *
from fm import *
from models.transformer import *
from models.EGNN import *
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data
import wandb
import argparse

#%%

# basic arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default="dataset") 
parser.add_argument("--result-path", type=str, default="results") 
parser.add_argument("--model-type", type=str, default="egnn")
parser.add_argument("--exp_name", type=str, default="egnn_3lr_3e-4_t250_FM")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize wandb
wandb.init(project='diffusionMD')
wandb.run.name = args.exp_name
exp_name = 'egnn_3lr_3e-4_t250_FM'

# options for the model and training
dataset_path = "dataset"
result_path = "results"

save_interval = 10 # how often to save the model
num_epochs = 2001
learning_rate = 3e-4 # 0.00001

#data load
#(50000, 64, 9)
x_tensor=np.load(os.path.join(dataset_path, "input.npy"))
s = np.shape(x_tensor)
# print(x_tensor.shape)
# x_tensor = np.reshape(x_tensor, (s[0]*s[1],s[2], s[3]))

d1,d2,d3 = np.shape(x_tensor)
x_tensor_norm, mean_list_x, std_list_x = normalize_md(x_tensor)
x_tensor_norm=np.reshape(x_tensor_norm, [d1,1,d2,d3])
x_tensor_norm=torch.from_numpy(x_tensor_norm)

y_tensor=np.load(os.path.join(dataset_path, "output.npy"))
s = np.shape(y_tensor)
#y_tensor = np.reshape(y_tensor, (s[0]*s[1],s[2], s[3]))
d1,d2,d3 = np.shape(y_tensor)
y_tensor_norm, mean_list_y, std_list_y = normalize_md(y_tensor)
y_tensor_norm=np.reshape(y_tensor_norm, [d1,1,d2,d3]) 
y_tensor_norm=torch.from_numpy(y_tensor_norm)

TEST_SIZE = batch_size*50 # 256 * 25 frames
TRAIN_SIZE = d1 - TEST_SIZE

tr_x, te_x =  x_tensor_norm[:-TEST_SIZE], x_tensor_norm[-TEST_SIZE:]
tr_y, te_y =  y_tensor_norm[:-TEST_SIZE], y_tensor_norm[-TEST_SIZE:]
print(tr_x.shape, tr_y.shape, te_x.shape, te_y.shape) #[batch_size, channel, n_atom, atom_feature_size]
# Define beta schedule
D_TIME_STEP = 100
sigma_min = 1.e-5


#%%

model = EGNN(in_dim=64,
            out_dim=9, # coord, force, vel output
            h_dim=64,
            num_layer=1,
            num_timesteps=D_TIME_STEP,
            update_coord='last',
            use_attention=True,
            num_head=4,
            use_condition=True,)
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, amsgrad=True)

# Training
iter_cnt_total = 0
for epoch in range(0, num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    mean_loss = 0.0
    iter_cnt_per_epoch = 0

    for step in range(0,TRAIN_SIZE-batch_size,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = tr_x[indices,:,:,:], tr_y[indices,:,:,:]
        
        # zero the parameter gradients
        optimizer.zero_grad()
        t = torch.randint(0, D_TIME_STEP, (batch_size,), device=device).long()
        loss = get_loss_cond_egnn(model, \
                                input_batch.float().cuda(), \
                                t, label_batch.float().cuda(), \
                                max_timestep=D_TIME_STEP)
        loss.backward()
        optimizer.step()

        wandb.log({'loss': loss}, step=iter_cnt_total)
        indices = np.random.permutation(np.arange(start=0, stop=batch_size))
        input_batch, label_batch = tr_x[indices,:,:,:], tr_y[indices,:,:,:]
        val_loss = get_loss_cond_egnn(model, \
                                    input_batch.float().cuda(), \
                                    t, label_batch.float().cuda(), \
                                    max_timestep=D_TIME_STEP)
        wandb.log({'val_loss': val_loss}, step=iter_cnt_total)
        wandb.log({'epoch': epoch}, step=iter_cnt_total)

        mean_loss += loss.item()
        iter_cnt_per_epoch += 1
        iter_cnt_total += 1
        
    mean_loss /= iter_cnt_per_epoch
    #print('Epoch',epoch, 'Mean Loss',mean_loss)
    wandb.log({'mean_loss': mean_loss}, step=iter_cnt_total)
    if epoch % save_interval == 0:
        torch.save(model.state_dict(), os.path.join(result_path, './Diffusion_MD_trial_'+str(args.exp_name)+'_'+str(num_epochs)+'.pt'))
        print('Model saved')

#%%
# laod weights
model_name = 'FM_MD_trial_egnn_3lr_3e-4_t250_2001'
model.load_state_dict(torch.load(os.path.join(result_path, model_name+'.pt')))
#model.load_state_dict(torch.load(os.path.join(result_path, './Diffusion_MD_trial_'+str(args.exp_name)+'_'+str(num_epochs)+'.pt')))
print("Loading successful")

#%%

#Inference
#Sample Nens trajectories, get the mean of the trajectories 
#device = "cpu"
T = D_TIME_STEP
Nens = 1
Nsteps = 1 # number of experiments
d1,d2,d3,d4 = np.shape(te_x)
pred = np.zeros([d1,Nsteps,Nens,1,64,9]) # 64 atoms 9 features
print(pred.shape)

#%%
for i in range(d1):
    for k in range(0, Nsteps):
        print(i, k ,'time step',k)
        if (k==0):
            for ens in range (0, Nens):
                tt =  torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
                pseudo_x1 = torch.randn_like(te_x[i,:,:,:].reshape([1,1,64,9]).float())
                x_noisy = stochastic_interpolant_forward(te_x[i,:,:,:].reshape([1,1,64,9]).float(), \
                                                        pseudo_x1, \
                                                        tt, D_TIME_STEP)

                u=sample_from_egnn(model, \
                                x_noisy, \
                                te_x[i,:,:,:].reshape([1,1,64,9]).float().to(device), \
                                tt, max_timestep=D_TIME_STEP)
                pred[i, k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
        else:
            mean_traj = torch.from_numpy(np.mean(pred [i, k-1,:,:,:,:],0).reshape([1,1,64,9])).float()
            for ens in range (0, Nens):
                tt =  torch.randint(0, T, (1,), device=device).long()
                pseudo_x1 = torch.randn_like(mean_traj)
                x_noisy = stochastic_interpolant_forward(mean_traj, pseudo_x1, tt, D_TIME_STEP)
                #u=x_noisy - model(x_noisy, tt)
                u=sample_from_egnn(model, x_noisy, te_x[i,:,:,:].reshape([1,1,64,9]).float().to(device), tt)
                pred[i,k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())


#Denormalize
print("pred and te_y shape : ", pred.shape, te_y.shape) # (1000, 100, 20, 1, 64, 9) torch.Size([1000, 1, 64, 9])

pred_denorm = denormalize_md_pred(pred, mean_list_y, std_list_y)
te_y_denorm = denormalize_md(te_y, mean_list_y, std_list_y)

#%% 

np.savez(os.path.join(result_path, './'+'FM_MD_trial_egnn_3lr_3e-4_t250_1000'+'.npz'), pred = pred, GT = te_y)
#np.savez(os.path.join(result_path, './'+str(args.exp_name)+'.npz'), pred = pred, GT = te_y_denorm)
print('Saved Predictions')

#%%
