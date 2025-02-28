#%%
import os
import wandb
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.args import get_parser
from utils.functions import *
from utils.sampler import *
from utils.data_loader import *
from models.transformer_multiframe import *

from torch.utils.data import DataLoader, TensorDataset

#%%

parser = get_parser()
args = parser.parse_args()
print(args.root, args.temperature, args.temperature[0])

# dataset selection
dset_generator = argon_dataset_rev
tmep_ = args.temperature
use_temp_embed = len(tmep_) >= 1

# sampling strategy
loss_calculator = get_loss_unified
next_frame_sampler = sample_one_step_unified
loss_definition = args.how_to_sample


tr_x, te_x, tr_y, te_y, \
mean_list_x, std_list_x, mean_list_y, std_list_y, \
TRAIN_SIZE, TEST_SIZE, tr_ts, te_ts \
     = dset_generator(args)
#%%
print(te_x.shape, tr_x.shape)
print(tr_ts.shape)
print(TRAIN_SIZE, TEST_SIZE)
print(mean_list_x)
print("sampler", next_frame_sampler)
print("loss_calculator", loss_calculator)
print("use normalization", args.do_norm)

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

if not args.exp_name:
    exp_name = f"md_ep{num_epochs}_unnormd_\
single_temp_{args.temperature}K_T{args.timesteps}_lr{learning_rate}_lazy_{args.how_to_sample}"
else:
    exp_name = args.exp_name
print("exp name : ", exp_name)

# wandb.init(project='diffusionMD')
# wandb.run.name = exp_name

model_save_path = os.path.join('./', args.result_path, exp_name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


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
iter_cnt_total = 0
for epoch in range(0, num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    mean_loss = 0.0
    iter_cnt_per_epoch = 0

    trainN=tr_x.shape[0]
    
    for step in range(0,trainN-batch_size,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = tr_x[indices,:,:,:], tr_y[indices,:,:,:]
        input_temp = tr_ts[indices]

        t = torch.randint(1, T, (1,), device=device).long() 
        t = t.repeat(batch_size)
        #print(t)

        x_0 = input_batch.float().to(device)
        label_batch = label_batch.float().to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()      

        loss = loss_calculator(model, input_batch.float().cuda(), t,
                             label_batch.float().cuda(), input_temp.float().cuda())

        loss.backward()
        optimizer.step()
        wandb.log({'loss': loss}, step=iter_cnt_total)
        print('Epoch',epoch, 'Step',step, 'Loss',loss)
        mean_loss += loss.item()
        iter_cnt_per_epoch += 1
        iter_cnt_total += 1

        # remove gradient info
        # del x_0, label_batch, x_noisy, noise, u, loss
    if scheduler:
        scheduler.step()
    mean_loss /= iter_cnt_per_epoch
    print('Epoch',epoch, 'Mean Loss',mean_loss)
    wandb.log({'mean_loss': mean_loss}, step=iter_cnt_total)
    if epoch % save_interval == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path, str(epoch+1)+'.pt'))
        print('Model saved')

#%%

# load weights to the model
try : 
    model.load_state_dict(torch.load(os.path.join(f'./{args.result_path}/{exp_name}/{num_epochs}'+'.pt')))
    print("model successfully loaded")
except:
    print("model not loaded")
    raise Exception("model not loaded")


#Inference
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

pred = np.zeros([Nsteps,Nens,1,64,9], dtype=np.float32) # 64 atoms 9 features

with torch.no_grad():
    for k in range(0, Nsteps):
        print('time step',k)   
        if (k==0):
            for ens in range (0, Nens):
                if "one_step" in args.how_to_sample:
                    #tt = torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
                    tt =  torch.tensor([T-1], device=device).long() # 이제 randn이니까?
                else:
                    tt =  torch.tensor([T-1], device=device).long()
                x_0 = te_x[0,:,:,:].reshape([1,1,64,9]).float().to(device)
                # generate from pure noise : 
                x_noisy, noise = forward_diffusion_sample(x_0, tt, device)
                x_noisy = x_noisy.unsqueeze(1)
                #print(x_noisy.device, x_0.device, tt.device)
                cond = x_0
                # sample
                u = next_frame_sampler(model, x_noisy, x_0, tt, temp_cond)
                pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
        else:
            mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,1,64,9])).float().to(device) 
            # choose random ensemble? error 누적될수도
            single_traj = torch.from_numpy(pred [k-1,0,:,:,:].reshape([1,1,64,9])).float().to(device)
            for ens in range (0, Nens):
                if "one_step" in args.how_to_sample:
                    # tt = torch.randint(0, T, (1,), device=device).long() # diffusion time step--> random tt 
                    tt =  torch.tensor([T-1], device=device).long()
                else:
                    tt =  torch.tensor([T-1], device=device).long()
                # generate from pure noise : 
                x_noisy, noise = forward_diffusion_sample(single_traj, tt, device)
                x_noisy = x_noisy.unsqueeze(1)
                cond = single_traj
                # sample
                u = next_frame_sampler(model, x_noisy, single_traj, tt, temp_cond)
                pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
#%%
print(pred.shape, te_y.shape, np.array(mean_list_y).shape) #(100, 20, 1, 64, 9) torch.Size([100, 1, 64, 9])
#Denormalize
if not args.do_norm:
    mean_list_y = np.array(mean_list_y)
    std_list_y = np.array(std_list_y)
    te_y = te_y.cpu().numpy()

    pred_denorm = mean_list_y[None, None, :, :, :] + pred * std_list_y[None, None, :, :, :]
    te_y_denorm = mean_list_y[None, :, :, :] + te_y * std_list_y[None, :, :, :]
else:
    pred_denorm = denormalize_md_pred(pred, mean_list_y, std_list_y)
    te_y_denorm = denormalize_md(te_y, mean_list_y, std_list_y)

np.savez(os.path.join(args.result_path, f'./single_temp/{exp_name}'),pred_denorm,te_y_denorm[:Nsteps])
print('Saved Predictions')

# %%

print(mean_list_y, mean_list_x)
print(std_list_x, std_list_y)
pred[-1][..., :3]

# %%
