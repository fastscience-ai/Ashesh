import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from functions import *
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data
import wandb
#torch.cuda.set_device(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print("\n\n\n")
wandb.init(project='diffusionMD')


#Data normalization
psi_test_input_Tr_torch, psi_test_label_Tr_torch  = load_test_data(FF,lead)
M_test_level1=torch.mean((psi_test_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))
STD_test_level1=torch.std((psi_test_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))
M_test_level2=torch.mean((psi_test_input_Tr_torch[:,1,0:Nlat,0:Nlon].flatten()))
STD_test_level2=torch.std((psi_test_input_Tr_torch[:,1,0:Nlat,0:Nlon].flatten()))
M_test_level3=torch.mean((psi_test_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))
STD_test_level3=torch.std((psi_test_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))
psi_test_input_Tr_torch_norm_level1 = ((psi_test_input_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1)
psi_test_label_Tr_torch_norm_level1 = ((psi_test_label_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1)
psi_test_input_Tr_torch_norm_level2 = ((psi_test_input_Tr_torch[:,1,None,:,:]-M_test_level2)/STD_test_level2)
psi_test_label_Tr_torch_norm_level2 = ((psi_test_label_Tr_torch[:,1,None,:,:]-M_test_level2)/STD_test_level2)
psi_test_input_Tr_torch_norm_level3 = ((psi_test_input_Tr_torch[:,2,None,:,:]-M_test_level3)/STD_test_level3)
psi_test_label_Tr_torch_norm_level3 = ((psi_test_label_Tr_torch[:,2,None,:,:]-M_test_level3)/STD_test_level3)
psi_test_input_Tr_torch_norm = torch.cat((psi_test_input_Tr_torch_norm_level1,psi_test_input_Tr_torch_norm_level2,psi_test_input_Tr_torch_norm_level3),1)
psi_test_label_Tr_torch_norm = torch.cat((psi_test_label_Tr_torch_norm_level1,psi_test_label_Tr_torch_norm_level2,psi_test_label_Tr_torch_norm_level3),1)

print('mean value',M_test_level1)
print('std value',STD_test_level1)
print('mean value',M_test_level2)
print('std value',STD_test_level2)
print('mean value',M_test_level3)
print('std value',STD_test_level3)

    
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
psi_test_label_Tr = psi_test_label_Tr_torch.detach().cpu().numpy()

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
fileList_train=[]
fileList_train.append('./moistQG/151/output.3d-001.nc')

for epoch in range(0, num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for k in fileList_train:
        print('File index',k)
        trainN=9000
        psi_train_input_Tr_torch, psi_train_label_Tr_torch  = load_train_data(nc.Dataset(k),lead,trainN)
        M_train_level1=torch.mean((psi_train_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))
        STD_train_level1=torch.std((psi_train_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))
        M_train_level2=torch.mean((psi_train_input_Tr_torch[:,1,0:Nlat,0:Nlon].flatten()))
        STD_train_level2=torch.std((psi_train_input_Tr_torch[:,1,:,:].flatten()))
        M_train_level3=torch.mean((psi_train_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))
        STD_train_level3=torch.std((psi_train_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))
        psi_train_input_Tr_torch_norm_level1 = ((psi_train_input_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)
        psi_train_label_Tr_torch_norm_level1 = ((psi_train_label_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)
        psi_train_input_Tr_torch_norm_level2 = ((psi_train_input_Tr_torch[:,1,None,:,:]-M_train_level2)/STD_train_level2)
        psi_train_label_Tr_torch_norm_level2 = ((psi_train_label_Tr_torch[:,1,None,:,:]-M_train_level2)/STD_train_level2)
        psi_train_input_Tr_torch_norm_level3 = ((psi_train_input_Tr_torch[:,2,None,:,:]-M_train_level3)/STD_train_level3)
        psi_train_label_Tr_torch_norm_level3 = ((psi_train_label_Tr_torch[:,2,None,:,:]-M_train_level3)/STD_train_level3)
        psi_train_input_Tr_torch_norm = torch.cat((psi_train_input_Tr_torch_norm_level1,psi_train_input_Tr_torch_norm_level2,psi_train_input_Tr_torch_norm_level3),1)
        psi_train_label_Tr_torch_norm = torch.cat((psi_train_label_Tr_torch_norm_level1,psi_train_label_Tr_torch_norm_level2,psi_train_label_Tr_torch_norm_level3),1)
        for step in range(0,trainN-batch_size,batch_size):
            # get the inputs; data is a list of [inputs, labels]
            indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
            input_batch, label_batch = psi_train_input_Tr_torch_norm[indices,:,:,:], psi_train_label_Tr_torch_norm[indices,:,:,:]
            print('shape of input', input_batch.shape)
            print('shape of label1', label_batch.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch_size,), device=device).long()
            loss = get_loss_cond(model, input_batch.float().cuda(), t, label_batch.float().cuda())
            loss.backward()
            optimizer.step()
            wandb.log({'epoch': epoch, 'loss': loss})
            print('Epoch',epoch, 'Step',step, 'Loss',loss)
    torch.save(model.state_dict(), './Diffusion_FFT_spectralloss_lead'+str(lead)+'.pt')
    print('Model saved')


psi_test_label_Tr_torch_denorm = psi_test_label_Tr_torch_norm_level1*STD_test_level1+M_test_level1
psi_test_label_Tr = psi_test_label_Tr_torch.detach().cpu().numpy()
Nens = 20
Nsteps = 500
pred = np.zeros([Nsteps,Nens,3,Nx,Nx])
for k in range(0,Nsteps):
 print('time step',k)   
 if (k==0):
   for ens in range (0,Nens):
    tt =  torch.randint(0, T, (1,), device=device).long()
    x_noisy, noise = forward_diffusion_sample(psi_test_input_Tr_torch_norm[0,:,:,:].reshape([1,3,Nx,Ny]).float(), tt, device)
    u=x_noisy - model(x_noisy, tt)
    pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
  
 else:
   mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,3,Nx,Ny])).float() 
   for ens in range (0, Nens):
     tt =  torch.randint(0, T, (1,), device=device).long()
     x_noisy, noise = forward_diffusion_sample(mean_traj, tt, device)
     u=x_noisy - model(x_noisy, tt)
     pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
 

STD_test_level1=STD_test_level1.detach().cpu().numpy()
M_test_level1=M_test_level1.detach().cpu().numpy()
STD_test_level2=STD_test_level2.detach().cpu().numpy()
M_test_level2=M_test_level2.detach().cpu().numpy()
STD_test_level3=STD_test_level3.detach().cpu().numpy()
M_test_level3=M_test_level3.detach().cpu().numpy()

pred_denorm1 = pred [:,:,0,None,:,:]*STD_test_level1+M_test_level1
pred_denorm2 = pred [:,:,1,None,:,:]*STD_test_level2+M_test_level2
pred_denorm3 = pred [:,:,2,None,:,:]*STD_test_level3+M_test_level3

pred = np.concatenate((pred_denorm1,pred_denorm2,pred_denorm3),axis=2)
np.savez(path_outputs+'predicted_QG_spectral_loss_diffusion_lamda_'+str(lamda_reg),pred,psi_test_label_Tr)
print('Saved Predictions')
