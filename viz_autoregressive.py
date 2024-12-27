import matplotlib.pyplot as plt
import numpy as np
import os

#os.mkdir("./results/uncond_averaged/")
#os.mkdir("./results/uncond_sampled/")
os.mkdir("./results/cond_long_averaged/")
os.mkdir("./results/cond_long_sampled/")
#os.mkdir("./results/uncond_averaged_train/")

# conditional diffusion model
#Conditional diffusion Model
cond=np.load('./results/predicted_diffusion_cond.npz')
cond_pre, cond_gt, cond_x = cond['arr_0'], cond['arr_1'], cond['arr_2']
cond_pre_ave = np.mean(cond_pre, 2)

print(np.shape(cond_pre), np.shape(cond_gt), np.shape(cond_x))


# Conditional Average Draw
d1, d2, d3, d4 = np.shape(cond_gt)

# (10,1000, 20, 1, 64, 9) torch.Size([10,1000, 64, 9])


for idx in range(d1): # Only 10 Experiments
    #idx = np.random.randint(0, d1-1)
    for diff_t in range(d2):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(d3):
            if i == d3 -1:
                ax.scatter(cond_gt[idx,diff_t,i,0],cond_gt[idx,diff_t,i,1],cond_gt[idx,diff_t,i,2], marker='o', c='black', label="GT")
            else:    
                ax.scatter(cond_gt[idx,diff_t,i,0],cond_gt[idx,diff_t,i,1],cond_gt[idx,diff_t,i,2], marker='o', c='black')
        for i in range(d3):
            if i == d3 -1:
                ax.scatter(cond_pre_ave[idx,diff_t,0,i,0],cond_pre_ave[idx,diff_t,0,i,1],cond_pre_ave[idx,diff_t,0,i,2], marker='*', c='red', label="Pre")
            else:
                ax.scatter(cond_pre_ave[idx,diff_t,0,i,0],cond_pre_ave[idx,diff_t,0,i,1],cond_pre_ave[idx,diff_t,0,i,2], marker='*', c='red')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()
        ax.set_title("uncond_#"+str(idx)+"_diff_tt_#"+str(diff_t))
        plt.savefig("./results/cond_long_averaged/uncond_#"+str(idx)+"_diff_tt_#"+str(diff_t)+".png")    
        plt.close()


#Sampled Draw

d1, d2, d3, d4 = np.shape(cond_gt) # (10,1000, 20, 1, 64, 9) torch.Size([10,1000,  64, 9])
#
for idx in range(d1): # Only 10 Experiments
    #idx = np.random.randint(0, d1-1)
    for diff_t in range(d2):
        for s in range(20):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i in range(d3):
                if i == d3 -1:
                    ax.scatter(cond_gt[idx,diff_t,i,0],cond_gt[idx,diff_t,i,1],cond_gt[idx,diff_t,i,2], marker='o', c='black', label="GT")
                else:    
                    ax.scatter(cond_gt[idx,diff_T,i,0],cond_gt[idx,diff_t,i,1],cond_gt[idx,diff_t,i,2], marker='o', c='black')
            for i in range(d3):
                if i == d3 -1:
                    ax.scatter(cond_pre[idx,diff_t,s,0,i,0],cond_pre[idx,diff_t,s,0,i,1],cond_pre[idx,diff_t,s,0,i,2], marker='*', c='red', label="Pre")
                else:
                    ax.scatter(cond_pre[idx,diff_t,s,0,i,0],cond_pre[idx,diff_t,s,0,i,1],cond_pre[idx,diff_t,s,0,i,2], marker='*', c='red')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.legend()
            ax.set_title("uncond_#"+str(idx)+"sample_#"+str(s)+"_diff_tt_#"+str(diff_t))
            plt.savefig("./results/cond_sampled/uncond_#"+str(idx)+"sample_#"+str(s)+"_diff_tt_#"+str(diff_t)+".png")    
            plt.close()

