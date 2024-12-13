import matplotlib.pyplot as plt
import numpy as np


# unconditional diffusion model
uncond=np.load('./results/egnn_3layer_traj_lr_3e-4.npz')
uncond_pre = uncond['arr_0']
uncond_gt = uncond['arr_1']
uncond_pre_ave = np.mean(uncond_pre, 2)

# averaged draw
d1, d2, d3, d4 = np.shape(uncond_gt)

for idx in range(10): # Only 10 Experiments
    #idx = np.random.randint(0, d1-1)
    for diff_t in range(100):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(d3):
            if i == d3 -1:
                ax.scatter(uncond_gt[idx,0,i,0],uncond_gt[idx,0,i,1],uncond_gt[idx,0,i,2], marker='o', c='black', label="GT")
            else:    
                ax.scatter(uncond_gt[idx,0,i,0],uncond_gt[idx,0,i,1],uncond_gt[idx,0,i,2], marker='o', c='black')
        for i in range(d3):
            if i == d3 -1:
                ax.scatter(uncond_pre_ave[idx,diff_t,0,i,0],uncond_pre_ave[idx,diff_t,0,i,1],uncond_pre_ave[idx,diff_t,0,i,2], marker='*', c='red', label="Pre")
            else:
                ax.scatter(uncond_pre_ave[idx,diff_t,0,i,0],uncond_pre_ave[idx,diff_t,0,i,1],uncond_pre_ave[idx,diff_t,0,i,2], marker='*', c='red')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()
        ax.title("uncond_#"+str(idx)+"_diff_tt_#"+str(diff_t))
        plt.savefig("./results/uncond_averaged/uncond_#"+str(idx)+"_diff_tt_#"+str(diff_t)+".png")    
        plt.close()


#Sampled Draw

d1, d2, d3, d4 = np.shape(uncond_gt) #(100, 1, 64, 9))
#(1000, 100, 20, 1, 64, 9): uncond_pre
for idx in range(10): # Only 10 Experiments
    #idx = np.random.randint(0, d1-1)
    for diff_t in range(100):
        for s in range(20):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i in range(d3):
                if i == d3 -1:
                    ax.scatter(uncond_gt[idx,0,i,0],uncond_gt[idx,0,i,1],uncond_gt[idx,0,i,2], marker='o', c='black', label="GT")
                else:    
                    ax.scatter(uncond_gt[idx,0,i,0],uncond_gt[idx,0,i,1],uncond_gt[idx,0,i,2], marker='o', c='black')
            for i in range(d3):
                if i == d3 -1:
                    ax.scatter(uncond_pre[idx,diff_t,s,0,i,0],uncond_pre[idx,diff_t,s,0,i,1],uncond_pre[idx,diff_t,s,0,i,2], marker='*', c='red', label="Pre")
                else:
                    ax.scatter(uncond_pre[idx,diff_t,s,0,i,0],uncond_pre[idx,diff_t,s,0,i,1],uncond_pre[idx,diff_t,s,0,i,2], marker='*', c='red')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.legend()
            ax.title("uncond_#"+str(idx)+"sample_#"+str(s)+"_diff_tt_#"+str(diff_t))
            plt.savefig("./results/uncond_sampled/uncond_#"+str(idx)+"sample_#"+str(s)+"_diff_tt_#"+str(diff_t)+".png")    
            plt.close()
