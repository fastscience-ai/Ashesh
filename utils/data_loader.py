import os
import argparse
import numpy as np

import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

from .functions import *



def argon_dataset(args):
    root = args.root
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    temp_ = args.temperature

    x_tensor=np.load(os.path.join(root, dataset_path, f"input_long_{temp_}K.npy"))[1:]
    s = np.shape(x_tensor)
    print(x_tensor.shape)
    x_tensor = np.reshape(x_tensor, (s[0]*s[1],s[2], s[3])) # (s[0], s[1],s[2])) #

    d1,d2,d3 = np.shape(x_tensor)
    x_tensor_norm, mean_list_x, std_list_x = normalize_md(x_tensor)
    x_tensor_norm=np.reshape(x_tensor_norm, [d1,1,d2,d3])
    x_tensor_norm=torch.from_numpy(x_tensor_norm)

    y_tensor=np.load(os.path.join(root, dataset_path, f"output_long_{temp_}K.npy"))[1:]
    s = np.shape(y_tensor)
    print(y_tensor.shape)
    y_tensor = np.reshape(y_tensor, (s[0]*s[1],s[2], s[3])) # (s[0], s[1],s[2]))# 
    d1,d2,d3 = np.shape(y_tensor)
    y_tensor_norm, mean_list_y, std_list_y = normalize_md(y_tensor)
    y_tensor_norm=np.reshape(y_tensor_norm, [d1,1,d2,d3]) 
    y_tensor_norm=torch.from_numpy(y_tensor_norm)

    TEST_SIZE = batch_size*100 # 256 * 100 frames
    TRAIN_SIZE = d1 - TEST_SIZE

    tr_x, te_x =  x_tensor_norm[:-TEST_SIZE], x_tensor_norm[-TEST_SIZE:]
    tr_y, te_y =  y_tensor_norm[:-TEST_SIZE], y_tensor_norm[-TEST_SIZE:]

    return tr_x, te_x, tr_y, te_y, mean_list_x, std_list_x, mean_list_y, std_list_y, TRAIN_SIZE, TEST_SIZE

# %%
