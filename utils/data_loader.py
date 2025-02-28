import os
import argparse
import numpy as np

import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

from .functions import *
from ase import units



def argon_dataset(args):
    root = args.root
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    temp_ = args.temperature

    x_tensor=np.load(os.path.join(root, dataset_path, f"input_long_{temp_}K.npy"))[-1]
    if len(x_tensor.shape) == 3:
        x_tensor = np.expand_dims(x_tensor, axis=0)
    s = np.shape(x_tensor)
    print(x_tensor.shape)
    x_tensor = np.reshape(x_tensor, (s[0]*s[1],s[2], s[3])) # (s[0], s[1],s[2])) #

    d1,d2,d3 = np.shape(x_tensor)
    x_tensor_norm, mean_list_x, std_list_x = normalize_md(x_tensor)
    x_tensor_norm=np.reshape(x_tensor_norm, [d1,1,d2,d3])
    x_tensor_norm=torch.from_numpy(x_tensor_norm)

    y_tensor=np.load(os.path.join(root, dataset_path, f"output_long_{temp_}K.npy"))[-1]
    if len(y_tensor.shape) == 3:
        y_tensor = np.expand_dims(y_tensor, axis=0)
    s = np.shape(y_tensor)
    print(y_tensor.shape)
    y_tensor = np.reshape(y_tensor, (s[0]*s[1],s[2], s[3])) # (s[0], s[1],s[2]))# 
    d1,d2,d3 = np.shape(y_tensor)
    y_tensor_norm, mean_list_y, std_list_y = normalize_md(y_tensor)
    y_tensor_norm=np.reshape(y_tensor_norm, [d1,1,d2,d3]) 
    y_tensor_norm=torch.from_numpy(y_tensor_norm)

    TEST_SIZE = batch_size*2 # 256 * 100 frames
    TRAIN_SIZE = d1 - TEST_SIZE

    tr_x, te_x =  x_tensor_norm[:-TEST_SIZE], x_tensor_norm[-TEST_SIZE:]
    tr_y, te_y =  y_tensor_norm[:-TEST_SIZE], y_tensor_norm[-TEST_SIZE:]

    return tr_x, te_x, tr_y, te_y, mean_list_x, std_list_x, mean_list_y, std_list_y, TRAIN_SIZE, TEST_SIZE, None, None

# %%

def argon_dataset_mixed_temp(args):
    root = args.root
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    temp_list = args.t_selection

    xs = []  
    ys = []
    ts = []
    
    t_standard = 300

    assert len(temp_list) >= 1, "Temperature list should exist"

    total_samples = 0
    for temp_ in temp_list:
        x_tensor = np.load(os.path.join(root, dataset_path, f"input_long_{temp_}K.npy"))[-1]
        y_tensor = np.load(os.path.join(root, dataset_path, f"output_long_{temp_}K.npy"))[-1]
        
        if len(x_tensor.shape) == 3:
            x_tensor = np.expand_dims(x_tensor, axis=0)
            y_tensor = np.expand_dims(y_tensor, axis=0)

        s = np.shape(x_tensor)
        print(f"x tensor shape for {temp_}K: ", x_tensor.shape)
        
        x_tensor = np.reshape(x_tensor, (s[0]*s[1], s[2], s[3]))
        y_tensor = np.reshape(y_tensor, (s[0]*s[1], s[2], s[3]))
        
        xs.append(x_tensor)
        ys.append(y_tensor)
        ts.append(np.ones(x_tensor.shape[0]) * temp_ / t_standard)
        total_samples += x_tensor.shape[0]

    # concatenate all the data
    x_tensor = np.concatenate(xs, axis=0)
    y_tensor = np.concatenate(ys, axis=0)
    ts = np.concatenate(ts, axis=0)

    assert x_tensor.shape == y_tensor.shape, "Input and output tensor shape should be the same"
    d1, d2, d3 = x_tensor.shape

    # normalization
    x_tensor_norm, mean_list_x, std_list_x = normalize_md(x_tensor)
    x_tensor_norm = np.reshape(x_tensor_norm, [d1, 1, d2, d3])
    x_tensor_norm = torch.from_numpy(x_tensor_norm)

    y_tensor_norm, mean_list_y, std_list_y = normalize_md(y_tensor)
    y_tensor_norm = np.reshape(y_tensor_norm, [d1, 1, d2, d3])
    y_tensor_norm = torch.from_numpy(y_tensor_norm)

    ts = torch.from_numpy(ts)

    # shuffle datas
    indices = np.arange(d1)
    np.random.shuffle(indices)
    x_tensor_norm = x_tensor_norm[indices]
    y_tensor_norm = y_tensor_norm[indices]
    ts = ts[indices]

    print("Final shapes:", x_tensor_norm.shape, y_tensor_norm.shape, ts.shape)

    TEST_SIZE = batch_size * 2 * len(temp_list)
    TRAIN_SIZE = d1 - TEST_SIZE

    tr_x, te_x = x_tensor_norm[:-TEST_SIZE], x_tensor_norm[-TEST_SIZE:]
    tr_y, te_y = y_tensor_norm[:-TEST_SIZE], y_tensor_norm[-TEST_SIZE:]
    tr_ts, te_ts = ts[:-TEST_SIZE], ts[-TEST_SIZE:]

    return tr_x, te_x, tr_y, te_y, mean_list_x, std_list_x, mean_list_y, std_list_y, TRAIN_SIZE, TEST_SIZE, tr_ts, te_ts

#%%
def argon_dataset_rev(args):
    '''
    input : args
    output : x and ys with shape of [total_len//n_bunch, n_bunch, 64, 9]
    xs : 
    '''
    root = args.root
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    temp_ = args.temperature
    dt_unit = args.dt_unit
    n_bunch = args.n_bunch # how much previous frames to use
    n_offset = int(args.n_offset - 1) # from how much distant frames to predict

    # gather dset per temperature
    assert isinstance(temp_, list), "Temperature should be a list"
    print("dt = ", dt_unit)
    print("offset = ", n_offset)

    tr_x, te_x, tr_y, te_y = [], [], [], []
    mean_list_x, std_list_x, mean_list_y, std_list_y = [], [], [], []
    tr_ts, te_ts = [], []

    for T in temp_:
        # discard the first tens of K frames
        x_tensor=np.load(os.path.join(root, dataset_path, f"input_64_{T}.npy"))[-4:]
        y_tensor=np.load(os.path.join(root, dataset_path, f"output_64_{T}.npy"))[-4:]

        s = np.shape(x_tensor)
        x_tensor = np.reshape(x_tensor, (s[0]*s[1],s[2], s[3])) # (s[0], s[1],s[2])) #
        y_tensor = np.reshape(y_tensor, (s[0]*s[1],s[2], s[3])) # (s[0], s[1],s[2]))# 
        assert np.shape(x_tensor) == np.shape(y_tensor), "x and y should have the same shape"

        # # stride by timestep
        # try : 
        #     x_tensor = x_tensor[::dt_unit][:40000]
        #     y_tensor = y_tensor[::dt_unit][:40000]
        # except:
        #     x_tensor = x_tensor[::dt_unit]
        #     y_tensor = y_tensor[::dt_unit]

        # chunck params
        # offset : baicsally y is 1 frame ahead of x : 
        y_tensor = y_tensor[n_offset:]
        # bunch : basically 1
        d1,d2,d3 = np.shape(y_tensor)
        quotient = d1 // n_bunch
        x_tensor = x_tensor[:quotient*n_bunch]
        y_tensor = y_tensor[:quotient*n_bunch]
        # reshape
        d1,d2,d3 = np.shape(y_tensor)
        
        if args.do_norm : 
            x_tensor_norm, mean_x, std_x = normalize_md(x_tensor)
            y_tensor_norm, mean_y, std_y = normalize_md(y_tensor)
        else:
            x_tensor_norm, mean_x, std_x = normalize_md_rev(x_tensor)
            y_tensor_norm, mean_y, std_y = normalize_md_rev(y_tensor)
        x_tensor_norm=np.reshape(x_tensor_norm, [quotient,n_bunch,d2,d3])
        x_tensor_norm=torch.from_numpy(x_tensor_norm)

        y_tensor_norm=np.reshape(y_tensor_norm, [quotient,n_bunch,d2,d3]) 
        y_tensor_norm=torch.from_numpy(y_tensor_norm)

        TEST_SIZE = int(quotient * 0.2)
        TRAIN_SIZE = quotient - TEST_SIZE

        tr_x.append(x_tensor_norm[:-TEST_SIZE])
        te_x.append(x_tensor_norm[-TEST_SIZE:])
        tr_y.append(y_tensor_norm[:-TEST_SIZE])
        te_y.append(y_tensor_norm[-TEST_SIZE:])
        mean_list_x.append(mean_x)
        std_list_x.append(std_x)
        mean_list_y.append(mean_y)
        std_list_y.append(std_y)
        tr_ts.append(torch.ones(TRAIN_SIZE) * T / 1000.)
        te_ts.append(torch.ones(TEST_SIZE) * T / 1000.)

    tr_x = torch.cat(tr_x, dim=0)
    te_x = torch.cat(te_x, dim=0)
    tr_y = torch.cat(tr_y, dim=0)
    te_y = torch.cat(te_y, dim=0)
    tr_ts = torch.cat(tr_ts, dim=0)
    te_ts = torch.cat(te_ts, dim=0)

    train_size = tr_x.shape[0]
    test_size = te_x.shape[0]

    return tr_x, te_x, tr_y, te_y, \
            np.mean(mean_list_x, axis=0).tolist(), np.mean(std_list_x, axis=0).tolist(), \
            np.mean(mean_list_y, axis=0).tolist(), np.mean(std_list_y, axis=0).tolist(), \
            train_size, test_size, tr_ts, te_ts