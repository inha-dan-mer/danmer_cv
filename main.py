# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:04:48 2022


"""
import os
cur_dir = os.getcwd()
import numpy as np
import torch
from soft_dtw_cuda import SoftDTW


tutor = np.load('danmer_cv-main/save_tutor_coordin.npy')
tutee = np.load('danmer_cv-main/save_tutee_coordin.npy')[:627]
dun = np.load('danmer_cv-main/save_dundun_coordin.npy')[:627]

tutor_th = torch.from_numpy(tutor).cuda()
tutee_th = torch.from_numpy(tutee).cuda()
dun_th = torch.from_numpy(dun).cuda()

# assert tutor_th.size(0) == tutee_th.size(0), "Match the length!"


sdtw = SoftDTW(use_cuda=True, gamma=0.1)
dance_distance = sdtw(tutor_th,tutee_th) / 627



# %%

### Examples

# Create the sequences
batch_size, len_x, len_y, dims = 8, 15, 12, 5
x = torch.rand((batch_size, len_x, dims), requires_grad=True)
y = torch.rand((batch_size, len_y, dims))
# Transfer tensors to the GPU
x = x.cuda()
y = y.cuda()


prediction = model(x)

# Create the "criterion" object
sdtw = SoftDTW(use_cuda=True, gamma=0.1)

# Compute the loss value
loss = sdtw(prediction, y)  # Just like any torch.nn.xyzLoss()

# Aggregate and call backward()
loss.mean().backward()




tensor([[0.5062, 0.1479, 0.1110, 0.1015, 0.2969],
        [0.8426, 0.2881, 0.3250, 0.2369, 0.8188],
        [0.1764, 0.3169, 0.3711, 0.9908, 0.3336],
        [0.2953, 0.6420, 0.8349, 0.7865, 0.3519],
        [0.9065, 0.0614, 0.8787, 0.8353, 0.9782],
        [0.1755, 0.8782, 0.1793, 0.2818, 0.6281],
        [0.8685, 0.6669, 0.2344, 0.8569, 0.5213],
        [0.5613, 0.6196, 0.9763, 0.4167, 0.2171],
        [0.9650, 0.3056, 0.2146, 0.3144, 0.7225],
        [0.2390, 0.8673, 0.9793, 0.7172, 0.5660],
        [0.4718, 0.9432, 0.5929, 0.6324, 0.5134],
        [0.2749, 0.0982, 0.5816, 0.9434, 0.8234],
        [0.6299, 0.5631, 0.6806, 0.4152, 0.4458],
        [0.6257, 0.8616, 0.7436, 0.4587, 0.5202],
        [0.0467, 0.3515, 0.9525, 0.3740, 0.1890]], device='cuda:0',
       grad_fn=<SelectBackward>)
