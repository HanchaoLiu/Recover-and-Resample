import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F


def resize_torch_interp_batch(data, target_frame):

    window = target_frame

    n,c,t,v,m = data.shape 
    data = data.permute(0,1,3,4,2).contiguous().view(n,c*v*m,t)
    data = data[:,:,:,None]
    data = F.interpolate(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3)
    data = data.reshape(n,c,v,m,window)
    data = data.permute(0,1,4,2,3).contiguous()
    return data 


def resize_torch_interp_batch_numpy(data_numpy, target_frame):
    data = torch.FloatTensor(data_numpy)
    return resize_torch_interp_batch(data, target_frame).numpy()