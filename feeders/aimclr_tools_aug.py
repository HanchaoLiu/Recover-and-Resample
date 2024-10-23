import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos


transform_order = {
    'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
}


def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    # return data_numpy
    return data_numpy.astype(np.float32)


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


def random_spatial_flip(seq):
    index = transform_order['ntu']
    trans_seq = seq[:, :, index, :]
    return trans_seq


def random_time_flip(seq):
    T = seq.shape[1]
    
    time_range_order = [i for i in range(T)]
    time_range_reverse = list(reversed(time_range_order))
    return seq[:, time_range_reverse, :, :]
    


def random_rotate(seq, aug_angle=30):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        temp = np.matmul(seq, R)
        return temp

    new_seq = seq.copy()
    # C, T, V, M -> T, V, M, C
    new_seq = np.transpose(new_seq, (1, 2, 3, 0))
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            # rotate_angle = random.uniform(0, 30)
            rotate_angle = random.uniform(0, aug_angle)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    new_seq = np.transpose(new_seq, (3, 0, 1, 2))

    return new_seq


def gaus_noise(data_numpy, std=0.01):

    mean= 0
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    noise = np.random.normal(mean, std, size=(C, T, V, M))
    return temp + noise
    

def gaus_filter(data_numpy, sigma_value):
    g = GaussianBlurConv(3, sigma=[0.1, sigma_value])
    return g(data_numpy)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel = 15, sigma = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        
        # normalize all filter elements to sum 1, so acts like weights.
        blur_flter = blur_flter / blur_flter.sum()
        
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)

        # np.set_printoptions(3, suppress=True)
        # print(blur_flter)

        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1) # (3,1,1,5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        # if prob < 0.5:
        x = x.permute(3,0,2,1) # M,C,V,T
        x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
        x = x.permute(1,-1,-2, 0) #C,T,V,M

        return x.numpy()

class Zero_out_axis(object):
    def __init__(self, axis = None):
        self.first_axis = axis


    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)

        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp


def axis_mask(data_numpy, axis):
    assert axis in [0,1,2]
    am = Zero_out_axis(axis)
   
    return am(data_numpy)


def temporal_random(data_numpy):
    c,t,v,m=data_numpy.shape
    idx_list = np.sort(np.random.randint(0,t,size=t))
    data_numpy = data_numpy[:,idx_list,:,:]
    return data_numpy


def temporal_kernel_fileter(data_numpy):
    c,t,v,m=data_numpy.shape
    idx_list = np.sort(np.random.uniform(0.1,0.9,size=t))*(t-1)
    idx_list = np.round(idx_list).astype(int)
    idx_list[:2]  = np.array([0,2])
    idx_list[-2:] = np.array([t-3,t-1])
    data_numpy = data_numpy[:,idx_list,:,:]
    return data_numpy
    

def temporal_kernel_filter_weight(data_numpy):
    c,t,v,m=data_numpy.shape
    
    # weight_list[[0,1,2,3,4,t-5,t-4,t-3,t-2,t-1]]=0.0
    # weight_list[27:37]=0.0
    weight_list=np.ones(t)
    weight_list[10:20]=0.0

    # weight_list=np.zeros(t)
    # weight_list[20:30]=1.0



    # weight_list=weight_list*0.0
    # weight_list[0]=1.0

    weight_list=weight_list/weight_list.sum()
    idx_list = np.sort(np.random.choice(t, size=t, replace=True, p=weight_list))
    idx_list = np.round(idx_list).astype(int)
    data_numpy = data_numpy[:,idx_list,:,:]

    print(idx_list)
    return data_numpy



def temporal_kernel_filter_weight_random(data_numpy):
    c,t,v,m=data_numpy.shape
    
    # weight_list[[0,1,2,3,4,t-5,t-4,t-3,t-2,t-1]]=0.0
    # weight_list[27:37]=0.0
    weight_list=np.ones(t)

    a1 = np.random.randint(0,10)
    a2 = np.random.randint(5,10)
    weight_list[a1:a1+a2]=0.0
    # print(a1,a2)

    # weight_list=np.zeros(t)
    # weight_list[20:30]=1.0



    # weight_list=weight_list*0.0
    # weight_list[0]=1.0

    weight_list=weight_list/weight_list.sum()
    idx_list = np.sort(np.random.choice(t, size=t, replace=True, p=weight_list))
    idx_list = np.round(idx_list).astype(int)
    data_numpy = data_numpy[:,idx_list,:,:]
    return data_numpy




def test_gau_blur():

    g = GaussianBlurConv(3)
    data_seq = np.ones((3, 20, 25, 1))
    g(data_seq)






if __name__ == '__main__':
    # 
    # data_seq = axis_mask(data_seq)
    # print(data_seq.shape)

    test_gau_blur()