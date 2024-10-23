'''
copy from 
    action_base/skeleton/feeders/rotation_aug.py
'''

import os
import sys
import numpy as np
import math
import random
import torch 


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)


def get_rot_mat_random(theta):
    # angles = theta.reshape(-1)
    angles=theta
    basis = rotate_basis(np.eye(3), angles)
    return basis


def random_rotation(data_numpy, theta):
    '''
    theta=np.array([th1,th2,th3]) in rad.
    data_numpy: (C,T,V,M)
    theta=np.random.uniform(-10,10,size=3)/180.0*2*np.pi
    '''
    C,T,V,M=data_numpy.shape
    data_numpy= np.transpose(data_numpy, [3,1,2,0]).reshape(-1,C)

   
    rot_mat=get_rot_mat_random(theta)


    data_numpy= np.dot(data_numpy, rot_mat.T)
    data_numpy= data_numpy.reshape([M,T,V,C])
    data_numpy= np.transpose(data_numpy, [3,1,2,0])

    return data_numpy

def rotate_basis(local3d, angles):
    """
    Rotate local rectangular coordinates from given view_angles.

    :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return:
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    x = local3d[0]
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x.T @ mat33_z
    return local3d


def my_random_rot(data_numpy, theta_val):
    theta=np.random.uniform(-theta_val,theta_val,size=3)/180.0*2*np.pi
    data_numpy=random_rotation(data_numpy, theta)
    return data_numpy


def my_random_rot_batch(data_batch, random_theta_batch):
    '''
    data_batch: (n,c,t,v,m)
    random_theta_batch: (n,3)
    '''
    bs = data_batch.shape[0]
    res_list = []
    for i in range(bs):
        data_numpy=random_rotation(data_batch[i], random_theta_batch[i])
        res_list.append(data_numpy)
    res_list = np.stack(res_list,0)
    return res_list


def apply_my_rotation_batch_torch_fromThetaBatch(data_batch, random_theta_batch):
    '''
    data_batch: (n,c,t,v,m)
    '''
    N,C,T,V,M = data_batch.shape 

    rot_mat_list = []
    for i in range(N):
        # theta=np.random.uniform(-theta_val,theta_val,size=3)/180.0*2*np.pi
        theta = random_theta_batch[i]
        rot_mat=get_rot_mat_random(theta)
        rot_mat_list.append(rot_mat)
    rot_mat_list = np.stack(rot_mat_list,0)
    rot_mat_list = torch.FloatTensor(rot_mat_list).to(data_batch.device)
    # (n,3,3)
    rot_mat_list = rot_mat_list.permute(0,2,1).contiguous()
    # (n,1,1,3,3)
    rot_mat_list = rot_mat_list.unsqueeze(1).unsqueeze(2)

    # (n,t,v,m,3)
    data_batch = data_batch.permute(0,2,3,4,1).contiguous()
    # (n,t,v,m,3)
    data_batch = torch.matmul(data_batch, rot_mat_list)

    data_batch = data_batch.permute(0,4,1,2,3).contiguous()

    return data_batch


def apply_my_rotation_batch_torch(data_batch, theta_val):
    n = data_batch.shape[0]
    
    random_theta_batch = np.random.uniform(-theta_val,theta_val,size=(n,3))/180.0*2*np.pi
    res = apply_my_rotation_batch_torch_fromThetaBatch(data_batch, random_theta_batch)
    return res 


def check_npy_torch_same():
    n,c,t,v,m = 2,3,64,25,3
    data_batch = torch.rand((n,c,t,v,m))

    theta_val = 30
    random_theta_batch = np.random.uniform(-theta_val,theta_val,size=(n,3))/180.0*2*np.pi

    x1=my_random_rot_batch(data_batch, random_theta_batch)
    x2=apply_my_rotation_batch_torch(data_batch, random_theta_batch)
    x2=x2.numpy()
    print(x1.shape, x2.shape)

    assert np.allclose(x1,x2,atol=1e-4)


if __name__ == "__main__":
    check_npy_torch_same()