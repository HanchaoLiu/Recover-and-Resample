import os
import sys
import numpy as np
import math



def random_rotation_2d(data_numpy, theta):
    '''
    theta=np.array([th1,th2,th3]) in rad.
    data_numpy: (C,T,V,M)
    theta=np.random.uniform(-10,10,size=3)/180.0*2*np.pi
    '''
    C,T,V,M=data_numpy.shape
    data_numpy= np.transpose(data_numpy, [3,1,2,0]).reshape(-1,C)

    
    # rot_mat=get_rot_mat_random(theta)
    cos_val = np.cos(theta)
    sin_val = np.sin(theta)
    rot_mat = np.array([[cos_val, -sin_val],
                        [sin_val,  cos_val]])


    data_numpy= np.dot(data_numpy, rot_mat.T)
    data_numpy= data_numpy.reshape([M,T,V,C])
    data_numpy= np.transpose(data_numpy, [3,1,2,0])

    return data_numpy


def my_random_rot_2d(data_numpy, theta_val):
    theta=np.random.uniform(-theta_val,theta_val)/180.0*np.pi
    data_numpy=random_rotation_2d(data_numpy, theta)
    return data_numpy






