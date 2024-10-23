import os,sys
import numpy as np 
from IPython import embed

# import matplotlib.pyplot as plt 
# from vis_helpers import * 
# np.random.seed(1)




# inward to outward, from root(idx=1) to end effector.
# bone_idx: [inward_joint_idx, outward_joint_idx]
bones_dt={
    0:[1,0],
    1:[20,2],
    2:[2,3],
    3:[20,4],
    # 3:[2,4],
    4:[4,5],
    5:[5,6],
    6:[6,7],
    # 7:[2,8],
    7:[20,8],
    8:[8,9],
    9:[9,10],
    10:[10,11],
    11:[0,12],
    12:[12,13],
    13:[13,14],
    14:[14,15],
    15:[0,16],
    16:[16,17],
    17:[17,18],
    18:[18,19],
    19:[1,20],
    20:[22,21],
    21:[7,22],
    22:[24,23],
    23:[11,24]
}

bones_dt_mat = np.array([bones_dt[i] for i in range(24)])

joint_idx_dependents={
    0:[0],
    1:[],
    2:[19,1],
    3:[19,1,2],
    # 4:[19,1,3],
    # 5:[19,1,3,4],
    # 6:[19,1,3,4,5],
    # 7:[19,1,3,4,5,6],
    # 22:[19,1,3,4,5,6,21],
    # 21:[19,1,3,4,5,6,21,20],
    # 8:[19,1,7],
    # 9:[19,1,7,8],
    # 10:[19,1,7,8,9],
    # 11:[19,1,7,8,9,10],
    # 24:[19,1,7,8,9,10,23],
    # 23:[19,1,7,8,9,10,23,22],

    4:[19,3],
    5:[19,3,4],
    6:[19,3,4,5],
    7:[19,3,4,5,6],
    22:[19,3,4,5,6,21],
    21:[19,3,4,5,6,21,20],
    8:[19,7],
    9:[19,7,8],
    10:[19,7,8,9],
    11:[19,7,8,9,10],
    24:[19,7,8,9,10,23],
    23:[19,7,8,9,10,23,22],
    
    12:[0,11],
    13:[0,11,12],
    14:[0,11,12,13],
    15:[0,11,12,13,14],
    16:[0,15],
    17:[0,15,16],
    18:[0,15,16,17],
    19:[0,15,16,17,18],
    20:[19],
}


def get_bones(data_numpy):
    '''
    shape: (c,t,v,m)
    return (c,t,n_bones,m)
    '''
    bones = data_numpy[:,:,bones_dt_mat[:,1],:] - data_numpy[:,:,bones_dt_mat[:,0],:]
    return bones

def get_mask():
    '''
    return (n_joints, n_bones)
    '''
    n_joints=25
    n_bones=24
    m = np.zeros((n_joints,n_bones),float)
    for i in range(n_joints):
        m[i, joint_idx_dependents[i]]=1.0

    return m 

def apply_mask(M, bones):
    '''
    M: (n_joints, n_bones)
    bones: (c,t,n_bones,m)
    '''
    M = M[None,None,None]
    bones = np.transpose(bones,[0,1,3,2])[...,None]

    # (c,t,m,n_joints,1)
    res=np.matmul(M,bones)[...,0]
    res = np.transpose(res,[0,1,3,2])
    return res 

def exchange_limb_list_mat(data_src,data_dst, bone_list):
    n_joints=25
    n_bones=24

    mask_valid = get_mask()

    mask_ex = get_mask()
    if len(bone_list)!=0:
        mask_ex[:,bone_list]=0.0

    mask_ex_inv = 1 - mask_ex 
    mask_ex_inv = mask_valid * mask_ex_inv

    # here retrieve get_bones(), and modify each bones.
    res = apply_mask(mask_ex, get_bones(data_src)) + apply_mask(mask_ex_inv, get_bones(data_dst))

    return res 





def main():

    
    pass





if __name__ == "__main__":
    main()