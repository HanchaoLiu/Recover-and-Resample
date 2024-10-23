import os,sys
import numpy as np 
import torch 
from IPython import embed

import argparse

import torch.nn.functional as F
import torchvision

np.random.seed(1)
torch.manual_seed(0)

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--save_dir', default='/home/cscg/liuhc/res', help='the work folder for storing results')


    parser.add_argument('--raw_data_st',type=float,default=0.0,help='')
    parser.add_argument('--raw_data_ed',type=float,default=1.0,help='')

    parser.add_argument('--cut_data_st',type=float,default=0.0,help='')
    parser.add_argument('--cut_data_ed',type=float,default=1.0,help='')

    return parser




def get_first_frame_condition(data_input, bkg_pose_list):
    N,C,T,V,M = data_input.shape
    st_int=0
    ed_int=T
    frame0 = data_input[:,:,st_int:st_int+1,:,:]
    frame_bkg_estimate = get_frame0_wrapper_batch(frame0, bkg_pose_list)
    # print("bkg=",frame_bkg_estimate.shape, data_input[:,:,0:1,:,:].shape)

    return frame_bkg_estimate


def get_frame0_wrapper_batch(data_batch, pose_clusters_torch):
    if not isinstance(data_batch, torch.Tensor):
        data_batch = torch.FloatTensor(data_batch)

    res = select_frame0_from_pool_batch(data_batch, pose_clusters_torch)
    
    return res 




def get_frame0_wrapper(data_batch, pose_clusters_torch):
    '''
    data_batch: (n,c,t,v,m), n=1
    '''
    if not isinstance(data_batch, torch.Tensor):
        data_batch = torch.FloatTensor(data_batch)
    res = select_frame0_from_pool_single(data_batch, pose_clusters_torch)
    return res


def select_frame0_from_pool_batch(data_batch, frame0_pool, rotate=True):
    '''
    args:
        data_batch: (n,c,t,v,m),
        frame0_pool: (n_pool,c,t,v,m), t=1
    return:
        frame0: (n,c,t,v,m), t=1
    '''
    bs = data_batch.shape[0]
    res = []
    for i in range(bs):
        res.append( select_frame0_from_pool_single(data_batch[i:i+1],frame0_pool,rotate) )
    res = torch.cat(res,0)
    return res 


def select_frame0_from_pool_single(data_batch, frame0_pool, rotate=True):
    '''
    args:
        data_batch: (n,c,t,v,m), n=1
        frame0_pool: (n_pool,c,t,v,m), t=1
        rotate: whether to solve for best rotate
    return:
        frame0: (n,c,t,v,m), n=1, t=1

    all done in torch.tensor
    '''
    if data_batch[:,:,:,:,1].std().item() > 1e-3:
        num_person=2 
    else:
        num_person=1

    frame0_pool = frame0_pool[:,:,0:1,:,0:1]


    if num_person==1:
        # (1,c,1,v,1)
        frame0_selected = select_person_from_pool( data_batch[:,:,0:1,:,0:1], frame0_pool)
        zero_pad = torch.zeros_like(frame0_selected)
        frame0 = torch.cat([frame0_selected, zero_pad], 4)

    elif num_person==2:

        # select first person 
        frame0_selected_p1 = select_person_from_pool( data_batch[:,:,0:1,:,0:1], frame0_pool)

        # select second person (no) -> copy 1st person
        # check whether to apply rot on second person.
        ref_idx=1
        traj_p2 = data_batch[:,:,0:1,ref_idx:ref_idx+1,1:2]
        frame0_selected_p2 = frame0_selected_p1 + traj_p2
        if rotate:
            p2 = data_batch[:,:,0:1,:,1:2]
            frame0_selected_p2_raw = frame0_selected_p2
            frame0_selected_p2_rot = torch.cat(
                [-frame0_selected_p1[:,0:1],-frame0_selected_p1[:,1:2],frame0_selected_p1[:,2:3]],1
            )+traj_p2
            dist_raw = torch.norm(p2 - frame0_selected_p2_raw).item()
            dist_rot = torch.norm(p2 - frame0_selected_p2_rot).item()
            # print(f"dist_raw={dist_raw},dist_rot={dist_rot}")
            if dist_raw < dist_rot:
                frame0_selected_p2 = frame0_selected_p2_raw
            else:
                frame0_selected_p2 = frame0_selected_p2_rot 

        
        frame0 = torch.cat([frame0_selected_p1,frame0_selected_p2],4)

    else:
        raise ValueError()

    return frame0


def select_person_from_pool(data_batch, frame0_pool):
    '''
    (n,c,t,v,m)
    data_batch.shape = [1,c,1,v,1]
    frame0_pool      = [np,c,1,v,1]
    '''
    assert data_batch.shape[0]==1
    assert data_batch.shape[1:] == frame0_pool.shape[1:]



    data_batch_dim1 = data_batch.reshape(data_batch.shape[0], -1)
    frame0_pool_dim1= frame0_pool.reshape(frame0_pool.shape[0],-1)

    dist = torch.norm(data_batch_dim1 - frame0_pool_dim1, dim=1)
    min_idx = dist.min(dim=0)[1]

    ret = frame0_pool[min_idx:min_idx+1]
    return ret 


def main():
    pass 




if __name__ == "__main__":
    main()
    