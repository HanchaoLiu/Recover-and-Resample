import os,sys
import numpy as np 
import torch 
from IPython import embed

import torch.nn.functional as F


# np.random.seed(1)
# torch.manual_seed(0)


##########################################################################
def f_croppad_torch_batch(data_batch, temperal_padding_ratio=6):
    '''
    data_batch: (n,c,t,v,m), torch.FloatTensor
    '''
    
    N, C, T, V, M = data_batch.shape
    padding_len = T // temperal_padding_ratio

    M_all=[]
    for i in range(N):
        frame_start = np.random.randint(0, padding_len * 2 + 1)
        # data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
        #                              data_numpy,
        #                              data_numpy[:, -padding_len:][:, ::-1]),
        #                             axis=1)
        # data_numpy = data_numpy[:, frame_start:frame_start + T]

        idx1 = np.arange(padding_len-1,-1,-1)
        idx2 = np.arange(0,T,1)
        idx3 = np.arange(T-1,T-padding_len-1,-1)
        idx_all = np.concatenate([idx1,idx2,idx3],0)

        idx_selected = idx_all[frame_start:frame_start+T]
        # print(idx_selected)

        M = np.zeros((T,T))
        order_idx_list = np.arange(T)
        M[order_idx_list, idx_selected] = 1
        # print(M[:10,:10])
        # print(M.shape, x.shape)
        M_all.append( M )
    M_all = np.stack(M_all, 0)
    M_all = torch.FloatTensor(M_all).to(data_batch.device)

    res = torch.einsum('bit,bctvm->bcivm', M_all,data_batch)
    assert res.shape==data_batch.shape
    return res 
    

def f_cropresize_torch_batch(data_batch, p_interval):
    '''
    data_batch: (n,c,t,v,m)
    p_interval=[st_ratio, ed_ratio]
    '''
    N, C, T, V, M = data_batch.shape
    
    M_all=[]
    for i in range(N):
        valid_size = T
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = int(np.floor(valid_size*p))
        bias = np.random.randint(0,valid_size-cropped_length+1)

        M = np.zeros((T,T))
        order_idx_list = np.arange(T)
        sample_idx = np.linspace(bias, bias+cropped_length-1, num=64, endpoint=True)
        
        # batch form 
        M[order_idx_list, sample_idx.astype(int)] = 1 - (sample_idx - sample_idx.astype(int))

        sample_idx2 = sample_idx.astype(int)+1
        sample_idx2_selected = np.where(sample_idx2 < T)[0]
        M[ order_idx_list[sample_idx2_selected], (sample_idx+1).astype(int)[sample_idx2_selected]] = \
            (sample_idx - sample_idx.astype(int))[sample_idx2_selected]

        M_all.append(M)
    M_all = np.stack(M_all, 0)
    M_all = torch.FloatTensor(M_all).to(data_batch.device)
    res = torch.einsum('bit,bctvm->bcivm', M_all,data_batch)
    assert res.shape==data_batch.shape
    return res 



def f2_linear(data_batch, W, W_weight):
    '''
    data_batch: (n,c,t,v,m)
    p_interval=[st_ratio, ed_ratio]
    '''
    N, C, T, V, M = data_batch.shape
    
    M_all=[]
    for i in range(N):
        
        M = get_clusterW_sample(W, W_weight)
        assert M.shape==(64,64)

        M_all.append(M)
    M_all = np.stack(M_all, 0)
    M_all = torch.FloatTensor(M_all).to(data_batch.device)
    res = torch.einsum('bit,bctvm->bcivm', M_all,data_batch)
    assert res.shape==data_batch.shape
    return res 



def get_clusterW_sample(W, W_weight):
    n=W.shape[0]
    if W_weight is None:
        idx = np.random.choice(n)
        return W[idx]

    assert W.shape[0]==W_weight.shape[0]
    weight_list=W_weight/W_weight.sum()
    idx = np.random.choice(n, 1, replace=False, p=weight_list)
    return W[idx]


def main():
    pass






if __name__ == "__main__":
    main()
