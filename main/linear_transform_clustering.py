import os,sys
import numpy as np 
import torch 
from IPython import embed

import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict
from tqdm import tqdm


np.random.seed(1)
torch.manual_seed(0)



def set_mask_topk(coef_mat, topk):
    coef_mat = torch.FloatTensor(coef_mat)
    T0,_ = coef_mat.shape 
    _,topk_idx = coef_mat.topk(k=topk, dim=1)

    res = torch.zeros_like(coef_mat)
    for row in range(T0):
        res[row, topk_idx[row]] = coef_mat[row, topk_idx[row]]
    return res


def context_aware_aggregation(conf_mat):
    '''
    conf_mat.shape
    '''
    rows, cols = conf_mat.shape
    assert len(conf_mat.shape)==2
    if False:
        conf_mat = set_mask_topk(conf_mat, topk=32)
        conf_mat = conf_mat / (conf_mat.sum(dim=1, keepdim=True) + 1e-5)

    coord_mat = torch.arange(cols).reshape(1,-1).expand(rows, cols)
    mean_coord_mat = (conf_mat * coord_mat).sum(dim=1).long()
    traj_mat = torch.zeros_like(conf_mat)
    traj_mat[torch.arange(rows), mean_coord_mat] = 1.0
    return traj_mat



def resize_torch_interp_batch(data, target_frame):

    window = target_frame

    n,c,t,v,m = data.shape 
    data = data.permute(0,1,3,4,2).contiguous().view(n,c*v*m,t)
    data = data[:,:,:,None]
    data = F.interpolate(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3)
    data = data.reshape(n,c,v,m,window)
    data = data.permute(0,1,4,2,3).contiguous()
    return data 


def get_dist_matrix(x, y):
    '''
    x,y: (n,c,t,v,m), torch.tensor
    '''
    # (n,t,c,v,m)
    assert x.shape==y.shape
    n,c,t,v,m = x.shape
    x = x.permute(0,2,1,3,4).contiguous()
    y = y.permute(0,2,1,3,4).contiguous()
    x = x.reshape(n,t,c*v*m)[0]
    y = y.reshape(n,t,c*v*m)[0]

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    dist = ((x-y)**2).sum(dim=2)
    # print(dist.shape)
    return dist


def get_gt_frame_mapping(data_input, st, ed, temp=0.1):
    '''
    map [st,ed] -> [0,1]
    data_input: (n,c,t,v,m), torch.tensor
    return data_output
    '''
    N,C,T,V,M = data_input.shape
    res = torch.zeros((T,T))
    st_int = int(T*st)
    ed_int = int(T*ed)

    # [0,1,...,T-1] -> [st_int,..., ed_int]
    # [best_fit_index] -> [0, st_int-1], [ed_int+1,T]
    data_output = resize_torch_interp_batch(data_input[:,:,st_int:ed_int,:,:], T)

    
    if False:
        order_idx = np.arange(st_int,ed_int)
        sample_idx = np.round(np.linspace(0,T-1,num = ed_int-st_int))
        sample_idx = torch.LongTensor(sample_idx)
        res[order_idx, sample_idx] = 1.0

    # distance matrix, (n,n)
    distance_matrix = get_dist_matrix(data_input, data_output)
    _, min_dist_idx_list = distance_matrix.min(dim=1)

    
    res = F.softmax(-distance_matrix/temp, dim=1)

    return res, data_output
    


def do_w_clustering(prior_data_npy_name, n_clusters, T1=0.1, T2=0.1, topk=32, use_bkg=False, save_name=None, 
    use_sample=False, segment_list=None, bkg_pose_list=None):

    if use_bkg==1:
        assert bkg_pose_list is not None

    import matplotlib 
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt 
    from tqdm import tqdm

    from sklearn.cluster import KMeans
    from collections import Counter

    data = np.load(prior_data_npy_name)
    print('training prior data.shape = ', data.shape)
    N0,C0,T0,V0,M0 = data.shape

    # use_sample=False 
    if use_sample:
        print("using selected samples...")
        sample_list = np.random.choice(data.shape[0], size=1000)
    else:
        sample_list = np.arange(data.shape[0])

    M_list = []
    tr_list = []
    if segment_list is None:
        segment_list = [
            (0.0, 1.0), 
            (0.0, 0.5), (0.5, 1.0), (0.375, 0.875),
            (0.25, 0.75), (0.125, 0.625), 
            (0.0, 0.75), (0.25, 1.0)
        ]
    
    for i in tqdm(sample_list):
        data_input = torch.FloatTensor(data[i:i+1])

        # sample here 
        N,C,T,V,M = data_input.shape
        raw_data_st=0
        raw_data_ed=1
        raw_data_st_int = int(raw_data_st*T)
        raw_data_ed_int = int(raw_data_ed*T)
        data_input = resize_torch_interp_batch(data_input[:,:,raw_data_st_int:raw_data_ed_int,:,:], 64)




        for (st,ed) in segment_list:

        
            if use_bkg:
                M, _ = get_gt_frame_mapping_addbkg(data_input, st, ed, bkg_pose_list)
            else:
                M, _ = get_gt_frame_mapping(data_input, st, ed, T1)
            M = M.numpy()

            tr_list.append( M.copy() )
        
    tr_list = np.stack(tr_list)
    print('tr_list.shape=', tr_list.shape)

    n = tr_list.shape[0]
    tr_list = tr_list.reshape((n,-1))
    print(tr_list.shape, tr_list.dtype)
    
    
    X = tr_list
    kmeans = KMeans(n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    c = Counter(cluster_labels)
    print(c)

    # get each cluster center
    tr_cluster_center_list = []
    tr_cluster_weight_list = []
    cluster_labels = np.array(cluster_labels)
    n_cluster_label = np.unique(cluster_labels)
    for cluster_label in n_cluster_label:
        selected_idx_list = np.where(cluster_labels==cluster_label)[0]
        print(f'cluster_label={cluster_label}, number={selected_idx_list.shape}')

        tr_cluster_weight_list.append(selected_idx_list.shape[0])
        tr_cluster_center_list.append( tr_list[selected_idx_list].mean(axis=0) )
    
    # append center for all.
    tr_cluster_center_list.append( tr_list.mean(axis=0) )

    tr_cluster_center_list = np.stack(tr_cluster_center_list,0)
    tr_cluster_center_list = tr_cluster_center_list.reshape((tr_cluster_center_list.shape[0],T0,T0))
    print("adding mean(all samples) as last cluster center.")
    print(tr_cluster_center_list.shape)

    
    tr_cluster_center_list = torch.FloatTensor(tr_cluster_center_list).unsqueeze(1)
    tr_cluster_center_list_npy = tr_cluster_center_list.clone().numpy()
    
    
    # plot conf mat 
    if save_name:
        draw_vis = True 
    else:
        draw_vis = False

    if draw_vis:
        fig = plt.figure(figsize=(20,4))
    else:
        fig = None

    tr_cluster_weight_list = np.array(tr_cluster_weight_list)
    tr_cluster_weight_list = tr_cluster_weight_list/tr_cluster_weight_list.sum()
    tr_cluster_weight_list = np.append(tr_cluster_weight_list, 1.0)

    n_clusters_append = tr_cluster_center_list_npy.shape[0]

    conf_mat_list = []
    traj_mat_list = []

    for bs in range(tr_cluster_center_list_npy.shape[0]):
        print(tr_cluster_center_list_npy[bs,0].shape)
        conf_mat = tr_cluster_center_list_npy[bs,0]

        

        conf_mat = conf_mat / conf_mat.sum(axis=1, keepdims=True)
        # print(conf_mat.sum(axis=1))

        # set some threshold
        if False:
            conf_mat[conf_mat<0.01]=0.0
            conf_mat = conf_mat / (conf_mat.sum(axis=1, keepdims=True) + 1e-5)
        
        if True:
            conf_mat = set_mask_topk(conf_mat, topk=topk).numpy()
            conf_mat = conf_mat / (conf_mat.sum(axis=1, keepdims=True) + 1e-5)

        coord_mat = np.repeat(np.arange(T0)[None], T0, axis=0)

        conf_mat_masked = conf_mat.copy()
        mean_coord_mat = (conf_mat_masked * coord_mat).sum(axis=1).astype(int)
        traj_mat = np.zeros_like(conf_mat)
        traj_mat[np.arange(T0), mean_coord_mat] = 1.0

        conf_mat_list.append(conf_mat)
        traj_mat_list.append(traj_mat)

        if draw_vis:
            plt.subplot(2,n_clusters_append,bs+1)
            plt.imshow(conf_mat, vmin=0, vmax=0.1)
            if bs==tr_cluster_center_list_npy.shape[0]-1:
                plt.title(f'all({tr_cluster_weight_list[bs]:.2f})')
            else:
                plt.title(f'{bs}({tr_cluster_weight_list[bs]:.2f})')
            plt.axis('off')
            plt.subplot(2,n_clusters_append,bs+1+n_clusters_append)
            plt.imshow(traj_mat, vmin=0, vmax=1.0)
            plt.axis('off')

    if draw_vis:
        # save_confmat_name = os.path.join(args.save_dir, save_name)
        save_confmat_name = save_name
        plt.tight_layout()
        plt.savefig(save_confmat_name, dpi=50)
        print("save to ", save_confmat_name)
        # plt.close()
        

    
    conf_mat_list = np.stack(conf_mat_list,0)
    traj_mat_list = np.stack(traj_mat_list,0)
    
    append_all=False 
    if append_all:
        return conf_mat_list, traj_mat_list, tr_cluster_weight_list, fig
    else:
        return conf_mat_list[:-1], traj_mat_list[:-1], tr_cluster_weight_list[:-1], fig






def main():
    pass






if __name__ == "__main__":
    main()

    