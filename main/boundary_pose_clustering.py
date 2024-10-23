import os,sys
# sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from IPython import embed 

import glob 

import torch 
import torch.nn as nn 
import torch.nn.functional as F


def set_ax_view(ax, elev, azim):
    xlim = 1
    ax.set_xlim([-xlim,xlim])
    ax.set_ylim([-xlim,xlim])
    ax.set_zlim([-xlim,xlim])
    # elev = 30
    # azim = -60
    ax.view_init(elev=elev,azim=azim)
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def draw_skeleton_on_ax(ax, data):
    data_x = data[:,0:1,:,:,:]
    data_y = data[:,1:2,:,:,:]
    data_z = data[:,2:3,:,:,:]
    data = np.concatenate([data_x,-data_y,data_z],1)
    # to (t,c,v), t=1
    t=0
    skeleton=np.transpose(data[0,:,[t],:,0], [1,0,2])
    if data.shape[-1]==2:
        skeleton2=np.transpose(data[0,:,[t],:,1], [1,0,2])
    else:
        skeleton2=skeleton*0

    bones=[[3,2,20,1,0,16,17,18,19],[0,12,13,14,15],[21,7,6,5,4,20,8,9,10,11,23]]

    for bone_list in bones:
        # if True:
        # joint_locs = skeleton[:,[i,j]]
        joint_locs = skeleton[:,0][:,bone_list]
        # plot them
        # (3,25), (3,2)
        ax.plot(joint_locs[0],joint_locs[1],joint_locs[2], color='blue')

        if skeleton2.sum()!=0:
            for bone_list in bones:

                joint_locs = skeleton2[:,0][:,bone_list]
                ax.plot(joint_locs[0],joint_locs[1],joint_locs[2], color='blue')


def draw_3d_skeleton(data_numpy, n_views, save_name, text_list=None):
    '''
    organized in form (n_views, n_samples)
    data_numpy: (n,c,1,v,m)
    '''
    N,C,T,V,M = data_numpy.shape
    n_samples = N
    figsize = (n_samples*2, n_views*2)

    rows = n_views
    cols = n_samples

    
    fig = plt.figure(figsize=figsize)
    ax_list=[]
    nn = n_samples

    for i in range(n_views*n_samples):
        ax_list.append(fig.add_subplot(rows,cols,i+1,projection='3d'))    

    for i in range(nn):
        
        data = data_numpy[i:i+1] 

        
        ax=ax_list[i]
        set_ax_view(ax, 10, 90)
        draw_skeleton_on_ax(ax, data)
        if text_list is not None:
            ax.set_title(text_list[i])

        ax=ax_list[i+cols]
        set_ax_view(ax, 10, 60)
        draw_skeleton_on_ax(ax, data)

        ax=ax_list[i+cols+cols]
        set_ax_view(ax, 10, 0)
        draw_skeleton_on_ax(ax, data)

        
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=100)
    print("save to", save_name)


def get_cluster_dist(X, n_clusters):
    '''
    X = (n,feature_dim)
    '''
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = kmeans.labels_
    cluster_labels = np.array(cluster_labels)
    # c = Counter(cluster_labels)
    # print(c)
    return cluster_labels
    


def flatten_nctvm(x):
    bs = x.shape[0]
    x = x.reshape((bs,-1))
    return x 


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



def get_dist_matrix_diff_shape(x, y):
    '''
    x,y: (n,c,t,v,m), torch.tensor
    '''
    # (n,t,c,v,m)
    n,c,t,v,m = x.shape
    n,c,t2,v,m = y.shape
    x = x.permute(0,2,1,3,4).contiguous()
    y = y.permute(0,2,1,3,4).contiguous()
    x = x.reshape(n,t,c*v*m)[0]
    y = y.reshape(n,t2,c*v*m)[0]

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    dist = ((x-y)**2).sum(dim=2)
    # print(dist.shape)
    return dist



def do_bkg_clustering_person(data_all, n_clusters, feature_tag):
    if feature_tag=='data':
        data = flatten_nctvm(data_all)
        cluster_labels = get_cluster_dist(data, n_clusters)
    if feature_tag=='feat':
        # using feature encoder 
        from feature_encoder_utils import get_feature_encoder
        import torch 
        model = get_feature_encoder(0)
        with torch.no_grad():
            data_feature = model(torch.FloatTensor(data_all).to(0))
            data_feature = data_feature.data.cpu().numpy()
        print("data_feature.shape = ", data_feature.shape)
        # sys.exit()
        cluster_labels = get_cluster_dist(data_feature, n_clusters)

    res=[]
    class_weight_list = []
    for c in np.unique(cluster_labels):
        selected_idx_list = np.where(cluster_labels==c)[0]
        data_c = data_all[selected_idx_list].mean(axis=0, keepdims=True)
        res.append(data_c)
        class_weight_list.append(selected_idx_list.shape[0])
    res = np.concatenate(res,0)
    print('cluster result = ', res.shape)
    cluster_res=res
    class_weight_list = np.array(class_weight_list)
    class_weight_list = class_weight_list/np.sum(class_weight_list)
    class_weight_list = np.round(class_weight_list,2).astype(str)

    return cluster_res


def do_bkg_clustering(prior_data_npy_name, st, n_clusters, feature_tag='data'):
    data_all_t = np.load(prior_data_npy_name)

    
    print(f"clustering bkg on {data_all_t.shape}")

    # st=0 
    # feature_tag='data'
    assert feature_tag in ['data', 'feat']

    res_all = []
    for st in [st]:
        for feature_tag in ['data']:

            data_all = data_all_t[:,:,st:st+1,:,:1]
            cluster_res_single = do_bkg_clustering_person(data_all, n_clusters, feature_tag)
            print('single person.shape = ', cluster_res_single.shape)
            cluster_res_single = np.concatenate([cluster_res_single, cluster_res_single*0], 4)

            if False:
                data_all = data_all_t[:,:,st:st+1,:,:]
                cluster_res_double = do_bkg_clustering_person(data_all, n_clusters, feature_tag)
                print('double person.shape = ', cluster_res_double.shape)
                res_all = np.concatenate([cluster_res_single, cluster_res_double], 0)
            res_all=cluster_res_single

            return res_all

            



def main():
    pass 
    



if __name__ == "__main__":
    main()
