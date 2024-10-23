import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from config import cfg 

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

import glob

from IPython import embed




def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval('dict({})'.format(values))  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed for pytorch')
    parser.add_argument('--prior_data_dir', type=str ,default='',help='data loader will be used')
    parser.add_argument('--save_res_dir', type=str ,default='',help='data loader will be used')

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg



    def do_clustering_save_result(self):

        os.makedirs(self.arg.save_res_dir, exist_ok=True)

        
        # cluster bkg poses.
        from boundary_pose_clustering import do_bkg_clustering
        from linear_transform_clustering import do_w_clustering

        name_dt = {
            'N': 'ntu',
            'E': 'etriA',
            'PPrior': 'pku_prior',
            'P': 'pku'
        }

        # split_list = ['ntu', 'etriA', 'pku_prior', 'pku']

        split_list = ['ntu', 'etriA', 'pku']
        n_clusters_bkg_list = [5, 10, 20]
        n_cluster_w_list = [1,3,5,10,20]
        



        for split in split_list:
            
            prior_data_npy_name = f'{self.arg.prior_data_dir}/{split}/train_data_joint.npy'

            
            
            # bkg 
            for n_clusters_bkg in n_clusters_bkg_list:

                print(prior_data_npy_name)
                st=0
                feature_tag='data'
                bkg_clusters=do_bkg_clustering(prior_data_npy_name, st, n_clusters_bkg, feature_tag)
                bkg_clusters=bkg_clusters[:,:,:,:,:1]
                # (4, 3, 1, 25, 1)

                pose_clusters = np.concatenate([bkg_clusters, np.zeros_like(bkg_clusters)], 4)
                
                save_name = os.path.join(self.arg.save_res_dir, f'bkg_cluster_poses_{split}_{n_clusters_bkg}.npy')

                print(split, pose_clusters.shape)
                print("save to", save_name)
                np.save(save_name, pose_clusters)

            # tr
            for n_clusters in n_cluster_w_list:
                # for T1_each in [1, 0.1, 0.01, 0.001]:
                for T1_each in [0.1]:
                    segment_list = [
                    (0.0, 1.0), 
                    (0.0, 0.5), (0.5, 1.0), (0.25, 0.75), (0.125, 0.625), (0.375, 0.875),
                    (0.0, 0.75), (0.25, 1.0)
                    ]
                    
                    conf_mat_list, traj_mat_list, traj_weight_list, fig = do_w_clustering(prior_data_npy_name, n_clusters, T1=T1_each, topk=32, use_bkg=False, save_name=None, 
                        use_sample=False, segment_list=segment_list, bkg_pose_list=None)

                    save_name = os.path.join(self.arg.save_res_dir, f'tr_cluster_{split}_{n_clusters}_T{T1_each}.npy')

                    print(split, traj_mat_list.shape)
                    print("save to", save_name)
                    np.savez(save_name, conf_mat_list=conf_mat_list, traj_mat_list=traj_mat_list, traj_weight_list=traj_weight_list)


        


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def main():

    parser = get_parser()
    arg = parser.parse_args()

    init_seed(arg.seed)

    processor = Processor(arg)
    processor.do_clustering_save_result()



if __name__ == '__main__':
    main()


