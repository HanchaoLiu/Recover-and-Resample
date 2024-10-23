import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../main"))
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

from w_transforms import (f_croppad_torch_batch, f_cropresize_torch_batch, f2_linear)

from linear_extrapolation import get_first_frame_condition

from rotation_aug import random_rot_batch, random_rot_batch_2samples

from itertools import permutations
import random


class SpatialJigsaw:

    def __init__(self, num_classes=3):
        assert num_classes==3
        self.num_classes = num_classes
        self.label_list = list(range(self.num_classes))

    @staticmethod
    def spatial_cubism(data_numpy, order):
        '''
        (c,t,v,m)
        '''
        left_hand =  [8,9,10,11,23,24]
        right_hand = [4,5,6,7,21,22]
        left_foot =  [12,13,14,15]
        right_foot = [16,17,18,19]
        hand_ref = [20]
        foot_ref = [0]

        # data_numpy_ret = data_numpy.copy()
        data_numpy_ret = data_numpy.clone()
        
        if order==0:
            pass 
        
        elif order==1:
            
            data_numpy_ret[0:3,:,left_hand,:] = data_numpy[0:3,:,right_hand,:].clone()
            data_numpy_ret[0:3,:,right_hand,:] = data_numpy[0:3,:,left_hand,:].clone()
        elif order==2:
            left_hand, right_hand = left_foot, right_foot
            data_numpy_ret[0:3,:,left_hand,:] = data_numpy[0:3,:,right_hand,:].clone()
            data_numpy_ret[0:3,:,right_hand,:] = data_numpy[0:3,:,left_hand,:].clone()
        return data_numpy_ret

    def gen_data(self, data):
        n,c,t,v,m=data.shape 
        # frame: length list
        frame=[t]*n
        size=n
        trans_data = torch.zeros_like(data)
        labels = []

        for i in range(size):
            label = random.randint(0, self.num_classes)
            trans_data[i] = self.spatial_cubism(data[i], label)
            labels.append(label)     
            # print( shuffle_frames, label )
        labels = torch.LongTensor(labels)
        return trans_data, labels

    def gen_data_with_prob(self, data, ordered_prob=0.8):
        n,c,t,v,m=data.shape 
        # frame: length list
        frame=[t]*n
        size=n
        trans_data = torch.zeros_like(data)
        labels = []

        for i in range(size):
            # label = random.randint(0, self.num_classes)
            p_rest = (1-ordered_prob)/2.0
            label = np.random.choice([0,1,2], p=[ordered_prob,p_rest,p_rest])
            trans_data[i] = self.spatial_cubism(data[i], label)
            labels.append(label)     
            # print( shuffle_frames, label )
        labels = torch.LongTensor(labels)
        return trans_data, labels





###############################################################################################
# st-gcn has model.apply(weight_init)
# agcn does not have weight init
###############################################################################################

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


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

    parser.add_argument('--tensorboard', type=str2bool, default=False, help='use tensorboard or not')
    # parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score',type=str2bool,default=False,help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=0, help='random seed for pytorch')
    parser.add_argument('--log-interval',type=int,default=100,help='the interval for printing messages (#iteration)')

    parser.add_argument('--save-interval',type=int,default=2,help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval',type=int,default=5,help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    parser.add_argument('--show-topk',type=int,default=[1, 5],nargs='+',help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--train-feeder',default='feeder.feeder',help='data loader will be used')
    parser.add_argument('--train-feeder-args',action=DictAction,default=dict(),help='the arguments of data loader for training')
    parser.add_argument('--num-worker',type=int,default=1,help='the number of worker for data loader')
    parser.add_argument('--test-feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--test-feeder-args',action=DictAction,default=dict(),help='the arguments of data loader for test')

    parser.add_argument('--use-val', type=str2bool,default=False, help='data loader will be used')
    parser.add_argument('--val-feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--val-feeder-args',action=DictAction,default=dict(),help='the arguments of data loader for test')

    parser.add_argument('--train-feeder-name', default='', help='data loader will be used')
    parser.add_argument('--test-feeder-name', default='', help='data loader will be used')
    parser.add_argument('--val-feeder-name', default='', help='data loader will be used')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args',action=DictAction,default=dict(),help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')
    
    parser.add_argument('--weights_pretrained',default=None,help='the weights for network initialization')
    parser.add_argument('--weights_prior',default=None,help='the weights for network initialization')


    parser.add_argument('--ignore-weights',type=str,default=[],nargs='+',help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step',type=int,default=[20, 40, 60],nargs='+',help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch',type=int,default=0,help='start training from which epoch')
    parser.add_argument('--num-epoch',type=int,default=80,help='stop training in which epoch')
    parser.add_argument('--weight-decay',type=float,default=0.0005,help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)

    parser.add_argument('--use-aug', type=str2bool,default=False, help='data loader will be used')
    parser.add_argument('--weight_aug',type=float,default=0.0,help='weight decay for optimizer')
    parser.add_argument('--weight_dst',type=float,default=0.0,help='weight decay for optimizer')


    # large lr_max and small loops_adv
    parser.add_argument('--loops_adv',type=int,default=1,help='')
    parser.add_argument('--lr_max',type=float,default=1.0,help='')
    parser.add_argument('--gamma',type=float,default=1.0,help='')

    parser.add_argument('--scale',type=float,default=1.0,help='')

    # parser.add_argument('--type',type=str,default='',help='')

    parser.add_argument('--model_prior', default=None, help='the model will be used')
    parser.add_argument('--model_prior_args', default=None, help='the model will be used')
    parser.add_argument('--model_prior_weights', default=None, help='the model will be used')
    parser.add_argument('--model_prior_weights_dir', default=None, help='the model will be used')

    parser.add_argument('--recover_tag', default='none', help='data loader will be used')
    parser.add_argument('--resample_tag', default='none', help='data loader will be used')

    parser.add_argument('--n_cluster_w', type=int ,default=5,help='data loader will be used')
    parser.add_argument('--n_cluster_bkg', type=int ,default=5,help='data loader will be used')

    parser.add_argument('--prior_data_npy_name', type=str ,default='',help='data loader will be used')
    parser.add_argument('--prior_data_tag', type=str ,default='',help='data loader will be used')


    parser.add_argument('--n_seed', type=int ,default=3,help='data loader will be used')
    parser.add_argument('--src',type=str,default=['N','E'],nargs='+',help='the epoch where optimizer reduce the learning rate')
    

    parser.add_argument('--augRatio', type=float ,default=0.75,help='data loader will be used')

    parser.add_argument('--interp_type', type=str ,default='',help='data loader will be used')

    parser.add_argument('--bkg_type', type=str ,default='',help='data loader will be used')
    parser.add_argument('--tr_type', type=str ,default='',help='data loader will be used')

    parser.add_argument('--cluster_dir', type=str ,default='none',help='data loader will be used')
    parser.add_argument('--dataset_dir', type=str ,default='none',help='data loader will be used')


    parser.add_argument('--n_jigsaw',type=int,default=3,help='')
    parser.add_argument('--w_jigsaw',type=float,default=0.1,help='')

    return parser



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg

        self.spatial_jigsaw = SpatialJigsaw()
        self.num_label_jigsaw = 3 
        self.arg.model_args['n_jigsaw'] = self.num_label_jigsaw

        self.save_arg()
        if self.arg.tensorboard:
            if arg.phase == 'train':

                log_dir=os.path.join(self.arg.save_dir, "summary")
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
            
                self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(log_dir, 'val'), 'val')
                self.val_writer = SummaryWriter(os.path.join(log_dir, 'test'), 'test')
                
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

        self.vert_list = [4,5,6,7,8,9,10,11,20,22,23,24]

        



    def load_data(self):
        Feeder_train = import_class(self.arg.train_feeder)
        print(Feeder_train)
        Feeder_val = import_class(self.arg.val_feeder)
        Feeder_test = import_class(self.arg.test_feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder_train(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

            if self.arg.use_val:
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder_val(**self.arg.val_feeder_args),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    drop_last=False,
                    worker_init_fn=init_seed)
            
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder_test(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.model_pretrained = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)


        # weight init for st-gcn
        if self.arg.model.startswith("nets.stgcn"):
            self.model.apply(weights_init)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

        # model prior 
        if False:
            Model = import_class(self.arg.model_prior)
            self.print_log(f"Model_prior = {Model}")
            self.model_prior = Model(**self.arg.model_prior_args).cuda(self.output_device)
            print("not load_weights_pretrained ")
            # self.load_weights_pretrained(self.arg.model_prior_weights, self.model_prior, self.output_device)
            self.model_prior.eval()

        # model 
        # model_pretrained, weights_pretrained
        # model_prior, weights_prior, model_prior_pose_file

    @staticmethod
    def load_weights_pretrained(weights, model, output_device):
        if weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            print('Load weights from {}.'.format(weights))
            if '.pkl' in weights:
                with open(weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            try:
                model.load_state_dict(weights)
            except:
                state = model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                model.load_state_dict(state)
        

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    # def adjust_learning_rate(self, epoch):
    #     if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
    #         if epoch < self.arg.warm_up_epoch:
    #             lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
    #         else:
    #             if epoch <= 100:
    #                 lr = 1e-3 
    #             elif (epoch > 100) and (epoch <= 200):
    #                 lr = 1e-4 
    #             elif (epoch > 200) and (epoch <= 400):
    #                 lr = 5e-5
    #             else:
    #                 lr = 1e-5
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr
    #         return lr
    #     else:
    #         raise ValueError()

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            # localtime = time.asctime(time.localtime(time.time()))
            # str = "[ " + localtime + ' ] ' + str
            str = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


    def resize_torch_interp_batch(self, data, target_frame):

        window = target_frame

        n,c,t,v,m = data.shape 
        data = data.permute(0,1,3,4,2).contiguous().view(n,c*v*m,t)
        data = data[:,:,:,None]
        data = F.interpolate(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3)
        data = data.reshape(n,c,v,m,window)
        data = data.permute(0,1,4,2,3).contiguous()
        return data 




    # training and testing
    def train(self, epoch, save_model=False):


        self.print_log('-'*70)
        self.model.train()


        
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        lr = self.optimizer.param_groups[0]['lr']
        self.print_log('Training epoch: {}, lr = {}'.format(epoch, lr))

        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        if self.arg.tensorboard:
            self.train_writer.add_scalar('epoch', epoch, self.global_step)

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        # for batch_idx, (data, label, index) in enumerate(process):


        loss_src_list, loss_dst_list, loss_aug_list = [], [], []

        # eval advaug list 
        acc_aug_list = []
        acc_raw_list = []
        norm_aug_list = []

        coef_mat_list = []

        
        loss_diff_pred_interp_list = []
        loss_diff_pred_interp_list_1p = []
        for batch_idx, (data_src, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data_src = Variable(data_src.float().cuda(self.output_device), requires_grad=False)
            

            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            data_src = torch.cat([data_src],0)
            label = torch.cat([label], 0)

            # forward
            # output_src = self.model(data_src)
            # loss_src = self.loss(output_src, label)

            # loss_dst = torch.tensor(0.0).cuda(self.output_device)

            data_jigsaw, label_jig = self.spatial_jigsaw.gen_data_with_prob(data_src, ordered_prob=0.33)
            label_jig = label_jig.to(data_src.device)
            bs = data_jigsaw.shape[0]

            output_src, output_jig = self.model(torch.cat([data_src,data_jigsaw],0))
            label = torch.cat([label,label],0)
            loss_src  = self.loss(output_src, label)

            # _, output_jig = self.model(data_jigsaw)
            output_jig = output_jig[bs:bs*2]
            loss_dst = self.loss(output_jig, label_jig)



            loss_aug = torch.tensor(0.0).cuda(self.output_device)
            
            loss_src_list.append(loss_src.item())
            loss_dst_list.append(loss_dst.item())
            loss_aug_list.append(loss_aug.item() )
            
            
            loss = loss_src + loss_dst * self.arg.w_jigsaw
            

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()



            value, predict_label = torch.max(output_src.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            if self.arg.tensorboard:
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)
            
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            if self.arg.tensorboard:
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        self.print_log("loss_src={:.4f}, loss_dst={:.4f}, loss_aug={:.4f}".format(
            np.mean(loss_src_list), np.mean(loss_dst_list), np.mean(loss_aug_list)
        ))


        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            # torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')
            save_model_name = os.path.join(self.arg.work_dir, 'model_ep{}.pt'.format(epoch))
            torch.save(weights, save_model_name)

    def save_weights_model_prior(self, epoch):
        state_dict = self.model_prior.state_dict()
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in state_dict.items()])

        # torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')
        save_model_name = os.path.join(self.arg.work_dir, 'model_prior_ep{}.pt'.format(epoch))
        torch.save(weights, save_model_name)
        print("save to ", save_model_name)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            gt_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    

                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    output, _ = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    gt_frag.append(label.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            pred  = np.argmax(score, axis=1)
            gt    = np.concatenate(gt_frag)


            loss = np.mean(loss_value)
            # accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            accuracy = (pred==gt).mean()
            # if accuracy > self.best_acc:
            #     self.best_acc = accuracy
            # self.lr_scheduler.step(loss)
            self.result_dt[ln].append(accuracy)
            accuracy_bal = self.get_bal_acc(pred, gt)
            self.result_dt[ln+"_bal"].append(accuracy_bal)

            self.print_log('{} acccuracy = {:.4f}'.format(ln, accuracy) )
            if self.arg.phase == 'train':
                if self.arg.tensorboard:
                    self.val_writer.add_scalar('loss', loss, self.global_step)
                    self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                    self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # score_dict = dict(
            #     zip(self.data_loader[ln].dataset.sample_name, score))
            score_dict = list(
                zip(self.data_loader[ln].dataset.sample_name, score, gt ))
            # score_dict = None

            self.print_log('\tMean {} loss of {} batches: {:.5f}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            # for k in self.arg.show_topk:
            #     self.print_log('\tTop{}: {:.2f}%'.format(
            #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                score_dict_save_path = '{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln)
                with open(score_dict_save_path, 'wb') as f:
                    pickle.dump(score_dict, f)
                print("score dict save to ", score_dict_save_path)



    def eval_model_pretrained(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            gt_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    # data = Variable(
                    #     data.float().cuda(self.output_device),
                    #     requires_grad=False,
                    #     volatile=True)
                    # label = Variable(
                    #     label.long().cuda(self.output_device),
                    #     requires_grad=False,
                    #     volatile=True)

                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    output = self.model_pretrained(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    gt_frag.append(label.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            pred  = np.argmax(score, axis=1)
            gt    = np.concatenate(gt_frag)


            loss = np.mean(loss_value)
            # accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            accuracy = (pred==gt).mean()
            # if accuracy > self.best_acc:
            #     self.best_acc = accuracy
            # self.lr_scheduler.step(loss)
            # self.result_dt[ln].append(accuracy)
            # accuracy_bal = self.get_bal_acc(pred, gt)
            # self.result_dt[ln+"_bal"].append(accuracy_bal)

            self.print_log('{} acccuracy = {:.4f}'.format(ln, accuracy) )
            if self.arg.phase == 'train':
                if self.arg.tensorboard:
                    self.val_writer.add_scalar('loss', loss, self.global_step)
                    self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                    self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # score_dict = dict(
            #     zip(self.data_loader[ln].dataset.sample_name, score))
            score_dict = list(
                zip(self.data_loader[ln].dataset.sample_name, score, gt ))
            # score_dict = None

            self.print_log('\tMean {} loss of {} batches: {:.5f}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            # for k in self.arg.show_topk:
            #     self.print_log('\tTop{}: {:.2f}%'.format(
            #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                score_dict_save_path = '{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln)
                with open(score_dict_save_path, 'wb') as f:
                    pickle.dump(score_dict, f)
                print("score dict save to ", score_dict_save_path)



    # save result utils
    def show_acc_res(self, save_res=True, show_file_name=False):
        # self.print_log(f"{self.result_dt['val']}")
        # self.print_log(f"{self.result_dt['test']}")

        test_acc_list = np.array(self.result_dt['test'])
        best_test_acc = np.max(test_acc_list)
        best_test_idx = np.argmax(test_acc_list)

        if self.arg.use_val:
            val_acc_list = np.array(self.result_dt['val'])
            best_val_acc = np.max(val_acc_list)
            best_val_idx = np.argmax(val_acc_list)

        if self.arg.use_val:
            best_test_acc_at_val = test_acc_list[best_val_idx]
            result_str = "best_val_acc={:.3f} @ ep{}, best_test_acc={:.3f} @ ep{}, best_test_acc@val={:.3f} @ ep{}".format(
                best_val_acc, best_val_idx,
                best_test_acc, best_test_idx,
                best_test_acc_at_val, best_val_idx
            )
        else:
            result_str = "best_test_acc={:.3f} @ep {}".format(
                best_test_acc, best_test_idx
            )
        self.print_log(result_str)
        self.print_log(f"train={self.arg.train_feeder_name},val={self.arg.val_feeder_name},test={self.arg.test_feeder_name}")

        if save_res:
            save_name_pkl = os.path.join(self.arg.base_dir, "result.pkl")
            with open(save_name_pkl, "wb") as f:
                pickle.dump(self.result_dt, f)
            
            

            save_name = os.path.join(self.arg.base_dir, "result.txt")
            result_str = result_str + ", last_test_acc = {:.3f}".format(test_acc_list[-1])
            with open(save_name, "w") as f:
                f.write(result_str+"\n")

            save_plot_name = os.path.join(self.arg.base_dir, "result.png")
            if False:
                if self.arg.use_val:
                    from vis.curve import plot_curve_test_val
                    plot_curve_test_val(save_plot_name, self.result_dt['test'], self.result_dt['val'], self.arg.seed)
                else:
                    from vis.curve import plot_curve_test
                    plot_curve_test(save_plot_name, self.result_dt['test'], self.arg.seed)

            if show_file_name:
                self.print_log(f"result save to {save_name_pkl}")
                self.print_log(f"result save to {save_name}")
                self.print_log(f"log save to {self.print_file_name}")
                self.print_log(f"fig save to {save_plot_name}")


            # + save best model
            best_model_name = os.path.join(self.arg.work_dir, 'model_ep{}.pt'.format(best_test_idx))
            best_model_name_save = os.path.join(self.arg.work_dir, 'model_epbest.pt')
            shutil.copy(best_model_name, best_model_name_save)

            print("model save to:", best_model_name_save)



            # + save best model with best@val_idx, 1101 new add.
            best_model_name = os.path.join(self.arg.work_dir, 'model_ep{}.pt'.format(best_val_idx))
            best_model_name_save = os.path.join(self.arg.work_dir, 'model_epbest_atval.pt')
            shutil.copy(best_model_name, best_model_name_save)
            print("best@val model save to:", best_model_name_save)

    

    # def get_bal_acc(self, pred, gt):
    #     perclass_acc_list = []
    #     pred = np.array(pred)
    #     gt = np.array(gt)
    #     unique_label = np.unique(gt)
    #     for c in unique_label:
    #         acc = (pred[gt==c]==gt[gt==c]).mean()
    #         perclass_acc_list.append( acc )
    #     perclass_acc_mean = np.mean(perclass_acc_list)
    #     return perclass_acc_mean

    def get_bal_acc(self, pred, gt):
        perclass_acc_list = []
        pred = np.array(pred)
        gt = np.array(gt)
        # unique_label = np.unique(gt)
        # for c in unique_label:
        #     acc = (pred[gt==c]==gt[gt==c]).mean()
        #     perclass_acc_list.append( acc )
        # perclass_acc_mean = np.mean(perclass_acc_list)
        perclass_acc_mean = np.mean(pred==gt)
        return perclass_acc_mean

    def show_acc_res_all(self, tag='', save_res=True, show_file_name=False):

        def get_acc_and_idx(acc_list):
            best_acc = np.max(acc_list)
            best_idx = np.argmax(acc_list)
            return best_acc, best_idx
        
        if tag=='bal':
            test_acc_list     = np.array(self.result_dt[f'test_{tag}'])
            val_acc_list      = np.array(self.result_dt[f'val_{tag}'])
        else:
            test_acc_list     = np.array(self.result_dt[f'test'])
            val_acc_list      = np.array(self.result_dt[f'val'])

        best_val_acc, best_val_idx = get_acc_and_idx(val_acc_list)
        best_test_acc, best_test_idx = get_acc_and_idx(test_acc_list)

        best_mean_acc_seperate = (best_val_acc + best_test_acc) / 2

        acc_mean_list = (test_acc_list+val_acc_list)/2
        best_mean_acc_combined, best_mean_idx_combined = get_acc_and_idx(acc_mean_list)

        last_val_acc = val_acc_list[-1]
        last_test_acc= test_acc_list[-1]

        last_idx = len(val_acc_list)-1

        result_str = f"({tag})" + \
            f"best_val_acc={best_val_acc:.3f} @ ep{best_val_idx}, best_test_acc={best_test_acc:.3f} @ ep{best_test_idx}, " +\
            f"last_val_acc={last_val_acc:.3f} @ ep{last_idx}, last_test_acc={last_test_acc:.3f} @ ep{last_idx}, " +\
            f"best_mean_acc_sep={best_mean_acc_seperate:.3f}, best_mean_acc_comb={best_mean_acc_combined:.3f} @ ep{best_mean_idx_combined}"

        self.print_log(result_str)
        self.print_log(f"train={self.arg.train_feeder_name},val={self.arg.val_feeder_name},test={self.arg.test_feeder_name}")

        if tag=='bal':
            self.print_log("val(bal):"+str(np.round(val_acc_list*100,1)))
            self.print_log("test(bal):"+str(np.round(test_acc_list*100,1)))

        # save txt and save pkl here.
        if save_res:
            result_str = result_str + "\n" + f"train={self.arg.train_feeder_name},val={self.arg.val_feeder_name},test={self.arg.test_feeder_name}"
            dt = {
                'best_val_acc': best_val_acc,
                'best_val_idx': best_val_idx,
                'best_test_acc':best_test_acc,
                'best_test_idx':best_test_idx,
                'last_val_acc':last_val_acc, 
                'last_test_acc':last_test_acc,
                'best_mean_acc_sep':best_mean_acc_seperate,
                'best_mean_acc_comb':best_mean_acc_combined,
                'train_name': self.arg.train_feeder_name, 
                'val_name': self.arg.val_feeder_name,
                'test_name':self.arg.test_feeder_name,
            }

            if tag=='bal':
                save_name_pkl = os.path.join(self.arg.base_dir, f"result_{tag}.pkl")
                save_name = os.path.join(self.arg.base_dir, f"result_{tag}.txt")
            else:
                save_name_pkl = os.path.join(self.arg.base_dir, f"result.pkl")
                save_name = os.path.join(self.arg.base_dir, f"result.txt")

            with open(save_name_pkl, "wb") as f:
                pickle.dump(dt, f)
            with open(save_name, "w") as f:
                f.write(result_str+"\n")

        dt = {
                'best_val_acc': best_val_acc,
                'best_val_idx': best_val_idx,
                'best_test_acc':best_test_acc,
                'best_test_idx':best_test_idx,
                'last_val_acc':last_val_acc, 
                'last_test_acc':last_test_acc,
                'best_mean_acc_sep':best_mean_acc_seperate,
                'best_mean_acc_comb':best_mean_acc_combined,
                'train_name': self.arg.train_feeder_name, 
                'val_name': self.arg.val_feeder_name,
                'test_name':self.arg.test_feeder_name,
                'tag': tag
            }

        return dt

    

    def load_best_model(self, loader_name):

        dt = self.dt_final_bal
        best_val_idx = dt['best_val_idx']
        best_test_idx = dt['best_test_idx']
        if loader_name=='val':
            best_idx = best_val_idx
        elif loader_name=='test':
            best_idx = best_test_idx
        else:
            raise ValueError()

        model_name = os.path.join(self.arg.work_dir, 'model_ep{}.pt'.format(best_idx))
        assert os.path.exists(model_name)
        
        self.load_weights_pretrained(model_name, self.model, self.output_device)
        self.model.eval()


    def eval_and_save(self, loader_name='test', save_tag=None):
        
        self.model.eval()
        ln=loader_name
        loss_value = []
        score_frag = []
        gt_frag = []
        right_num_total = 0
        total_num = 0
        loss_total = 0
        step = 0
        process = tqdm(self.data_loader[ln])
        for batch_idx, (data, label, index) in enumerate(process):
            with torch.no_grad():
                

                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

                # output = self.model(data)
                output, _ = self.model(data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.item())
                gt_frag.append(label.data.cpu().numpy())

                _, predict_label = torch.max(output.data, 1)
                step += 1

        score = np.concatenate(score_frag)
        pred  = np.argmax(score, axis=1)
        gt    = np.concatenate(gt_frag)


        loss = np.mean(loss_value)
        # accuracy = self.data_loader[ln].dataset.top_k(score, 1)
        accuracy = (pred==gt).mean()
       
        accuracy_bal = self.get_bal_acc(pred, gt)
        # self.result_dt[ln+"_bal"].append(accuracy_bal)

        self.print_log('{} acccuracy = {:.4f}'.format(ln, accuracy_bal) )
        
        # score_dict = dict(
        #     zip(self.data_loader[ln].dataset.sample_name, score))
        score_dict = list(
            zip(self.data_loader[ln].dataset.sample_name, score, gt ))
        # score_dict = None

        self.print_log('\tMean {} loss of {} batches: {:.5f}.'.format(
            ln, len(self.data_loader[ln]), np.mean(loss_value)))
        # for k in self.arg.show_topk:
        #     self.print_log('\tTop{}: {:.2f}%'.format(
        #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

        save_score = True
        if save_score:
            score_dict_save_path = '{}/{}_epoch{}_score.pkl'.format(
                    self.arg.work_dir, save_tag, 'best')
            with open(score_dict_save_path, 'wb') as f:
                pickle.dump(score_dict, f)
            print("score dict save to ", score_dict_save_path)




    def start(self):

        # remove old log dir
        self.print_file_name = '{}/log.txt'.format(self.arg.work_dir)
        if os.path.exists(self.print_file_name):
            os.remove(self.print_file_name)
        
        import json
        # a=json.dumps(vars(self.arg), indent=2)
        a=json.dumps(vars(self.arg))
        self.print_log(a)
        self.print_log("work_dir = "+self.arg.work_dir)
        self.print_log("base_dir = "+self.arg.base_dir)

        # self.result_dt = {'val': [], 'test': []}
        self.result_dt = {'val': [], 'test': [], 'val_bal':[], 'test_bal':[]}

        if self.arg.phase == 'train':
            # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # if self.lr < 1e-3:
                #     break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    if self.arg.use_val:
                        self.eval( epoch, save_score=self.arg.save_score, loader_name=['val'])
                    self.eval( epoch, save_score=self.arg.save_score, loader_name=['test'])

                # self.show_acc_res(save_res=True, show_file_name=False)

                self.show_acc_res_all(tag='',save_res=True, show_file_name=False)
                self.show_acc_res_all(tag='bal',save_res=True, show_file_name=False)

            # self.show_acc_res(save_res=True, show_file_name=True)
            self.show_acc_res_all(tag='',save_res=True, show_file_name=False)
            self.dt_final_bal = self.show_acc_res_all(tag='bal',save_res=True, show_file_name=False)

            # + delete other models
            # model_name_to_remove = glob.glob(os.path.join(self.arg.work_dir, "model_ep*.pt"))
            # model_name_to_remove = [name for name in model_name_to_remove if 'best' not in name]
            # for name in model_name_to_remove:
            #     os.remove(name)

            self.load_best_model(loader_name='val')
            self.eval_and_save(loader_name='val', save_tag=f"{self.arg.train_feeder_name}_{self.arg.val_feeder_name}")

            self.load_best_model(loader_name='test')
            self.eval_and_save(loader_name='test', save_tag=f"{self.arg.train_feeder_name}_{self.arg.test_feeder_name}")
            
            # + copy log file to result_dir
            old_log_file = '{}/log.txt'.format(self.arg.work_dir)
            new_log_file = '{}/log.txt'.format(self.arg.base_dir)
            shutil.copy(old_log_file, new_log_file)


        elif self.arg.phase == 'test':
            # if not self.arg.test_feeder_args['debug']:
            if False:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

            sys.exit(0)


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

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    work_dir=arg.work_dir
    arg.work_dir = os.path.join(cfg.work_dir_base, work_dir)
    arg.base_dir = os.path.join(cfg.base_dir_base, work_dir)
    for k in arg.train_feeder_args.keys():
        if k.startswith('data_path') or k.startswith('label_path'):
            arg.train_feeder_args[k] = os.path.join( cfg.data_dir_base, arg.train_feeder_args[k] )


    if not os.path.exists(arg.work_dir):
        os.mkdir(arg.work_dir)
    if not os.path.exists(arg.base_dir):
        os.mkdir(arg.base_dir)

    if arg.use_val:
        arg.val_feeder_args['data_path'] = os.path.join( cfg.data_dir_base, arg.val_feeder_args['data_path'] )
        arg.val_feeder_args['label_path'] = os.path.join( cfg.data_dir_base, arg.val_feeder_args['label_path'] )

    arg.test_feeder_args['data_path'] = os.path.join( cfg.data_dir_base, arg.test_feeder_args['data_path'] )
    arg.test_feeder_args['label_path'] = os.path.join( cfg.data_dir_base, arg.test_feeder_args['label_path'] )

    processor = Processor(arg)
    processor.start()



def main_with_seed():

    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    dataset_list = arg.src 
    work_dir_0 = arg.work_dir
    dt_all_dataset = {}
    for each_dataset in dataset_list:
        # set_dataset(arg, each_dataset)

        dt_list=[]

        seed_list = list(range(arg.n_seed))
        for seed in seed_list:
            # add these two lines
            arg.seed=seed 
            arg.work_dir=work_dir_0+f"_{each_dataset}_seed{seed}"

            init_seed(arg.seed)
            work_dir=arg.work_dir
            print(f"work_dir={work_dir}")

            set_dataset(arg, each_dataset)


            arg.work_dir = os.path.join(cfg.work_dir_base, work_dir)
            arg.base_dir = os.path.join(cfg.base_dir_base, work_dir)
            for k in arg.train_feeder_args.keys():
                if k.startswith('data_path') or k.startswith('label_path'):
                    arg.train_feeder_args[k] = os.path.join( arg.dataset_dir, arg.train_feeder_args[k] )
                    
            

            if not os.path.exists(arg.work_dir):
                os.mkdir(arg.work_dir)
            if not os.path.exists(arg.base_dir):
                os.mkdir(arg.base_dir)

            if arg.use_val:
                arg.val_feeder_args['data_path'] = os.path.join( arg.dataset_dir, arg.val_feeder_args['data_path'] )
                arg.val_feeder_args['label_path'] = os.path.join( arg.dataset_dir, arg.val_feeder_args['label_path'] )
                

            arg.test_feeder_args['data_path'] = os.path.join( arg.dataset_dir, arg.test_feeder_args['data_path'] )
            arg.test_feeder_args['label_path'] = os.path.join( arg.dataset_dir, arg.test_feeder_args['label_path'] )
            

            processor = Processor(arg)
            processor.start()


            dt_final_this_seed = processor.dt_final_bal
            dt_list.append(dt_final_this_seed)

        if False:
            
            # save cls report
            dt_res, save_path = get_result_report_all('', os.path.join(cfg.base_dir_base, work_dir_0), seed_list)
            processor.print_log(f'save to {save_path}')
            processor.print_log('\n'+str(dt_res))

            dt_res, save_path = get_result_report_all('bal', os.path.join(cfg.base_dir_base, work_dir_0), seed_list)
            processor.print_log(f'save to {save_path}')
            processor.print_log('\n'+str(dt_res))


        dt_list_merged = merge_and_print(dt_list)
        print(dt_list_merged)

        dt_all_dataset[each_dataset] = dt_list_merged

    print("\n\nFinal Result:")
    for k,v in dt_all_dataset.items():
        print("="*150)
        print(k)
        print(v)

    save_dir = os.path.join(cfg.base_dir_base, work_dir_0)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, "result_all.pkl")
    with open(save_path, 'wb') as f:
        # print(df, file=f)
        pickle.dump(dt_all_dataset, f)
    print("save to", save_path)

    
def merge_and_print(dt_list):
    '''
    dt_list = [dt1, dt2, dt_n,]
    '''
    import pandas as pd 
    dt0 = dt_list[0]
    keys= list(dt0.keys())
    dt_all={k:[] for k in keys}
    for k in keys:
        for d in dt_list:
            dt_all[k].append(d[k])
    df=pd.DataFrame(dt_all)
    mean_row = df.mean()
    std_row  = df.std(ddof=0)
    df.loc['mean'] = mean_row
    df.loc['std'] =  std_row
    df=df.round(3)
    return df 




def set_dataset(arg, each_dataset):

    if each_dataset=='N51':
        arg.train_feeder_name="n51"
        arg.train_feeder_args['data_path'] ="ntu/all_data_joint.npy"
        arg.train_feeder_args['label_path']="ntu/all_label.pkl"

        arg.val_feeder_name="n51"
        arg.val_feeder_args['data_path'] ="ntu/test_data_joint.npy"
        arg.val_feeder_args['label_path']="ntu/test_label.pkl"

        arg.test_feeder_name="p51"
        arg.test_feeder_args['data_path']= "pku/test_data_joint.npy"
        arg.test_feeder_args['label_path']="pku/test_label.pkl"


    elif each_dataset=='P51':
        arg.train_feeder_name="p51"
        arg.train_feeder_args['data_path'] ="pku/all_data_joint.npy"
        arg.train_feeder_args['label_path']="pku/all_label.pkl"

        arg.val_feeder_name="p51"
        arg.val_feeder_args['data_path'] ="pku/test_data_joint.npy"
        arg.val_feeder_args['label_path']="pku/test_label.pkl"

        arg.test_feeder_name="n51"
        arg.test_feeder_args['data_path'] ="ntu/test_data_joint.npy"
        arg.test_feeder_args['label_path']="ntu/test_label.pkl"

    else:
        raise ValueError()




    
def get_result_report_all(tag, file_dir, seed_list):

    import pandas as pd 


    def parse_one_pkl(file_name, seed):
        data = np.load(file_name, allow_pickle=True)
        df = pd.DataFrame(data, index=[seed])
        return df
    

    df_list = []
    for seed in seed_list:
        if tag=='bal':
            file_name = file_dir+f"_seed{seed}/result_bal.pkl"
        else:
            file_name = file_dir+f"_seed{seed}/result.pkl"
        print(file_name)
        assert os.path.exists(file_name)
        df = parse_one_pkl(file_name, seed)
        df_list.append(df)
    
    df_res = pd.concat(df_list)
    df = df_res

    # print(df)
    pd.set_option("display.precision", 3)
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std(ddof=0)

    if tag=='bal':
        save_path = file_dir+f"_seed{seed_list[-1]}/results_bal_all.txt"
    else:
        save_path = file_dir+f"_seed{seed_list[-1]}/results_all.txt"
    with open(save_path, 'w') as f:
        print(df, file=f)
    print("### result_all save to ", save_path)


    if tag=='bal':
        save_path = file_dir+f"_seed{seed_list[-1]}/results_bal_all.pkl"
    else:
        save_path = file_dir+f"_seed{seed_list[-1]}/results_all.pkl"
    with open(save_path, 'wb') as f:
        # print(df, file=f)
        pickle.dump(df, f)
    print("### result_all save to ", save_path)




    return df, save_path

    
def get_result_report(file_dir, seed_list):

    import pandas as pd 

    def parse_line(line):
        res_list = []
        str_list = line.split(',')
        for it in str_list:
            it = it.strip().split('=')[1].strip()
            it = it.split('@')[0]
            it = it.strip()
            it = float(it)
            res_list.append(it)

        best_val_acc = res_list[0]
        best_test_acc = res_list[1]
        best_test_acc_at_val = res_list[2]
        last_test_acc = res_list[3]
        dt = {
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'best_test_acc@val': best_test_acc_at_val,
            'last_test_acc': last_test_acc
        }
        return dt 

    def parse_all_seeds(all_lines):
        dt_list = []
        for line in all_lines:
            dt_list.append(parse_line(line))

        keys = list(dt_list[0].keys())
        dt_res = {
            k:[] for k in keys
        }
        for dt in dt_list:
            for k,v in dt.items():
                dt_res[k].append(v)
        return dt_res

    line_list = []
    for seed in seed_list:
        file_name = file_dir+f"_seed{seed}/result.txt"
        print(file_name)
        assert os.path.exists(file_name)
        line_str = open(file_name, "r").readlines()[0]
        line_list.append(line_str)
    dt_res = parse_all_seeds(line_list)

    df = pd.DataFrame(dt_res)
    # print(df)
    pd.set_option("display.precision", 3)
    df.loc['mean'] = df.mean()
    save_path = file_dir+f"_seed{seed_list[-1]}/results_all.txt"
    with open(save_path, 'w') as f:
        print(df, file=f)
    print("### result_all save to ", save_path)
    return df, save_path





if __name__ == '__main__':
    
    # main()
    main_with_seed()


