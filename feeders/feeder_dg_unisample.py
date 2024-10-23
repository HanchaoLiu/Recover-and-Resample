import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

import torch.nn.functional as F

import random

# sys.path.extend(['../'])
from . import tools
from . import rotation_aug
from . import aimclr_tools_aug
from . import spa_mag
from .scale_limbs import random_scale_limb




class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        if len(deprecated_kwargs):
            warning_r0('[UniformSampleFrames] The following args has been deprecated: ')
            for k, v in deprecated_kwargs.items():
                warning_r0(f'Arg name: {k}; Arg value: {v}')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def do_sample(self, num_frames, tag='train'):
        if tag=='train':
            inds = self._get_train_clips(num_frames, self.clip_len)
        elif tag=='test':
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            raise ValueError()
        inds = np.mod(inds, num_frames)
        # start_index = results['start_index']
        start_index = 0
        inds = inds + start_index
        return inds

    def __call__(self, results):
        num_frames = results['total_frames']

        if results.get('test_mode', False):
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'seed={self.seed})')
        return repr_str




class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=False, 
                 num_person=2,theta=30,**kwargs):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.num_person = num_person
        self.theta=theta

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        assert self.data.shape[0]==len(self.label)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.num_person==1:
            data_numpy = data_numpy[...,:1]

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def getitem(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.num_person==1:
            data_numpy = data_numpy[...,:1]

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index


    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


class Feeder_rot(Feeder):
    def __getitem__(self, index):
        data_numpy, label, index = self.getitem(index)
        data_numpy = rotation_aug.my_random_rot(data_numpy, self.theta)
        return data_numpy, label, index


# --train-feeder "feeders.feeder_dg_unisample.Feeder_rot_unisample_train" \
# --val-feeder "feeders.feeder_dg_unisample.Feeder_unisample_test" \
# --test-feeder "feeders.feeder_dg_unisample.Feeder_unisample_test" \


class Feeder_rot_unisample_train(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=False, 
                 num_person=2,theta=30,**kwargs):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.sampler = UniformSampleFrames(64, num_clips=1, seed=0)

        self.num_person = num_person
        self.theta=theta

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        assert self.data.shape[0]==len(self.label)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def get_len_from_seq(self, d):
        # c,t,v,m
        d_frame0 = d[:,0:1, :, :]
        d_frame0_expand = np.repeat(d_frame0, repeats=d.shape[1], axis=1)
        # print(d.shape, d_frame0_expand.shape)
        
        n = d.shape[1]
        d = np.transpose(d, [1,0,2,3]).reshape((n,-1))
        d_frame0_expand = np.transpose(d_frame0_expand, [1,0,2,3]).reshape((n,-1))
        # print(d.shape, d_frame0_expand.shape)

        perframe_diff = np.linalg.norm(d - d_frame0_expand, axis=1)
        # print(perframe_diff.shape)
        # print(perframe_diff)

        idx_0_list = np.where(perframe_diff==0)[0]
        # print(idx_0_list)
        if len(idx_0_list)==1:
            res = n
        elif len(idx_0_list)==2:
            assert idx_0_list[0] == 0
            res = idx_0_list[1]
        else:
            assert idx_0_list[0] == 0
            diff_list = np.diff(idx_0_list)
            # print(idx_0_list)
            assert np.all(diff_list==diff_list[0])
            res = idx_0_list[1]
        return res 

    def __getitem__(self, index):
        data_numpy, label, index = self.getitem(index)
        shape0=data_numpy.shape

        # do uniform sampling
        num_frames = self.get_len_from_seq(data_numpy)
        train_idx = self.sampler.do_sample(num_frames, tag='train')
        data_numpy = data_numpy[:, train_idx, :, :]
        # print(num_frames, shape0, data_numpy.shape, train_idx)

        data_numpy = rotation_aug.my_random_rot(data_numpy, self.theta)
        
        return data_numpy, label, index

    def getitem(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.num_person==1:
            data_numpy = data_numpy[...,:1]

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index


    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)



class Feeder_unisample_test(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=False, 
                 num_person=2,theta=30,**kwargs):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.sampler = UniformSampleFrames(64, num_clips=1, seed=0)
        # self.sampler = UniformSampleFrames(64, num_clips=10, seed=0)

        self.num_person = num_person
        self.theta=theta

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        assert self.data.shape[0]==len(self.label)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def get_len_from_seq(self, d):
        # c,t,v,m
        d_frame0 = d[:,0:1, :, :]
        d_frame0_expand = np.repeat(d_frame0, repeats=d.shape[1], axis=1)
        # print(d.shape, d_frame0_expand.shape)
        
        n = d.shape[1]
        d = np.transpose(d, [1,0,2,3]).reshape((n,-1))
        d_frame0_expand = np.transpose(d_frame0_expand, [1,0,2,3]).reshape((n,-1))
        # print(d.shape, d_frame0_expand.shape)

        perframe_diff = np.linalg.norm(d - d_frame0_expand, axis=1)
        # print(perframe_diff.shape)
        # print(perframe_diff)

        idx_0_list = np.where(perframe_diff==0)[0]
        # print(idx_0_list)
        if len(idx_0_list)==1:
            res = n
        elif len(idx_0_list)==2:
            assert idx_0_list[0] == 0
            res = idx_0_list[1]
        else:
            assert idx_0_list[0] == 0
            diff_list = np.diff(idx_0_list)
            # print(idx_0_list)
            assert np.all(diff_list==diff_list[0])
            res = idx_0_list[1]
        return res 

    def __getitem__(self, index):
        data_numpy, label, index = self.getitem(index)
        shape0=data_numpy.shape

        # do uniform sampling
        num_frames = self.get_len_from_seq(data_numpy)
        test_idx = self.sampler.do_sample(num_frames, tag='test')
        data_numpy = data_numpy[:, test_idx, :, :]
        # print(num_frames, shape0, data_numpy.shape, test_idx)

        # data_numpy = rotation_aug.my_random_rot(data_numpy, self.theta)
        
        return data_numpy, label, index

    def getitem(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.num_person==1:
            data_numpy = data_numpy[...,:1]

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index


    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)






class Feeder_spaMag(Feeder):

    def __getitem__(self, index):
        data_numpy, label, index = self.getitem(index)

        method='spaMag'
        if random.random()>0.5:

            if method=='spaMag':
                bone_list = [3,4,5,6,7,8,9,10,20,21,22,23]
                data_numpy[...,:1]=spa_mag.spatial_motion_mag_spa_tem(data_numpy[...,:1], bone_list, 
                    k_spa=3, k_tem=1.0, tem_type='none')
            elif method=='spaMagAll':
                bone_list = list(range(24))
                data_numpy[...,:1]=spa_mag.spatial_motion_mag_spa_tem(data_numpy[...,:1], bone_list, 
                    k_spa=3, k_tem=1.0, tem_type='none')
            else:
                raise ValueError()

        data_numpy = rotation_aug.my_random_rot(data_numpy, self.theta)
        return data_numpy, label, index

    





