import os, sys 
import numpy as np 
from tqdm import tqdm
import cv2 
import argparse
import shutil


def real_resize(data_numpy, n_frames, target_frame):
    # n_frame = length of nonzero 
    length = n_frames
    crop_size = target_frame
    C, T, V, M = data_numpy.shape
    new_data = np.zeros([C, crop_size, V, M])
    for i in range(M):
        tmp = cv2.resize(data_numpy[:, :length, :, i].transpose(
            [1, 2, 0]), (V, crop_size), interpolation=cv2.INTER_LINEAR)
        tmp = tmp.transpose([2, 0, 1])
        new_data[:, :, :, i] = tmp
    return new_data.astype(np.float32)


def downsample_with_seqlen(data, target_frame):
    '''
    data: (n,c,t,v,m)
    return: (n,c,t,v,m)
    '''

    def get_len_from_seq(d):
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

    n_samples=len(data)
    len_list = [get_len_from_seq(d) for d in data]
    # data_reduced = np.stack([slice_temporal_fn(data[k], len_list[k], target_frame) for k in range(n_samples)],0)
    # data_reduced = np.stack([real_resize(data[k], len_list[k], target_frame) for k in range(n_samples)],0)
    data_reduced = []
    for k in tqdm(range(n_samples)):
        data_reduced.append(
            real_resize(data[k], len_list[k], target_frame)
        )
    data_reduced = np.stack(data_reduced,0)
    return data_reduced



def f_300_to_64(data300_file, label300_file, data64_file, label64_file):

    print("loading", data300_file, label300_file)
    data300 = np.load(data300_file)
    data64 = downsample_with_seqlen(data300, 64)

    dirname = os.path.dirname(data64_file)
    os.makedirs(dirname, exist_ok=True)

    np.save(data64_file, data64)
    print("-> save to ", data64_file)
    shutil.copy(label300_file, label64_file)
    print("-> save to", label64_file)



def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--input_dir', default='none', help='')
    parser.add_argument('--output_dir', default='none', help='')

    return parser





def main():

    args = get_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    

    assert os.path.exists( os.path.join(args.input_dir, "train_data_joint.npy"))
    assert os.path.exists( os.path.join(args.input_dir, "test_data_joint.npy"))
    assert os.path.exists( os.path.join(args.input_dir, "train_label.pkl"))
    assert os.path.exists( os.path.join(args.input_dir, "test_label.pkl"))

    for split in ["train", "test"]:
        
        input_data_file = os.path.join(args.input_dir, f"{split}_data_joint.npy")
        output_data_file = os.path.join(args.output_dir, f"{split}_data_joint.npy")

        input_label_file = os.path.join(args.input_dir, f"{split}_label.pkl")
        output_label_file= os.path.join(args.output_dir, f"{split}_label.pkl")

        
        f_300_to_64(input_data_file, input_label_file, output_data_file, output_label_file)

    
    




if __name__ == "__main__":
    main()