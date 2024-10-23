import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
# from data_gen.preprocess import pre_normalization
from prenorm import pre_normalization

from IPython import embed 
import shutil

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

import numpy as np
import os

import pandas as pd 

import glob 






'''
etri220429
- all_ages
    - train_data, train_label, test_data, test_label
- adult
    - train_data, train_label, test_data, test_label
- elder
    - train_data, train_label, test_data, test_label
'''

all_ages_test_subjects = list(range(3,99+3,3))
all_ages_train_subjects = [i for i in range(1,100+1) if i not in all_ages_test_subjects]

elder_test_subjects = list(range(3,50,3))
elder_train_subjects = [i for i in range(1,51) if i not in elder_test_subjects]

adult_test_subjects = list(range(51,100,3))
adult_train_subjects = [i for i in range(51,101) if i not in adult_test_subjects]

debug_test_subjects = [1]

etri_map = {
    'all_ages': {
        'train': all_ages_train_subjects,
        'test':  all_ages_test_subjects
    },
    'elder': {
        'train': elder_train_subjects,
        'test':  elder_test_subjects
    },
    'adult': {
        'train': adult_train_subjects,
        'test':  adult_test_subjects
    },
    'debug': {
        'test': debug_test_subjects
    }
}

for k,v in etri_map.items():
    print(k,v)



ignored_sample_path = ["A039_P042_G001_C007.csv", "A039_P042_G001_C008.csv"]





def etri_read_skeleton_filter(file):
    '''
    {
        'numFrame': int
        'frameInfo' list of numFrame: []
            {'numBody': int, 'bodyInfo': list }

        bodyInfo = {'jointInfo'}
        jointInfo = {'x', 'y', 'z'}
    }

    how to deal with NaN.

    '''

    x=pd.read_csv(file)
    x = x.dropna(subset=['bodyindexID'],how='any')
    # print(x)
    # return 

    body_index_list = np.unique(x['bodyindexID'].tolist())
    frame_index_list = np.sort(np.unique(x['frameNum'].tolist()))

    
    
    skeleton_sequence = {}
    skeleton_sequence['numFrame'] = len(frame_index_list)
    skeleton_sequence['frameInfo'] = []
    # num_body = 0


    for t in frame_index_list:
        frame_info = {}
        xt = x[x['frameNum']==t]
        frame_info['numBody'] = len(np.unique(xt['bodyindexID'].values))
        frame_info['bodyInfo'] = []

        for m in range(frame_info['numBody']):
            body_info = {}

            y = xt[xt['bodyindexID']==body_index_list[m]]
            if y.shape[0]==0:
                continue
            
            body_info['numJoint'] = 25
            body_info['jointInfo'] = []
            for v in range(body_info['numJoint']):
                joint_info_key = [
                    'x', 'y', 'z'
                ]
                # joint_info = {
                #     k: float(v)
                #     for k, v in zip(joint_info_key, f.readline().split())
                # }
                joint_info = {}
                for k in joint_info_key:
                    tag='joint{}_3d{}'.format( v+1, k.upper() )
                    
                    joint_info[k] = float(y[tag])
                    
                body_info['jointInfo'].append(joint_info)
            frame_info['bodyInfo'].append(body_info)
        skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def etri_read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body

    # print(file)

    seq_info = etri_read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)

    # print(data.shape)
    # sys.exit(0)

    return data


def etri_gendata(data_path, out_path, benchmark='xview', part='eval'):
    
    sample_name = []
    sample_label = []
    
    data_path_1 = os.path.join(data_path, "P001-P050")
    data_path_2 = os.path.join(data_path, "P051-P100")
    data_path_list = os.listdir(data_path_1) + os.listdir(data_path_2)
    # print("len of all data path = ", len(data_path))

    folder_map={}
    for filename in data_path_list:
        
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        issample = (subject_id in etri_map[benchmark][part])

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
            
            if subject_id<=50:
                folder_map[filename]="P001-P050"
            else:
                folder_map[filename]="P051-P100"

    # print(filename)
    # embed()
    # sys.exit(0)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    
    print('{}/{}_label.pkl'.format(out_path, part), "len = ", len(sample_name))

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        split_folder = folder_map[s]
        data = etri_read_xyz(os.path.join(data_path, split_folder, s), max_body=max_body_kinect, num_joint=num_joint)
        # fp[i, :, 0:data.shape[1], :, :] = data

        t_max = min(data.shape[1], max_frame)
        fp[i, :, 0:t_max, :, :] = data[:,:t_max,:,:]

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

    print("saved to",'{}/{}_data_joint.npy'.format(out_path, part) )




def ff(args):
    input_name, max_body_kinect, num_joint, save_each_name = args
    data = etri_read_xyz(input_name, max_body=max_body_kinect, num_joint=num_joint)
    np.save(save_each_name, data)
    return None



def etri_gendata_multiprocessing(data_path, out_path, ignored_sample_path, 
    out_dir="tmp", benchmark='xview', part='eval'):
    '''
    save to os.path.join(out_path, out_dir)
    '''
    
    sample_name = []
    sample_label = []
    
    data_path_1 = os.path.join(data_path, "P001-P050")
    data_path_2 = os.path.join(data_path, "P051-P100")
    data_path_list = os.listdir(data_path_1) + os.listdir(data_path_2)
    # print("len of all data path = ", len(data_path))

    folder_map={}
    for filename in data_path_list:

        if filename in ignored_sample_path:
            continue
        
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        issample = (subject_id in etri_map[benchmark][part])

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
            
            if subject_id<=50:
                folder_map[filename]="P001-P050"
            else:
                folder_map[filename]="P051-P100"

    # print(filename)
    # embed()
    # sys.exit(0)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    
    print('{}/{}_label.pkl'.format(out_path, part), "len = ", len(sample_name))

    


    out_tmp_dir = os.path.join(out_path, out_dir)
    if not os.path.exists(out_tmp_dir):
        os.mkdir(out_tmp_dir)

    print("save to", out_tmp_dir)
    # for i, s in enumerate(tqdm(sample_name)):
    #     split_folder = folder_map[s]
    #     data = etri_read_xyz(os.path.join(data_path, split_folder, s), max_body=max_body_kinect, num_joint=num_joint)
    #     # fp[i, :, 0:data.shape[1], :, :] = data

    #     # t_max = min(data.shape[1], max_frame)
    #     # fp[i, :, 0:t_max, :, :] = data[:,:t_max,:,:]
    #     save_each_name = os.path.join(out_tmp_dir, f"{s}.npy")
    #     np.save(save_each_name, data)


    iterators = []
    for i, s in enumerate(tqdm(sample_name)):
        split_folder = folder_map[s]
        
        input_name = os.path.join(data_path, split_folder, s)
        # data = etri_read_xyz(os.path.join(data_path, split_folder, s), max_body=max_body_kinect, num_joint=num_joint)
        # fp[i, :, 0:data.shape[1], :, :] = data

        # t_max = min(data.shape[1], max_frame)
        # fp[i, :, 0:t_max, :, :] = data[:,:t_max,:,:]
        save_each_name = os.path.join(out_tmp_dir, f"{s}.npy")
        # np.save(save_each_name, data)

        iterators.append(
            [input_name, max_body_kinect, num_joint, save_each_name]
        )
    
    

    import multiprocessing
    with multiprocessing.Pool() as pool:

        with tqdm(total=len(iterators)) as pbar:
            for _ in pool.imap_unordered(ff, iterators):
                pbar.update()
        


    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    for i, s in enumerate(tqdm(sample_name)):
        tmp_file_name = os.path.join(out_tmp_dir,f"{s}.npy")
        data = np.load(tmp_file_name)

        t_max = min(data.shape[1], max_frame)
        fp[i, :, 0:t_max, :, :] = data[:,:t_max,:,:]

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

    print("saved to",'{}/{}_data_joint.npy'.format(out_path, part) )

    shutil.rmtree(out_tmp_dir)



def check_invalid_files():

    file_list = glob.glob("/home/yqs/liuhc/action_data/etri_data/P001-P050/*.csv") + \
                glob.glob("/home/yqs/liuhc/action_data/etri_data/P051-P100/*.csv")
    for i, file in enumerate(file_list):
        x = pd.read_csv(file)
        try:
            assert 'frameNum' in x.columns.tolist()
        except:
            print("not valid = ", file)
        if i%1000==0:
            print(i, len(file_list))
    sys.exit(0)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/ntu/')
    arg = parser.parse_args()

    # arg.data_path = "/home/yqs/liuhc/action_data/etri_data"
    # arg.out_folder = "/home/yqs/liuhc/action/data/etri220429"
    # arg.data_path = "/mnt/action/data2/etri"
    # arg.out_folder = "/mnt/tmp240422/etri"

    arg.data_path = "../data/raw_data/etri_data"
    arg.out_folder = "../data/processed_data/etri"



    # benchmark = ['adult', 'elder', 'all_ages']
    # part = ['train', 'test']

    benchmark = ['adult','elder']
    # part = ['test']
    part = ['test','train']

    # benchmark = ['debug']
    # part = ['test']
    

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            print("to path = ", out_path)
            # etri_gendata(
            #     arg.data_path,
            #     out_path,
            #     benchmark=b,
            #     part=p)

            etri_gendata_multiprocessing(
                arg.data_path,
                out_path, ignored_sample_path, out_dir="tmp",
                benchmark=b,
                part=p
            )