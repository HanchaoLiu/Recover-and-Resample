import os,sys 
import numpy as np 



def get_idx(sample_list_new, sample_list_old):
    res = []
    for name in sample_list_old:
        res.append(sample_list_new.index(name))
    return res 

def apply_mapping_label(labels, idx_mapping):
    a, b = labels 
    a1 = [a[i] for i in idx_mapping]
    b1 = [b[i] for i in idx_mapping]
    return [a1,b1]


def apply_mapping_data(data, idx_mapping):
    return data[idx_mapping]


def main():
    new_dir = "data/common18/xsub64"
    old_dir = "data/common18_220423/xsub64"

    # new_dir = "data/ntupku51/xsub64"
    # old_dir = "data/ntupku51_220423/xsub64"

    for domain in ["ntu", "pku", "etriA", "etriE", "etri"]:
        for split in ["test", "train"]:
            
            if domain=="etri" and split=="train":
                continue

            new_name_data = os.path.join(new_dir, domain, split+"_data_joint.npy")
            old_name_data = os.path.join(old_dir, domain, split+"_data_joint.npy")

            # if not (os.path.exists(new_name_data) and os.path.exists(old_name_data)):
            #     continue

            new_name_label = os.path.join(new_dir, domain, split+"_label.pkl")
            old_name_label = os.path.join(old_dir, domain, split+"_label.pkl")
            label_new = np.load(new_name_label, allow_pickle=True)
            label_old = np.load(old_name_label, allow_pickle=True)

            new_data = np.load(new_name_data, allow_pickle=True)
            old_data = np.load(old_name_data, allow_pickle=True)

            print(new_name_data, old_name_data)    
            print(new_data.shape, old_data.shape)

            # idx_mapping = get_idx(label_new[0], label_old[0])
            # label_new = apply_mapping_label(label_new, idx_mapping) 
            # new_data =  apply_mapping_data(new_data, idx_mapping)
            # print(label_new[0][:5], label_old[0][:5])
            # assert list(sorted(label_new[0]))==list(sorted(label_old[0]))


            assert label_new[0]==label_old[0]
            assert label_new[1]==label_old[1]

            diff = np.abs(new_data - old_data).max()
            print(diff)

            
            


    # for domain in ["ntu", "pku", "etriA", "etriE", "etri"]:
    #     for split in ["train", "test"]:
    #         new_name = os.path.join(new_dir, domain, split+"_data_joint.npy")
    #         old_name = os.path.join(old_dir, domain, split+"_data_joint.npy")
    #         if os.path.exists(new_name) and os.path.exists(old_name):
    #             print(new_name, old_name)
    #             new_data = np.load(new_name, allow_pickle=True)
    #             old_data = np.load(old_name, allow_pickle=True)
    #             diff = np.abs(new_data - old_data).max()
    #             print(diff)
                





main()


def main_get_ntu_common18_order_index():

    # new_dir = "data/common18/xsub64"
    old_dir = "data/common18_220423/xsub64"

    # new_dir = "data/ntupku51/xsub64"
    # old_dir = "data/ntupku51_220423/xsub64"

    dt = {}

    for domain in ["ntu"]:
        for split in ["test", "train"]:

            old_name_label = os.path.join(old_dir, domain, split+"_label.pkl")
            label_old = np.load(old_name_label, allow_pickle=True)

            sample_list_old = label_old[0]
            sample_list_ordered = list(sorted(label_old[0]))
            order_idx = [sample_list_ordered.index(name) for name in sample_list_old]
            dt[split] = order_idx 
    
    # print(dt)
    np.save("./gendata/ntu_common18_sample_order_index.npy", dt)


# main_get_ntu_common18_order_index()


def main_get_ntu_ntupku51_order_index():

    old_dir = "data/ntupku51_220423/xsub64"

    dt = {}

    for domain in ["ntu"]:
        for split in ["test", "train"]:

            old_name_label = os.path.join(old_dir, domain, split+"_label.pkl")
            label_old = np.load(old_name_label, allow_pickle=True)

            sample_list_old = label_old[0]
            sample_list_ordered = list(sorted(label_old[0]))
            order_idx = [sample_list_ordered.index(name) for name in sample_list_old]
            dt[split] = order_idx 
    
    # print(dt)
    np.save("./gendata/ntu_ntupku51_sample_order_index.npy", dt)

# main_get_ntu_ntupku51_order_index()

