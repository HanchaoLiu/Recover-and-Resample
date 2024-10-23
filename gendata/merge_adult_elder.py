import os,sys
import numpy as np 
import pickle


def load_pkl(file_name):
    return pickle.load(open(file_name,"rb"))

def save_pkl(fname, dt):
    with open(fname, "wb") as f:
        pickle.dump(dt, f)


def merge_data(data_a, label_a, data_e, label_e):

    data_all = []
    label_all_sample = []
    label_all_label  = []

    label_a_sample, label_a_label = label_a 
    label_e_sample, label_e_label = label_e

    label_a_sample, label_a_label = np.array(label_a_sample), np.array(label_a_label)
    label_e_sample, label_e_label = np.array(label_e_sample), np.array(label_e_label)

    n_class=18 
    for c in range(n_class):
        print(c)
        idx_a = np.where(label_a_label==c)[0]
        label_all_sample.append( label_a_sample[idx_a] )
        label_all_label.append( label_a_label[idx_a] )

        idx_e = np.where(label_e_label==c)[0]
        label_all_sample.append( label_e_sample[idx_e] )
        label_all_label.append( label_e_label[idx_e] )

        data_all.append(data_a[idx_a])
        data_all.append(data_e[idx_e])
        


    label_all_sample = np.concatenate(label_all_sample, 0).tolist()
    label_all_label  = np.concatenate(label_all_label, 0).tolist()

    label_all = [label_all_sample, label_all_label]

    data_all = np.concatenate(data_all, 0)

    print(data_all.shape, len(label_all[0]), len(label_all[1]) )

    return data_all, label_all


def main():
    
   

    base_dir = "../data/processed_data/common18/xsub64"
    split_list = ["test"]

    for split in split_list:
        
        data_file_a = f"{base_dir}/etriA/{split}_data_joint.npy"
        data_file_e = f"{base_dir}/etriE/{split}_data_joint.npy"

        label_file_a = f"{base_dir}/etriA/{split}_label.pkl"
        label_file_e = f"{base_dir}/etriE/{split}_label.pkl"

        assert os.path.exists(data_file_a)
        assert os.path.exists(data_file_e)

        assert os.path.exists(label_file_a)
        assert os.path.exists(label_file_e)

        data_a = np.load(data_file_a)
        data_e = np.load(data_file_e)
        label_a = load_pkl(label_file_a)
        label_e = load_pkl(label_file_e)

        data_all, label_all = merge_data(data_a, label_a, data_e, label_e)

        os.makedirs(f"{base_dir}/etri", exist_ok=True)
        save_data_name  = f"{base_dir}/etri/{split}_data_joint.npy"
        save_label_name = f"{base_dir}/etri/{split}_label.pkl"

        np.save(save_data_name, data_all)
        save_pkl(save_label_name, label_all)




if __name__ == "__main__":
    main()