
import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

# from utils.ntu_read_skeleton import read_xyz

import numpy as np
from tqdm import tqdm
import os

def save_pkl(fname, dt):
    with open(fname, "wb") as f:
        pickle.dump(dt, f)


def load_data(file_name):
    return np.load(file_name,mmap_mode="r")
    # return np.load(file_name)

def load_pkl(file_name):
    return pickle.load(open(file_name,"rb"))


def save_npy(save_name, data):
    np.save(save_name, data)




def get_mappings_51():
    # common labels 41 classes.
    # all starting from 0.
    # (0,50) of original pku -> (0,40) labels
    # (0,59) of original ntu -> (0,40) labels 

    pku_label_str='''1,bow
    2,brushing hair
    3,brushing teeth
    4,check time (from watch)
    5,cheer up
    6,clapping
    7,cross hands in front (say stop)
    8,drink water
    9,drop
    10,eat meal/snack
    11,falling
    12,giving something to other person
    13,hand waving
    14,handshaking
    15,hopping (one foot jumping)
    16,hugging other person
    17,jump up
    18,kicking other person
    19,kicking something
    20,make a phone call/answer phone
    21,pat on back of other person
    22,pickup
    23,playing with phone/tablet
    24,point finger at the other person
    25,pointing to something with finger
    26,punching/slapping other person
    27,pushing other person
    28,put on a hat/cap
    29,put something inside pocket
    30,reading
    31,rub two hands together
    32,salute
    33,sitting down
    34,standing up
    35,take off a hat/cap
    36,take off glasses
    37,take off jacket
    38,take out something from pocket
    39,taking a selfie
    40,tear up paper
    41,throw
    42,touch back (backache)
    43,touch chest (stomachache/heart pain)
    44,touch head (headache)
    45,touch neck (neckache)
    46,typing on a keyboard
    47,use a fan (with hand or paper)/feeling warm
    48,wear jacket
    49,wear on glasses
    50,wipe face
    51,writing'''


    pku_label_list=pku_label_str.split("\n")
    pku_label_list=[a.split(",") for a in pku_label_list]
    pku_label_list=[[a[0].strip(),a[1].strip()] for a in pku_label_list]
    pku_label_dt={int(a[0]):a[1] for a in pku_label_list}

    # remove_list=[12,14,16,18,21,24,26,27,29,38]
    original_pku_list=list(range(1,52))
    new_pku_list=[i for i in original_pku_list]


    # print(pku_label_dt)
    # print("-"*60)
    # print(original_pku_list)
    # print(new_pku_list)
    # print(len(new_pku_list))
    n_new=51

    # [0,...,40]
    common_list=list(range(n_new))

    pku2common_dt={new_pku_list[i]:common_list[i] for i in range(n_new)}
    # print("pku2common")
    # print(pku2common_dt)

    pku2common_0indexed={k-1:v for k,v in pku2common_dt.items()}

    common_name_list=[pku_label_dt[i] for i in new_pku_list]
    # print(common_name_list)  

    # pku2ntu_mapping='''bow,nod head/bow
    # brushing hair,brush hair
    # brushing teeth,brush teeth
    # check time (from watch),check time (from watch)
    # cheer up,cheer up 
    # clapping,clapping
    # cross hands in front (say stop),cross hands in front
    # drink water,drink water
    # drop,drop
    # eat meal/snack,eat meal
    # falling,falling down
    # giving something to other person,giving object
    # hand waving,hand waving 
    # handshaking,shaking hands
    # hopping (one foot jumping),hopping
    # hugging other person,hugging
    # jump up,jump up 
    # kicking other person,kicking
    # kicking something,kicking something
    # make a phone call/answer phone,phone call
    # pat on back of other person,pat on back
    # pickup ,pick up 
    # playing with phone/tablet,play with phone/tablet
    # point finger at the other person,point finger
    # pointing to something with finger,point to something
    # punching/slapping other person,punch/slap
    # pushing other person,pushing
    # put on a hat/cap,put on a hat/cap
    # put something inside pocket,touch pocket
    # reading,reading
    # rub two hands together,rub two hands
    # salute,salute
    # sitting down,sit down
    # standing up ,stand up
    # take off a hat/cap,take off a hat/cap
    # take off glasses,take off glasses
    # take off jacket,take off jacket
    # take out something from pocket,reach into pocket
    # taking a selfie,taking a selfie
    # tear up paper,tear up paper
    # throw,throw
    # touch back (backache),back pain
    # touch chest (stomachache/heart pain),chest pain
    # touch head (headache),headache
    # touch neck (neckache),neck pain
    # typing on a keyboard,type on a keyboard
    # use a fan (with hand or paper)/feeling warm,fan self
    # wear jacket,put on jacket
    # wear on glasses,put on glasses
    # wipe face,wipe face
    # writing,writing'''

    pku2ntu_mapping='''bow,nod head/bow
    brushing hair,brushing hair
    brushing teeth,brushing teeth
    check time (from watch),check time (from watch)
    cheer up,cheer up 
    clapping,clapping
    cross hands in front (say stop),cross hands in front (say stop)
    drink water,drink water
    drop,drop
    eat meal/snack,eat meal/snack
    falling,falling
    giving something to other person,giving something to other person
    hand waving,hand waving 
    handshaking,handshaking
    hopping (one foot jumping),hopping (one foot jumping)
    hugging other person,hugging other person
    jump up,jump up 
    kicking other person,kicking other person
    kicking something,kicking something
    make a phone call/answer phone,make a phone call/answer phone
    pat on back of other person,pat on back of other person
    pickup ,pickup 
    playing with phone/tablet,playing with phone/tablet
    point finger at the other person,point finger at the other person
    pointing to something with finger,pointing to something with finger
    punching/slapping other person,punching/slapping other person
    pushing other person,pushing other person
    put on a hat/cap,put on a hat/cap
    put something inside pocket,touch other person's pocket
    reading,reading
    rub two hands together,rub two hands together
    salute,salute
    sitting down,sitting down
    standing up ,standing up (from sitting position)
    take off a hat/cap,take off a hat/cap
    take off glasses,take off glasses
    take off jacket,take off jacket
    take out something from pocket,reach into pocket
    taking a selfie,taking a selfie
    tear up paper,tear up paper
    throw,throw
    touch back (backache),touch back (backache)
    touch chest (stomachache/heart pain),touch chest (stomachache/heart pain)
    touch head (headache),touch head (headache)
    touch neck (neckache),touch neck (neckache)
    typing on a keyboard,typing on a keyboard
    use a fan (with hand or paper)/feeling warm,use a fan (with hand or paper)/feeling warm
    wear jacket,wear jacket
    wear on glasses,wear on glasses
    wipe face,wipe face
    writing,writing'''
    

    pku2ntu_mapping=pku2ntu_mapping.split("\n")
    # print("*"*60)
    # print(pku2ntu_mapping)
    pku2ntu_mapping_list = [a.split(",") for a in pku2ntu_mapping]
    pku2ntu_mapping_list = [[a[0].strip(), a[1].strip()] for a in pku2ntu_mapping_list]
    pku2ntu_mapping_dt ={k:v for (k,v) in pku2ntu_mapping_list }
    pku_label_list_my=list(pku2ntu_mapping_dt.keys())
    pku_label_list_gt=[i[1] for i in pku_label_list]

    # print(pku_label_list_my)
    # print(pku_label_list_gt)

    assert pku_label_list_my==pku_label_list_gt

    # print(common_name_list)

    ntu_name_map_list=[pku2ntu_mapping_dt[i] for i in common_name_list]
    # print(ntu_name_map_list)

    ntu_actions={
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person's stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person's ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
    }
    ntu60_actions={k:v for k,v in ntu_actions.items() if k <= 60}

    ntu60_actions_inv={v:k for k,v in ntu60_actions.items()}


    ntu_name_map_idx_list=[ntu60_actions_inv[i]-1 for i in ntu_name_map_list]
    # print(ntu_name_map_idx_list)
    ntu2common_0indexed={ ntu_name_map_idx_list[i]:i for i in range(51) }

    # for i in ntu_name_map_list:
    #     print(i, ntu60_actions_inv[i])



    return pku2common_0indexed, ntu2common_0indexed, common_name_list


def to_data_label_with_mapping(data, label, mapping):

    sample_list, label_list = label 
    sample_list = np.array(sample_list)
    label_list  = np.array(label_list)

    selected_idx_list = []
    for src,dst in mapping.items():
        idx_list = np.where(label_list==src)[0]
        selected_idx_list.append(idx_list)
    selected_idx_list = np.concatenate(selected_idx_list,0)

    new_data = data[selected_idx_list]


    new_sample_list = sample_list[selected_idx_list]
    new_label_list  = label_list[selected_idx_list]
    new_label_list = np.array([mapping[k] for k in new_label_list])
    new_label = [new_sample_list.tolist(), new_label_list.tolist()]
    print(new_data.shape, len(new_label[0]))
    return new_data, new_label





def main_gen_pku_ntu51_pku(src_data_dir, dst_dir, domain):

    # given all data, process to common18
    pku2common_0indexed, ntu2common_0indexed, common_name_list = get_mappings_51()

    for split in ['train', 'test']:

        if split=="train":
            old_data_pku_file = f"{src_data_dir}/train_data.npy"
            old_label_pku_file =f"{src_data_dir}/train_label.pkl"
        elif split=="test":
            old_data_pku_file = f"{src_data_dir}/test_data.npy"
            old_label_pku_file =f"{src_data_dir}/test_label.pkl"
        else:
            raise ValueError()

        

        old_data_pku = load_data(old_data_pku_file)
        old_label_pku = load_pkl(old_label_pku_file)

        # not used here.
        # new_data_pku, new_label_pku   = to_data_label_with_mapping(old_data_pku, old_label_pku, pku2common_0indexed)

        new_data_pku, new_label_pku = old_data_pku, old_label_pku

        new_data_pku_file =   f"{dst_dir}/{domain}/{split}_data_joint.npy"
        new_label_pku_file =  f"{dst_dir}/{domain}/{split}_label.pkl"

        os.makedirs(f"{dst_dir}/{domain}", exist_ok=True)

        # downsample to 64 
        from downsample_motion import downsample_with_seqlen
        new_data_pku = downsample_with_seqlen(new_data_pku, 64)

        print("-> save to", new_data_pku_file, new_data_pku.shape)
        print("-> save to", new_label_pku_file, len(new_label_pku[1]) )

        save_npy(new_data_pku_file, new_data_pku)
        save_pkl(new_label_pku_file, new_label_pku)



def get_selected_index(label_src, idx_list):
    a,b = label_src 
    a = [a[i] for i in idx_list]
    b = [b[i] for i in idx_list]
    return [a,b]


def reorder_sample(data, labels, sorted_to_gt_mapping):

    def get_index_mapping(label_src, label_dst):
        assert sorted(label_src)==sorted(label_dst)
        return [label_src.index(i) for i in label_dst]

    def get_selected_index(label_src, idx_list):
        a,b = label_src 
        a = [a[i] for i in idx_list]
        b = [b[i] for i in idx_list]
        return [a,b]


    # index1: my_label -> sorted(my_label)
    index1 = get_index_mapping( labels[0], list(sorted(labels[0])) )
    labels = get_selected_index(labels, index1)
    data   = data[index1]
    assert list(sorted(labels[0])) == labels[0]


    # index2: sorted(my_label) -> etri220501_full_label
    # index2 = get_index_mapping( label_old[0], label_new[0] )
    labels_reordered = get_selected_index(labels, sorted_to_gt_mapping)
    data_reordered   = data[sorted_to_gt_mapping]

    return data_reordered, labels_reordered


def main_gen_pku_ntu51_ntu(src_data_dir, dst_dir, domain):

    # given all data, process to common18
    pku2common_0indexed, ntu2common_0indexed, common_name_list = get_mappings_51()

    for split in ['train', 'test']:

        if split=="train":
            old_data_pku_file = f"{src_data_dir}/train_data_joint.npy"
            old_label_pku_file =f"{src_data_dir}/train_label.pkl"
        elif split=="test":
            old_data_pku_file = f"{src_data_dir}/val_data_joint.npy"
            old_label_pku_file =f"{src_data_dir}/val_label.pkl"
        else:
            raise ValueError()

        
        old_data_pku = load_data(old_data_pku_file)
        old_label_pku = load_pkl(old_label_pku_file)

        # not used here.
        # new_data_pku, new_label_pku   = to_data_label_with_mapping(old_data_pku, old_label_pku, ntu2common_0indexed)

        # to_data_label_with_mapping, keep original order of each sample.
        selected_idx_list = []

        for ii, each_label in enumerate(old_label_pku[1]):
            if each_label in ntu2common_0indexed.keys():
                selected_idx_list.append(ii)
        
        new_data_pku  = old_data_pku[selected_idx_list]
        new_label_pku = get_selected_index(old_label_pku, selected_idx_list)
        # update label to [0,50]
        new_label_pku[1] = [ntu2common_0indexed[a] for a in new_label_pku[1]]


        new_data_pku_file =   f"{dst_dir}/{domain}/{split}_data_joint.npy"
        new_label_pku_file =  f"{dst_dir}/{domain}/{split}_label.pkl"

        os.makedirs(f"{dst_dir}/{domain}", exist_ok=True)

        # downsample to 64 
        from downsample_motion import downsample_with_seqlen
        new_data_pku = downsample_with_seqlen(new_data_pku, 64)

        # reorder index 
        reorder_index_dt = np.load("./ntu_ntupku51_sample_order_index.npy", allow_pickle=True).item()
        reorder_index = reorder_index_dt[f"{split}"]
        new_data_pku, new_label_pku = reorder_sample(new_data_pku, new_label_pku, reorder_index)


        print("-> save to", new_data_pku_file, new_data_pku.shape)
        print("-> save to", new_label_pku_file, len(new_label_pku[1]) )

        save_npy(new_data_pku_file, new_data_pku)
        save_pkl(new_label_pku_file, new_label_pku)



def merge_train_test(dst_dir, domain):
    train_data_file = os.path.join(dst_dir, f"{domain}/train_data_joint.npy")
    test_data_file  = os.path.join(dst_dir, f"{domain}/test_data_joint.npy")

    train_data = np.load(train_data_file)
    test_data  = np.load(test_data_file)

    all_data = np.concatenate([train_data, test_data], 0)

    save_data_file = os.path.join(dst_dir, f"{domain}/all_data_joint.npy")
    print("-> save to ", save_data_file, all_data.shape)
    np.save(save_data_file, all_data)


    train_label_file = os.path.join(dst_dir, f"{domain}/train_label.pkl")
    test_label_file  = os.path.join(dst_dir, f"{domain}/test_label.pkl")

    train_label = np.load(train_label_file, allow_pickle=True)
    test_label  = np.load(test_label_file,  allow_pickle=True)
    all_label = [ train_label[0]+test_label[0], train_label[1]+test_label[1] ]

    save_label_file = os.path.join(dst_dir, f"{domain}/all_label.pkl")
    print("-> save to ", save_label_file, len(all_label[0]))
    save_pkl(save_label_file, all_label)


def main():
    pku2common_0indexed, ntu2common_0indexed, common_name_list = get_mappings_51()

    print(pku2common_0indexed)
    print(ntu2common_0indexed)  


    dst_dir = '../data/processed_data/ntupku51/xsub64'
    os.makedirs(dst_dir, exist_ok=True)

    main_gen_pku_ntu51_ntu(src_data_dir="../data/processed_data/ntu/xsub", dst_dir=dst_dir, domain="ntu")
    merge_train_test(dst_dir, domain="ntu")




    main_gen_pku_ntu51_pku(src_data_dir="../data/processed_data/pku", dst_dir=dst_dir, domain="pku")
    merge_train_test(dst_dir, domain="pku")
    




    
    




if __name__ == "__main__":
    main()
