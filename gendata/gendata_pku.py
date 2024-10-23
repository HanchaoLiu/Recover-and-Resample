import os,sys
import numpy as np 
from IPython import embed
import glob  
from prenorm import pre_normalization

import pickle

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




def generate_pku_dataset(base_dir):
    '''
    return:
        data_batch: (n,c,t,v,m) t=300
        labels: [sample_name, label_list]
    '''
    # assert interp in ["nearest", "linear"]
    # base_dir="/mnt/data1/liuhc/action/data/pkummd"
    # base_dir="/home/cscg/data/pkummd"

    label_dir=os.path.join(base_dir, "Label/Train_Label_PKU_final")
    data_dir =os.path.join(base_dir, "Data/PKU_Skeleton_Renew")

    print(label_dir,data_dir)

    data_file_list  = sorted(glob.glob(os.path.join(data_dir,"*.txt")))
    label_file_list = sorted(glob.glob(os.path.join(label_dir,"*.txt")))

    tag_list = [os.path.basename(i) for i in data_file_list]

    # tag_list = tag_list[:100]

    print(len(data_file_list), len(label_file_list))
    # assert data_file_list == label_file_list

    n_action_list=[]
    action_label_list=[]
    
    action_len_list=[]

    data_all_list=[]
    sample_name_all_list=[]
    sample_label_all_list=[]

    # [(data,sample_name,label)]
    data_raw_list = []
    length_list = []

    for i,tag in enumerate(tag_list):
        data_file=os.path.join(data_dir,tag)
        label_file=os.path.join(label_dir,tag)
        assert os.path.exists(data_file) and os.path.exists(label_file)

        label_video=np.loadtxt(label_file,delimiter=',').astype(int)
        
        data_video=np.loadtxt(data_file)

        # (t,m,v,c)
        data_video=data_video.reshape((data_video.shape[0],2,25,3))
        # (t,m,v,c) -> (c,t,v,m)
        data_video=np.transpose(data_video,[3,0,2,1])
        # data_video=data_video[:,:25,:]
        # assert np.all(data_video[:,25:,:]==0.0)
        # print()

        # print(tag, label_video.shape, data_video.shape)
        if (i%100==0):
            print(i,len(tag_list),"pku data")

        label_video=label_video[:,:3]
        for action_i, (action_label, st, ed) in enumerate(label_video):
            
            
            # (c,t,v,m)
            data_cut=data_video[:,st:ed+1,:,:]
            try:
                assert not np.all(data_cut==0)
            except:
                print(tag, action_i, "all zero !")
                continue

            x = data_cut

            sample_name = "{}_{}".format(tag.replace(".txt",""),action_i)
            item = [x, action_label-1, sample_name]

            data_raw_list.append(item)
            length_list.append(ed-st)


    # total n = ~21000  n(>300)=407, max=700
    # n(>200)=2277 (around 10%)
    print(len(data_raw_list))


    # len_list = [i[0].shape[1] for i in data_raw_list]
    # from collections import Counter
    # c = Counter(len_list)
    # print(c)
    # if False:
    #     c=dict(c)
    #     k_list = sorted(list(c.keys()))
    #     for k in k_list:
    #         print(k, ":", c[k])

    # save_path="/home/cscg/data/pkummd/pkummd_split_samples.pkl"
    # print("save to", save_path)
    # with open(save_path,"wb") as f:
    #     pickle.dump(data_raw_list, f)
    nn=len(data_raw_list)
    data_all = np.zeros((nn,3,300,25,2),np.float32)
    sample_name_all = []
    label_list_all  = []

    # label_all original range [0,50], but only 41 valid.
    # so map label from [0,50] -> [0,40]
    # y_mapping = {
    #     0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 
    #     12: 11, 14: 12, 16: 13, 18: 14, 19: 15, 21: 16, 22: 17, 24: 18, 27: 19, 
    #     29: 20, 30: 21, 31: 22, 32: 23, 33: 24, 34: 25, 35: 26, 36: 27, 38: 28, 
    #     39: 29, 40: 30, 41: 31, 42: 32, 43: 33, 44: 34, 45: 35, 46: 36, 47: 37, 
    #     48: 38, 49: 39, 50: 40}
    y_mapping = {kk:kk for kk in range(51)}

    for i in range(nn):
        x, action_label, sample_name = data_raw_list[i]
        # print(x.shape)
        action_label_new = y_mapping[action_label]

        max_len=300
        this_len=x.shape[1]
        tt = min(max_len,this_len)
        data_all[i,:,:tt,:,:]=x[:,:tt,:,:] 
        sample_name_all.append(sample_name)
        label_list_all.append(action_label_new)

    label_all = [sample_name_all, label_list_all]

    # using pre norm!!!
    # print("pre_normalization!!")
    data_all = pre_normalization(data_all)

    return data_all, label_all

    
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

    
def to_split(split_file, src_dir, dst_dir):
    # split_file = "/home/cscg/data/pkummd/PKUMMDv2/Split/cross_subject_v2.txt"
    with open(split_file,"r") as f:
        split_data = f.readlines()
        split_data = [i.strip() for i in split_data]
    # for line in split_data:
    #     print(line)

    train_split = split_data[1].split(",")
    train_split = [i.strip() for i in train_split]
    test_split  = split_data[3].split(",")
    test_split  = [i.strip() for i in test_split]

    # src_dir="/home/cscg/data/pkummd/xsub64"
    data_all_name = os.path.join(src_dir, "all_data.npy")
    label_all_name = os.path.join(src_dir, "all_label.pkl")
    data_all = np.load(data_all_name)
    label_all= np.load(label_all_name,allow_pickle=True)
    sample_name_all = np.load(label_all_name,allow_pickle=True)[0]

    

    train_idx_list=[]
    test_idx_list =[]
    for i,sample_name in enumerate(sample_name_all):
        sample_name = sample_name.split("_")[0]
        # print(sample_name)
        if sample_name in train_split:
            train_idx_list.append(i)
        elif sample_name in test_split:
            test_idx_list.append(i)
        else:
            raise ValueError()

    
    train_data = data_all[train_idx_list]
    test_data  = data_all[test_idx_list]
    train_label= get_label_by_index(label_all, train_idx_list)
    test_label = get_label_by_index(label_all, test_idx_list)

    # base_dir="/home/cscg/data/pkummd/xsub64"

    base_dir = dst_dir
    train_data_name = os.path.join(base_dir, "train_data.npy")
    train_label_name  = os.path.join(base_dir, "train_label.pkl")
    test_data_name = os.path.join(base_dir, "test_data.npy")
    test_label_name  = os.path.join(base_dir, "test_label.pkl")

    np.save(train_data_name, train_data)
    save_pkl(train_label_name, train_label)

    np.save(test_data_name, test_data)
    save_pkl(test_label_name, test_label)

    print(train_data_name)
    print(train_label_name)
    print(test_data_name)
    print(test_label_name)

    print("train=",train_data.shape[0] , "test=",test_data.shape[0])



def get_label_by_index(label_list,idx_list):
    a,b=label_list
    new_a = [a[i] for i in idx_list]
    new_b = [b[i] for i in idx_list]
    return [new_a,new_b]



# def main():

#     pku2common_0indexed, ntu2common_0indexed, common_name_list = get_mappings_51()
#     embed()


def generate_pku_dataset_seqlen(base_dir):
    '''
    return:
        data_batch: (n,c,t,v,m) t=300
        labels: [sample_name, label_list]
    '''
    # assert interp in ["nearest", "linear"]
    # base_dir="/mnt/data1/liuhc/action/data/pkummd"
    # base_dir="/home/cscg/data/pkummd"

    label_dir=os.path.join(base_dir, "Label/Train_Label_PKU_final")
    data_dir =os.path.join(base_dir, "Data/PKU_Skeleton_Renew")

    print(label_dir,data_dir)

    data_file_list  = sorted(glob.glob(os.path.join(data_dir,"*.txt")))
    label_file_list = sorted(glob.glob(os.path.join(label_dir,"*.txt")))

    tag_list = [os.path.basename(i) for i in data_file_list]

    # tag_list = tag_list[:100]

    print(len(data_file_list), len(label_file_list))
    # assert data_file_list == label_file_list

    n_action_list=[]
    action_label_list=[]
    
    action_len_list=[]

    data_all_list=[]
    sample_name_all_list=[]
    sample_label_all_list=[]

    # [(data,sample_name,label)]
    data_raw_list = []
    length_list = []

    for i,tag in enumerate(tag_list):
        data_file=os.path.join(data_dir,tag)
        label_file=os.path.join(label_dir,tag)
        assert os.path.exists(data_file) and os.path.exists(label_file)

        label_video=np.loadtxt(label_file,delimiter=',').astype(int)
        
        # data_video=np.loadtxt(data_file)

        # # (t,m,v,c)
        # data_video=data_video.reshape((data_video.shape[0],2,25,3))
        # # (t,m,v,c) -> (c,t,v,m)
        # data_video=np.transpose(data_video,[3,0,2,1])
        # data_video=data_video[:,:25,:]
        # assert np.all(data_video[:,25:,:]==0.0)
        # print()

        # print(tag, label_video.shape, data_video.shape)
        if (i%100==0):
            print(i,len(tag_list),"pku data")

        label_video=label_video[:,:3]
        for action_i, (action_label, st, ed) in enumerate(label_video):
            
            
            # (c,t,v,m)
            # data_cut=data_video[:,st:ed+1,:,:]
            # try:
            #     assert not np.all(data_cut==0)
            # except:
            #     print(tag, action_i, "all zero !")
            #     continue

            # x = data_cut

            sample_name = "{}_{}".format(tag.replace(".txt",""),action_i)
            item = [ed-st, action_label-1, sample_name]

            data_raw_list.append(item)
            length_list.append(ed-st)


    # total n = ~21000  n(>300)=407, max=700
    # n(>200)=2277 (around 10%)
    print(len(data_raw_list))


    # len_list = [i[0].shape[1] for i in data_raw_list]
    # from collections import Counter
    # c = Counter(len_list)
    # print(c)
    # if False:
    #     c=dict(c)
    #     k_list = sorted(list(c.keys()))
    #     for k in k_list:
    #         print(k, ":", c[k])

    # save_path="/home/cscg/data/pkummd/pkummd_split_samples.pkl"
    # print("save to", save_path)
    # with open(save_path,"wb") as f:
    #     pickle.dump(data_raw_list, f)
    # nn=len(data_raw_list)
    # data_all = np.zeros((nn,3,300,25,2),np.float32)
    # sample_name_all = []
    # label_list_all  = []

    return data_raw_list, length_list




def main_seqlen():
    base_dir = "/mnt/action/data2/pku_v1"

    if True:
        data_raw_list, length_list = generate_pku_dataset_seqlen(base_dir)
        dt = {}
        for (seqlen, label, sample_name) in data_raw_list:
            dt[sample_name]=seqlen

        save_pkl('/mnt/action/data2/pku_v1/xsub_raw/seqlen.pkl', dt)




def main():
    # base_dir = "/mnt/action/data2/pku_v1"
    base_dir = "../data/raw_data/pkummd_v1"
    

    
    # dst_dir = "/mnt/action/data2/pku_v1/xsub_raw"
    # dst_dir = "/mnt/tmp240422/pku"
    dst_dir = "../data/processed_data/pku"
    os.makedirs(dst_dir, exist_ok=True)

    if True:
        data_all, label_all = generate_pku_dataset(base_dir)

        # to len64
        # from adaptive_sampling import downsample_with_seqlen
        # data_all_64 = downsample_with_seqlen(data_all, 64)
        # print(data_all_64.shape, len(label_all[0]))

        np.save(f"{dst_dir}/all_data.npy", data_all)
        save_pkl(f"{dst_dir}/all_label.pkl",label_all)

    # sys.exit(0)

    # split to train, and test 
    split_file = f"{base_dir}/Split/cross-subject.txt"
    assert os.path.exists(split_file)
    src_dir = dst_dir
    to_split(split_file, src_dir, dst_dir)







if __name__ == "__main__":
    
    # main_seqlen()

    main()