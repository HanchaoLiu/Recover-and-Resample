import os, sys
import numpy as np 
import shutil
import argparse
import pickle
from tqdm import tqdm

from IPython import embed 



def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--domain', default='none', help='the work folder for storing results')

    return parser





#######################################################################
#  common18 labels and mappings
#######################################################################

def get_common18_label():
    string = '''
    1. eat
    2. drink
    3. brush teeth
    5. brush hair
    6. wear clothes
    7. take off clothes
    9. put on/take off glasses
    10. read
    11. write
    12. phone call
    13. play with phone
    15. clap
    17. bow
    18. handshake
    19. hug
    21. hand wave
    22. point finger
    23. fall down
    '''
    string = string.split("\n")
    s = [i.strip() for i in string]
    s = [i for i in s if len(i)!=0]
    s = [i.split('.')[1].strip() for i in s]
    assert len(s)==18
    return s 



def index1_to_index0(etri_idx_list_1indexed):
    etri_idx_list_0indexed=[]
    for i in etri_idx_list_1indexed:
        if isinstance(i, list):
            etri_idx_list_0indexed.append( (np.array(i)-1).tolist() )
        else:
            etri_idx_list_0indexed.append( i-1 )
    return etri_idx_list_0indexed


def to_dict(src_list, dst_list):

    dt= {}
    assert len(src_list)==len(dst_list)
    for i in range(len(src_list)):
        src = src_list[i]
        dst = dst_list[i]
        if isinstance(src, list):
            for k in src:
                dt[k]=dst
        else:
            dt[src]=dst 
    return dt 




def get_mapping_ntu_etri_common23():

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


    etri_actions = {
        1: "eating food with a fork",
        2: "pouring water into a cup",
        3: "taking medicine",
        4: "drinking water",
        5: "putting food in the fridge/taking food from the fridge",
        6: "trimming vegetables",
        7: "peeling fruit",
        8: "using a gas stove",
        9: "cutting vegetable on the cutting board",
        10: "brushing teeth",
        11: "washing hands",
        12: "washing face",
        13: "wiping face with a towel",
        14: "putting on cosmetics",
        15: "putting on lipstick",
        16: "brushing hair",
        17: "blow drying hair",
        18: "putting on a jacket",
        19: "taking off a jacket",
        20: "putting on/taking off shoes",
        21: "putting on/taking off glasses",
        22: "washing the dishes",
        23: "vacuuming the floor",
        24: "scrubbing the floor with a rag",
        25: "wiping off the dinning table",
        26: "rubbing up furniture",
        27: "spreading bedding/folding bedding",
        28: "washing a towel by hands",
        29: "hanging out laundry",
        30: "looking around for something",
        31: "using a remote control",
        32: "reading a book",
        33: "reading a newspaper",
        34: "handwriting",
        35: "talking on the phone",
        36: "playing with a mobile phone",
        37: "using a computer",
        38: "smoking",
        39: "clapping",
        40: "rubbing face with hands",
        41: "doing freehand exercise",
        42: "doing neck roll exercise",
        43: "massaging a shoulder oneself",
        44: "taking a bow",
        45: "talking to each other",
        46: "handshaking",
        47: "hugging each other",
        48: "fighting each other",
        49: "waving a hand",
        50: "flapping a hand up and down (beckoning)",
        51: "pointing with a finger",
        52: "opening the door and walking in",
        53: "fallen on the floor",
        54: "sitting up/standing up",
        55: "lying down",
    }
    etri_action_inv={v:k for k,v in etri_actions.items()}


    # ntu120_list=[ntu_actions[i+1] for i in range(120)]
    ntu120_list=[ntu60_actions[i+1] for i in range(60)]
    etri55_list=[etri_actions[i+1] for i in range(55)]

    # print(ntu120_list)
    # print(etri55_list)


    ntu23_dt = {
    0 : ['eat meal/snack'] ,
    1 : ['drink water'] ,
    2 : ['brushing teeth'] ,
    3 : ['rub two hands together'] ,
    4 : ['brushing hair'] ,
    5 : ['wear jacket'] , 
    6 : ['take off jacket'] , 
    7 : ['wear a shoe', 'take off a shoe'] , 
    8 : ['wear on glasses', 'take off glasses'] , 
    9 : ['reading'] , 
    10 : ['writing'] , 
    11 : ['make a phone call/answer phone'] , 
    12 : ['playing with phone/tablet'] ,
    13 : ['typing on a keyboard'] , 
    14 : ['clapping'] , 
    15 : ['wipe face'] , 
    16 : ['nod head/bow'] , 
    # 17 : ['shake head'] , 
    17 : ['handshaking'] , 
    18 : ['hugging other person'] ,
    19 : ['punching/slapping other person'] , 
    20 : ['hand waving'] , 
    21 : ['pointing to something with finger'] ,
    22 : ['falling'] ,
    }

    etri23_dt = {
        0 : ['eating food with a fork'] , 
        1 : ['drinking water'] , 
        2 : ['brushing teeth'] , 
        3 : ['washing hands'] , 
        4 : ['brushing hair'] , 
        5 : ['putting on a jacket'] , 
        6 : ['taking off a jacket'] , 
        7 : ['putting on/taking off shoes'] , 
        8 : ['putting on/taking off glasses'] , 
        9 : ['reading a book'] , 
        10 : ['handwriting'] , 
        11 : ['talking on the phone'] , 
        12 : ['playing with a mobile phone'] , 
        13 : ['using a computer'] , 
        14 : ['clapping'] , 
        15 : ['rubbing face with hands'] , 
        16 : ['taking a bow'] ,
        17 : ['handshaking'] , 
        18 : ['hugging each other'] , 
        19 : ['fighting each other'] , 
        20 : ['waving a hand'] , 
        21 : ['pointing with a finger'] , 
        22 : ['fallen on the floor'] , 
    }


    ntu23_action_idx_dt = {}
    for i in range(23):
        v_list = ntu23_dt[i]
        ntu23_action_idx_dt[i] = [ntu120_list.index(v) for v in v_list]


    etri23_action_idx_dt = {}
    for i in range(23):
        v_list = etri23_dt[i]
        etri23_action_idx_dt[i] = [etri55_list.index(v) for v in v_list]

    # print(ntu23_action_idx_dt)
    # print(etri23_action_idx_dt)


    ntu60_to_23_list = [None]*23 
    etri55_to_23_list = [None]*23

    for k,v in ntu23_action_idx_dt.items():
        if len(v)==1:
            ntu60_to_23_list[k] = v[0]
        else:
            ntu60_to_23_list[k] = v 
    
    for k,v in etri23_action_idx_dt.items():
        if len(v)==1:
            etri55_to_23_list[k] = v[0]
        else:
            etri55_to_23_list[k] = v 

    # print(ntu60_to_23_list)
    # print(etri55_to_23_list)

    return ntu60_to_23_list, etri55_to_23_list


    



def get_mapping_common18_pku():
    # 1indexed because we map labels in different datasets via their text (there are 1-indexed tags in text.)
    # ntu and etri are first mapped by common23 classes and then mapped to common18 classes.
    common_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    pku_idx_list_1indexed =[10, 8, 3, 2, 48, 37,[36,49],30,51,20,23,6,1,14,16,13,25,11]
    pku_idx_list_0indexed  = index1_to_index0(pku_idx_list_1indexed)
    pku_to_d3  = to_dict(pku_idx_list_0indexed, common_idx)

    return pku_to_d3
    


def get_mapping_common18_ntu():

    # get ntu60->common23 mapping
    ntu60_to_23_list, etri55_to_23_list = get_mapping_ntu_etri_common23()

    # get common23->common18 mapping
    common23_to_18_idx_list_1indexed=[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 17, 18, 19, 21, 22, 23]
    common23_to_18_idx_list_0indexed = index1_to_index0(common23_to_18_idx_list_1indexed)

    # 0-indexed
    ntu60_to_18_list = [ntu60_to_23_list[k] for k in common23_to_18_idx_list_0indexed]

    common_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ntu60_to_d3  = to_dict(ntu60_to_18_list, common_idx)

    return ntu60_to_d3
    

def get_mapping_common18_etri():

    # get etri55->common23 mapping
    ntu60_to_23_list, etri55_to_23_list = get_mapping_ntu_etri_common23()

    # get common23->common18 mapping
    common23_to_18_idx_list_1indexed=[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 17, 18, 19, 21, 22, 23]
    common23_to_18_idx_list_0indexed = index1_to_index0(common23_to_18_idx_list_1indexed)

    # 0-indexed
    etri55_to_18_list= [etri55_to_23_list[k] for k in common23_to_18_idx_list_0indexed]

    common_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    etri55_to_d3  = to_dict(etri55_to_18_list, common_idx)

    return etri55_to_d3
    



#######################################################################
#  utils
#######################################################################

def save_pkl(fname, dt):
    with open(fname, "wb") as f:
        pickle.dump(dt, f)


def load_data(file_name):
    return np.load(file_name,mmap_mode="r")
    # return np.load(file_name)

def load_pkl(file_name):
    return pickle.load(open(file_name,"rb"))


def save_npy(save_name, data):
    os.makedirs(os.path.dirname(save_name),exist_ok=True)
    np.save(save_name, data)



#######################################################################
#  gen data 
#######################################################################


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



def main_gen_common18_each(src_data_dir, dst_dir, mapping_to_d3, domain, tag_joint_in_name=True):

    # given all data, process to common18
    pku_to_d3 = mapping_to_d3
   
    # pku 
    # test_data.npy    test_label.pkl   
    # train_data.npy   train_label.pkl

    for split in ['train', 'test']:

        if tag_joint_in_name:
            old_data_pku_file = f"{pku_data_dir}/{split}_data_joint.npy"
        else:
            old_data_pku_file = f"{pku_data_dir}/{split}_data.npy"

        old_label_pku_file =f"{pku_data_dir}/{split}_label.pkl"

        old_data_pku = load_data(old_data_pku_file)
        old_label_pku = load_pkl(old_label_pku_file)

        new_data_pku, new_label_pku   = to_data_label_with_mapping(old_data_pku, old_label_pku, pku_to_d3)

        new_data_pku_file =   f"{dst_dir}/{domain}/{split}_data_joint.npy"
        new_label_pku_file =  f"{dst_dir}/{domain}/{split}_label.pkl"

        print("-> save to", new_data_pku_file)
        print("-> save to", new_label_pku_file)

        save_npy(new_data_pku_file, new_data_pku)
        save_pkl(new_label_pku_file, new_label_pku)
    

def main_gen_common18_pku(src_data_dir, dst_dir, domain):

    # given all data, process to common18
    mapping_to_d3 = get_mapping_common18_pku()
    # domain = "pku"

    for split in ['train', 'test']:

        if split=="train":
            old_data_pku_file = f"{src_data_dir}/train_data.npy"
        elif split=="test":
            old_data_pku_file = f"{src_data_dir}/test_data.npy"
        else:
            raise ValueError()

        old_label_pku_file =f"{src_data_dir}/{split}_label.pkl"

        old_data_pku = load_data(old_data_pku_file)
        old_label_pku = load_pkl(old_label_pku_file)

        new_data_pku, new_label_pku   = to_data_label_with_mapping(old_data_pku, old_label_pku, mapping_to_d3)

        new_data_pku_file =   f"{dst_dir}/{domain}/{split}_data_joint.npy"
        new_label_pku_file =  f"{dst_dir}/{domain}/{split}_label.pkl"

        print("-> save to", new_data_pku_file)
        print("-> save to", new_label_pku_file)

        save_npy(new_data_pku_file, new_data_pku)
        save_pkl(new_label_pku_file, new_label_pku)


def main_gen_common18_ntu(src_data_dir, dst_dir, domain, reorder_index_dt):

    # given all data, process to common18
    mapping_to_d3 = get_mapping_common18_ntu()

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

        new_data_pku, new_label_pku   = to_data_label_with_mapping(old_data_pku, old_label_pku, mapping_to_d3)

        # reorder index
        reorder_index = reorder_index_dt[f"{split}"]
        new_data_pku, new_label_pku = reorder_sample(new_data_pku, new_label_pku, reorder_index)



        new_data_pku_file =   f"{dst_dir}/{domain}/{split}_data_joint.npy"
        new_label_pku_file =  f"{dst_dir}/{domain}/{split}_label.pkl"

        print("-> save to", new_data_pku_file)
        print("-> save to", new_label_pku_file)

        save_npy(new_data_pku_file, new_data_pku)
        save_pkl(new_label_pku_file, new_label_pku)




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




def main_gen_common18_etri(src_data_dir, dst_dir, domain, reorder_index_dt):

    # given all data, process to common18
    mapping_to_d3 = get_mapping_common18_etri()

    for split in ['train', 'test']:
    # for split in ['test']:
        if split=="train":
            old_data_pku_file = f"{src_data_dir}/train_data_joint.npy"
        elif split=="test":
            old_data_pku_file = f"{src_data_dir}/test_data_joint.npy"
        else:
            raise ValueError()

        old_label_pku_file =f"{src_data_dir}/{split}_label.pkl"

        old_data_pku = load_data(old_data_pku_file)
        old_label_pku = load_pkl(old_label_pku_file)

        # reorder sample here 
        if domain in ["adult", "etriA"]:
            reorder_domain_tag = "adult"
        elif domain in ["elder", "etriE"]:
            reorder_domain_tag = "elder"
        else:
            raise ValueError()

        reorder_index = reorder_index_dt[f"{reorder_domain_tag}_{split}"]
        old_data_pku, old_label_pku = reorder_sample(old_data_pku, old_label_pku, reorder_index)


        new_data_pku, new_label_pku   = to_data_label_with_mapping(old_data_pku, old_label_pku, mapping_to_d3)

        new_data_pku_file =   f"{dst_dir}/{domain}/{split}_data_joint.npy"
        new_label_pku_file =  f"{dst_dir}/{domain}/{split}_label.pkl"

        print("-> save to", new_data_pku_file)
        print("-> save to", new_label_pku_file)

        save_npy(new_data_pku_file, new_data_pku)
        save_pkl(new_label_pku_file, new_label_pku)




def main_debug():

    print(get_mapping_common18_pku())
    print(get_mapping_common18_ntu())
    print(get_mapping_common18_etri())


def main():
    args = get_parser().parse_args()
    domain = args.domain 

    dst_dir = '../data/processed_data/common18/xsub300'
    os.makedirs(dst_dir, exist_ok=True)

    if domain=="pku":
        main_gen_common18_pku(src_data_dir="../data/processed_data/pku", dst_dir=dst_dir, domain="pku")

    elif domain=="ntu":
        reorder_index_dt = np.load("./ntu_common18_sample_order_index.npy", allow_pickle=True).item()
        main_gen_common18_ntu(src_data_dir="../data/processed_data/ntu/xsub", dst_dir=dst_dir, domain="ntu", reorder_index_dt=reorder_index_dt)

    elif domain=="etriA":
        reorder_index_dt = np.load("./etri_full_sample_order_index.npy", allow_pickle=True).item()
        main_gen_common18_etri(src_data_dir="../data/processed_data/etri/adult", dst_dir=dst_dir, domain="etriA", reorder_index_dt=reorder_index_dt)
    
    elif domain=="etriE":
        reorder_index_dt = np.load("./etri_full_sample_order_index.npy", allow_pickle=True).item()
        main_gen_common18_etri(src_data_dir="../data/processed_data/etri/elder", dst_dir=dst_dir, domain="etriE", reorder_index_dt=reorder_index_dt)

    else:
        raise ValueError()



#######################################################################
#  gen few-shot 
#######################################################################

def get_fewshot_data_label(data_list, label_list,fs=1):
    sample_list, label_list = label_list 
    label_list = np.array(label_list).astype(int)
    sample_list = np.array(sample_list).astype(str)
    class_list = np.unique(label_list)
    sample_res = []
    data_res = []
    label_res = []
    for c in class_list:
        selected_idx_list = np.where(label_list==c)[0][:fs]
        # print(c, selected_idx_list)
        data_res.append( data_list[selected_idx_list] )
        label_res.append( label_list[selected_idx_list] )
        sample_res.append( sample_list[selected_idx_list] )
    data_res = np.concatenate(data_res, 0)
    label_res = np.concatenate(label_res, 0).tolist()
    sample_res = np.concatenate(sample_res, 0).tolist()
    label_res = [sample_res, label_res]
    return data_res, label_res


def main_get_fewshot():
    dt ={
        'train': {},
        'test': {},
        'label_name_list': get_common18_label()
    }
    for split in ['train', 'test']:
        for domain in ['ntu', 'pku', 'etri']:
            data_file = f"/mnt/action/data/data220919/common18_d3/xsub64/{domain}/{split}_data_joint.npy"
            label_file= f"/mnt/action/data/data220919/common18_d3/xsub64/{domain}/{split}_label.pkl"

            data = load_data(data_file)
            labels = load_pkl(label_file)

            data, labels = get_fewshot_data_label(data, labels, 2)
            dt[split][domain] = [data, labels]

    save_path = '/mnt/results/common18_d3_fs2.pkl'
    save_pkl(save_path, dt)




if __name__ == "__main__":
    main()
    
