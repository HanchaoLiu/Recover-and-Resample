import os,sys
import numpy as np 
import pickle
from IPython import embed



subsetting_list= [
    ['N', 'ntu', 'etri'],
    ['N', 'ntu', 'pku'],
    ['E', 'etriA', 'ntu'],
    ['E', 'etriA', 'pku'],
]

def load_file(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    sample_list = [i[0] for i in data]
    score_list =  [i[1] for i in data]
    gt_list     = [i[2] for i in data]

    score_list = np.array(score_list)
    gt_list = np.array(gt_list)
    print(score_list.shape, gt_list.shape)
    return score_list, gt_list


def get_bal_acc(pred, gt):
    perclass_acc_list = []
    pred = np.array(pred)
    gt = np.array(gt)
    unique_label = np.unique(gt)
    for c in unique_label:
        acc = (pred[gt==c]==gt[gt==c]).mean()
        perclass_acc_list.append( acc )
    perclass_acc_mean = np.mean(perclass_acc_list)
    return perclass_acc_mean



def main():
    res_dt_st={}
    res_dt_s={}
    res_dt_t={}
    for (src_tag_short, src_tag, dst_tag) in subsetting_list:
        name = f"{src_tag}_{dst_tag}"
        if name not in res_dt_st.keys():
            res_dt_st[name]=[]
            res_dt_s[name]=[]
            res_dt_t[name]=[]


    for seed in [0,1,2,3,4,5,6,7,8,9]:
        for (src_tag_short, src_tag, dst_tag) in subsetting_list:
            file_pat_s = f'workdir/ws/common18__sjigsaw_saveres_noresample_{src_tag_short}_seed{seed}/{src_tag}_{dst_tag}_epochbest_score.pkl'
            assert os.path.exists(file_pat_s)
            print(file_pat_s)
            score_s, gt_s = load_file(file_pat_s)
            pred_idx_s = np.argmax(score_s, axis=1)
            acc_s = get_bal_acc(pred_idx_s, gt_s)
            print("acc_s = ", np.round(acc_s,3))

            file_pat_t = f'workdir/ws/common18__tjigsaw_saveres_noresample_{src_tag_short}_seed{seed}/{src_tag}_{dst_tag}_epochbest_score.pkl'
            assert os.path.exists(file_pat_t)
            print(file_pat_t)
            score_t, gt_t = load_file(file_pat_t)
            pred_idx_t = np.argmax(score_t, axis=1)
            acc_t = get_bal_acc(pred_idx_t, gt_t)
            print("acc_t = ", np.round(acc_t,3))

            score_st = score_s + score_t 
            assert np.all(gt_s==gt_t)
            pred_idx_st = np.argmax(score_st, axis=1)
            acc_st = get_bal_acc(pred_idx_st, gt_t)
            print("acc_st = ", np.round(acc_st, 3))

            name = f"{src_tag}_{dst_tag}"
            res_dt_st[name].append( acc_st )
            res_dt_s[name].append( acc_s )
            res_dt_t[name].append( acc_t )

    
    import pandas as pd 
    print("="*80)
    print("df_st")
    df_st = pd.DataFrame(res_dt_st)
    df_st.loc['mean'] = df_st.mean()
    df_st = df_st.round(3)
    print(df_st)

    print("="*80)
    print("df_s")
    df_s = pd.DataFrame(res_dt_s)
    df_s.loc['mean'] = df_s.mean()
    df_s = df_s.round(3)
    print(df_s)

    print("="*80)
    print("df_t")
    df_t = pd.DataFrame(res_dt_t)
    df_t.loc['mean'] = df_t.mean()
    df_t = df_t.round(3)
    print(df_t)





if __name__ == "__main__":
    main()