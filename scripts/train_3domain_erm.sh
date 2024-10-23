


device=$1
recover="none"
resample="none"
bkg_type="train"
tr_type="train"

prior_data_tag="none"

n_bkg=10
n_tr=20

model_prior_weights_dir="x"
interp_type="linear"

work_dir="common18_${dataset}_${recover}_${resample}_BKG${bkg_type}${n_bkg}_TR${tr_type}${n_tr}_final_erm"

python main/train_3domain_erm.py \
    --config configs/dataset_base_priornet.yaml \
    --work-dir ${work_dir} \
    --train-feeder-args "theta=30" \
    --resample_tag ${resample} --recover_tag ${recover} \
    --device ${device} \
    --n_cluster_w ${n_tr} --n_cluster_bkg ${n_bkg} \
    --src N E \
    --n_seed 10 \
    --interp_type ${interp_type} \
    --bkg_type ${bkg_type} --tr_type ${tr_type} \
    --cluster_dir "data/cluster_result" \
    --dataset_dir "data/common18/xsub64"
    


