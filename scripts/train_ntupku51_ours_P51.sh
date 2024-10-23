

device=$1
recover="ABL_linearextrap_beta"
resample="cropresize0.7"
bkg_type="train"
tr_type="train"

prior_data_tag="none"

n_bkg=5
n_tr=10

src="P51"

model_prior_weights_dir="x"
interp_type="linear"

work_dir="ntupku51_${dataset}_${recover}_${resample}_BKG${bkg_type}${n_bkg}_TR${tr_type}${n_tr}_final_hcn2"

python main/train_ntupku51_ours_P51.py \
    --config configs/dataset_base_priornet_hcn.yaml \
    --work-dir ${work_dir} \
    --train-feeder-args "theta=30" \
    --resample_tag ${resample} --recover_tag ${recover} \
    --device ${device} \
    --n_cluster_w ${n_tr} --n_cluster_bkg ${n_bkg} \
    --src ${src} \
    --n_seed 1 \
    --interp_type ${interp_type} \
    --bkg_type ${bkg_type} --tr_type ${tr_type} \
    --cluster_dir "data/cluster_result" \
    --dataset_dir "data/ntupku51/xsub64"
    

