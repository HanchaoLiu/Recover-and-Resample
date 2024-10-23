

device=$1

interp_type="linear"

work_dir="ntupku51_${dataset}_tjigsaw_saveres_resample_pku51_hcn"

python main_others/train_ntupku51_tjigsaw_hcn.py \
    --config configs/dataset_base_priornet_hcn.yaml \
    --work-dir ${work_dir} \
    --train-feeder-args "theta=30" \
    --device ${device} \
    --src N51 \
    --n_seed 1 \
    --interp_type ${interp_type} \
    --model "nets.agcn.hcn.HCN_jigsaw" \
    --model-args "n_jigsaw=6" \
    --n_jigsaw 3 \
    --w_jigsaw 0.1 \
    --dataset_dir "data/ntupku51/xsub64"


# --train-feeder "feeders.feeder_dg_resample.Feeder_resample" \


