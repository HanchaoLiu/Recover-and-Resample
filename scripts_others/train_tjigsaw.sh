

device=$1

recover="none"
resample="none"
prior_data_tag="train"
model_prior_weights_dir="x"
interp_type="linear"

work_dir="common18_${dataset}_tjigsaw_saveres_noresample"

python main_others/train_3domain_tjigsaw.py \
    --config configs/dataset_base_priornet.yaml \
    --work-dir ${work_dir} \
    --train-feeder-args "theta=30" \
    --device ${device} \
    --src N E \
    --n_seed 10 \
    --interp_type ${interp_type} \
    --model "nets.agcn.agcn.Model_thin_jigsaw" \
    --model-args "n_jigsaw=6" \
    --n_jigsaw 3 \
    --w_jigsaw 0.1 \
    --num-epoch 50 \
    --dataset_dir "data/common18/xsub64"

# --train-feeder "feeders.feeder_dg_resample.Feeder_resample" \

