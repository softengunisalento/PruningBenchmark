#!/bin/bash

export PYTHONPATH='.'
prune_ckpt_path='prune_log/gemma-2-2b-it'
tune_ckpt_path='tune_log/gemma-2-2b-it'
base_model='google/gemma-2-2b-it'
model_name='gemma-2-2b-it'

echo "[START] - Start Pruning Model"

python pruner.py --pruning_ratio 0.28571428571 \
                 --device cpu --eval_device cuda \
                 --base_model $base_model \
                 --save_ckpt_log_name $model_name \
                 --max_seq_len 2048 \
                 --ignore_first_x_layers 2 \
                 --ignore_last_x_layers 2 \
                #  --save_model \

# echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=0 python post_training.py \
#     --prune_model $prune_ckpt_path/pytorch_model.bin \
#     --data_path yahma/alpaca-cleaned \
#     --output_dir $tune_ckpt_path \
#     --wandb_project $model_name \
#     --lora_r 8 \
#     --num_epochs 2 \
#     --learning_rate 1e-4 \
#     --batch_size 256 \

# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {$tune_ckpt_path}/"

# epoch=388
# bash scripts/iter_evaluate.sh $model_name $base_model $tune_ckpt_path $prune_ckpt_path $epoch
# bash scripts/vanilla_evaluate.sh $model_name $base_model

# bash scripts/notifica.sh