#!/bin/bash

# ensure the script exits on error, undefined variables, and pipe failures
set -euo pipefail

export PYTHONPATH='.'

base_model=$1

prune_ckpt_path="prune_log/${base_model}"
tune_ckpt_path="tune_log/${base_model}"
mkdir -p $prune_ckpt_path
mkdir -p $tune_ckpt_path

# echo "[START] - Start Pruning Model ${base_model}"
# python3 pruner.py --pruning_ratio 0.28571428571 \
#                  --device cpu --eval_device cpu \
#                  --base_model $base_model \
#                  --save_ckpt_log_name $base_model \
#                  --max_seq_len 2048 \
#                  --save_model \
#                  --ignore_first_x_layers 1 \
#                  --ignore_last_x_layers 1 \
#                 #  --test_after_train \
# echo "[FINISH] - Finish Pruning Model ${base_model}"

# # Check if the pruned model file exists
# if [ ! -f "$prune_ckpt_path/pytorch_model.bin" ]; then
#     echo "Error: File $prune_ckpt_path/pytorch_model.bin does not exist"
#     exit 1
# fi

echo "[START] - Start Tuning ${base_model}"
CUDA_VISIBLE_DEVICES=0 python lora_training.py \
    --prune_model $prune_ckpt_path/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir $tune_ckpt_path \
    --wandb_project $base_model \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64
echo "[FINISH] - LoRA fine tuning of ${base_model}."
echo "[INFO] - The pruned model is at ${prune_ckpt_path}/pytorch_model.bin, and the recovery weights are at ${tune_ckpt_path}/"

# take the latest epoch from the tune_log folder
epoch=$(ls ${tune_ckpt_path} | grep '^checkpoint-' | sed 's/checkpoint-//' | sort -n | tail -n 1)

# Check if epoch is empty
if [ -z "$epoch" ]; then
    echo "Error: No checkpoints found in $tune_ckpt_path"
    exit 1
fi
echo "Latest epoch found: $epoch"

# Evaluate the pruned model
echo "[START] - Start Evaluation for Pruned ${base_model} at epoch ${epoch}"
bash scripts/evaluation/pruned_evaluate.sh $base_model $tune_ckpt_path $prune_ckpt_path $epoch
echo "[FINISH] - Finish Evaluation for Pruned ${base_model} at epoch ${epoch}"

# Evaluate the vanilla model
echo "[START] - Start Evaluation for Vanilla ${base_model}"
bash scripts/evaluation/vanilla_evaluate.sh $base_model
echo "[FINISH] - Finish Evaluation for Vanilla ${base_model}"
