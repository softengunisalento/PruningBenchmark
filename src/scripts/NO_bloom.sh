#!/bin/bash

# "bigscience/bloom-7b1"
# sembra funzionare
# bloom funziona solo col vecchio codice
export PYTHONPATH='.'

base_model="bigscience/bloomz-7b1"
model_name='bloomz-7b1'
prune_ckpt_path="prune_log/$model_name"
tune_ckpt_path="tune_log/$model_name"

echo "[START] - Start Pruning Model"

# python examples/old_bloom.py  \
#     --base_model $base_model \
#     --pruning_ratio 0.28571428571 \
#     --block_wise \
#     --block_mlp_layer_start 4 --block_mlp_layer_end 25 \
#     --block_attention_layer_start 4 --block_attention_layer_end 25 \
#     --pruner_type taylor \
#     --device cpu  --eval_device cuda \
#     --test_after_train \
#     --save_ckpt_log_name $model_name

python examples/bloom_v2.py --pruning_ratio 0.28571428571 \
                 --device cpu --eval_device cuda \
                 --base_model $base_model \
                 --save_ckpt_log_name $model_name \
                 --max_seq_len 2048 \
                #  --save_model \

echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=0 python post_training.py \
    --prune_model $prune_ckpt_path/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir $tune_ckpt_path \
    --wandb_project $model_name \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --lora_target_modules "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"

echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {$tune_ckpt_path}/"

epoch=1400
bash scripts/iter_evaluate.sh $model_name $base_model $tune_ckpt_path $prune_ckpt_path $epoch
# bash scripts/vanilla_evaluate.sh $model_name $base_model
bash scripts/notifica.sh