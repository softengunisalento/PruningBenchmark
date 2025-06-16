#!/bin/bash
export PYTHONPATH='.'

base_model="meta-llama/Llama-3.1-8B"
model_name='Llama-3-1-8B'
prune_ckpt_path="prune_log/$model_name"
tune_ckpt_path="tune_log/$model_name"

echo "[START] - Start Pruning Model"
# 0.428571428
# 0.28571428571
# python pruner.py --pruning_ratio 0.28571428571 \
#                  --device cpu --eval_device cuda \
#                  --base_model $base_model \
#                  --save_ckpt_log_name $model_name \
#                  --max_seq_len 2048 \
#                  --test_after_train \
#                 #  --save_model

echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=0 python post_training.py \
#     --prune_model $prune_ckpt_path/pytorch_model.bin \
#     --data_path yahma/alpaca-cleaned \
#     --output_dir $tune_ckpt_path \
#     --wandb_project $model_name \
#     --lora_r 8 \
#     --num_epochs 2 \
#     --learning_rate 1e-4 \
#     --batch_size 64 \

# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {$tune_ckpt_path}/"

epoch=1554
# bash scripts/pruned_evaluate.sh $model_name $base_model $tune_ckpt_path $prune_ckpt_path $epoch
# bash scripts/vanilla_evaluate.sh $model_name $base_model

echo "[START] - Start Inference"
# python lm-evaluation-harness/cc_inference.py \
#     --model_args "checkpoint=$prune_ckpt_path/pytorch_model.bin,peft=$tune_ckpt_path/checkpoint-$epoch,config_pretrained=$base_model" \
#     --out_file inference_output/stats/pruned_$model_name.json \
#     --chat_file inference_output/chats/pruned_$model_name.json \
#     --codecarbon_name pruned_$model_name.csv \
#     --batch_size 8 \

python lm-evaluation-harness/cc_inference.py \
    --model_args pretrained=$base_model \
    --out_file inference_output/stats/vanilla_$model_name.json \
    --chat_file inference_output/chats/vanilla_$model_name.json \
    --codecarbon_name vanilla_$model_name.csv \
    --batch_size 8 \
    --max_new_tokens 128 \

bash scripts/notifica.sh