# sembra funzionare
#!/bin/bash
export PYTHONPATH='.'

prune_ckpt_path='prune_log/Qwen2-5-7B'
tune_ckpt_path='tune_log/Qwen2-5-7B'
base_model="Qwen/Qwen2.5-7B"
model_name='Qwen2-5-7B'

echo "[START] - Start Pruning Model"

# python pruner.py --pruning_ratio 0.32 \
#                  --device cpu --eval_device cuda \
#                  --base_model $base_model \
#                  --save_ckpt_log_name $model_name \
#                  --max_seq_len 2048 \
#                  --test_after_train \
#                  --save_model

# python pruner.py --pruning_ratio 0.28571428571 \
#                  --device cpu --eval_device cuda \
#                  --base_model $base_model \
#                  --save_ckpt_log_name $model_name \
#                  --max_seq_len 2048 \
#                  --test_after_train \
#                  --save_model

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

echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {$tune_ckpt_path}/"

epoch=1554
bash scripts/pruned_evaluate.sh $model_name $base_model $tune_ckpt_path $prune_ckpt_path $epoch
bash scripts/vanilla_evaluate.sh $model_name $base_model

bash scripts/notifica.sh