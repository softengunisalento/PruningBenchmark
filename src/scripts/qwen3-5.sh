export PYTHONPATH='.'

base_model="Qwen/Qwen3-0.6B"
prune_ckpt_path="prune_log/${base_model}"
tune_ckpt_path="tune_log/${base_model}"

mkdir -p $prune_ckpt_path
mkdir -p $tune_ckpt_path

echo "[START] - Start Pruning Model"

python3 pruner.py --pruning_ratio 0.28571428571 \
                 --device cpu --eval_device cpu \
                 --base_model $base_model \
                 --save_ckpt_log_name $base_model \
                 --max_seq_len 2048 \
                 --test_after_train \
                 --save_model

# echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=0 python post_training.py \
#     --prune_model $prune_ckpt_path/pytorch_model.bin \
#     --data_path yahma/alpaca-cleaned \
#     --output_dir $tune_ckpt_path \
#     --wandb_project $base_model \
#     --lora_r 8 \
#     --num_epochs 2 \
#     --learning_rate 1e-4 \
#     --batch_size 64 \

# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {$tune_ckpt_path}/"

# epoch=1554
# bash scripts/evaluation/pruned_evaluate.sh $base_model $tune_ckpt_path $prune_ckpt_path $epoch
# bash scripts/evaluation/vanilla_evaluate.sh $base_model