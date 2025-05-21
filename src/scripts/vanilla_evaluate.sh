#!/bin/bash
export PYTHONPATH='.'

model_name=$1
base_model=$2 # e.g., baffo32/decapoda-research-llama-7B-hf

# Lista dei task disponibili
tasks=("openbookqa" "arc_easy" "winogrande" "hellaswag" "arc_challenge" "piqa" "boolq")
# tasks=("hellaswag")

# Dizionario per mappare i task ai rispettivi batch size
declare -A batch_sizes
batch_sizes["openbookqa"]=1 #23149 ERA 128
batch_sizes["arc_easy"]=1 #23933 ERA 64
batch_sizes["winogrande"]=1 #22817
batch_sizes["hellaswag"]=64 #22761
batch_sizes["arc_challenge"]=1 #21433 ERA 64
batch_sizes["piqa"]=1 #21967
batch_sizes["boolq"]=1 #23135

# Itera sui task
for i in $(seq 6 10);
do
    for task in "${tasks[@]}"; 
    do
        batch_size=${batch_sizes[$task]}

        python lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$base_model \
            --tasks $task \
            --device cuda:0 \
            --output_path results/vanilla/${model_name}/${task}/${i}.json \
            --no_cache \
            --batch_size $batch_size
    done
done