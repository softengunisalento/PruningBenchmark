#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., baffo32/decapoda-research-llama-7B-hf

# Lista dei task disponibili
tasks=("openbookqa" "arc_easy" "winogrande" "hellaswag" "arc_challenge" "piqa" "boolq")

for task in "${tasks[@]}";
do
    # Crea la cartella per il task se non esiste
    mkdir -p results/pruned/${base_model}/${task}
done

# Dizionario per mappare i task ai rispettivi batch size
declare -A batch_sizes
batch_sizes["openbookqa"]=1
batch_sizes["arc_easy"]=1
batch_sizes["winogrande"]=1
batch_sizes["hellaswag"]=64
batch_sizes["arc_challenge"]=1
batch_sizes["piqa"]=1
batch_sizes["boolq"]=1

# Itera sui task
for i in $(seq 1 10);
do
    for task in "${tasks[@]}"; 
    do
        batch_size=${batch_sizes[$task]}

        python lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args pretrained=$base_model \
            --tasks $task \
            --device cuda:0 \
            --output_path results/vanilla/${base_model}/${task}/${i}.json \
            --no_cache \
            --batch_size $batch_size
    done
done