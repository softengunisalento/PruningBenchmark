#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., baffo32/decapoda-research-llama-7B-hf
tune_ckpt_name=$2 # tune_log/{name} folder
prune_ckpt=$3 # prune_log/{name} folder
epoch=$4

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

# Prepara il checkpoint per l'adattatore
cp $tune_ckpt_name/adapter_config.json $tune_ckpt_name/checkpoint-$epoch/
mv $tune_ckpt_name/checkpoint-$epoch/pytorch_model.bin $tune_ckpt_name/checkpoint-$epoch/adapter_model.bin

# Itera sui task
for i in $(seq 1 10);
do
    for task in "${tasks[@]}"; 
    do
        # Ottieni il batch size corretto per il task corrente
        batch_size=${batch_sizes[$task]}
        
        python lm-evaluation-harness/main.py \
            --model hf-causal-experimental \
            --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/checkpoint-$epoch,config_pretrained=$base_model \
            --tasks $task \
            --device cuda:0 \
            --output_path results/pruned/${base_model}/${task}/${i}.json \
            --no_cache \
            --batch_size $batch_size
    done
done

