import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_llm(model_name, max_seq_len=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    if hasattr(model.config, "max_position_embeddings"):
        model.seqlen = min(max_seq_len, model.config.max_position_embeddings) if max_seq_len is not None else model.config.max_position_embeddings

    return model
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = get_llm(args.base_model, args.max_seq_len)
    if args.device != "cpu":
        model.half()
    
    model = model.to(args.device)
    
    # fix out of memory
    model = model.to(torch.bfloat16)

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log("Use {} pruner...".format('taylor'))
    
    print("!"*20)
    print("Before pruning")
    print(model)
    print("!"*20)

    ##############
    # Pruning
    ##############
    import torch_pruning as tp 
    # from LLMPruner.torch_pruning.utils import print_tool

    tp.utils.print_tool.before_pruning(model)
    text = "Hello world."
    inputs = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)
    num_heads = {}
    out_channel_groups = {}
    seperate_qkv = False
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if hasattr(m, "q_proj"):
                seperate_qkv = True
                num_heads[m.q_proj] = model.config.num_attention_heads
                num_heads[m.k_proj] = model.config.num_key_value_heads
                num_heads[m.v_proj] = model.config.num_key_value_heads
            elif hasattr(m, "qkv_proj"):
                seperate_qkv = False
                num_heads[m.qkv_proj] = model.config.num_attention_heads
        if name.endswith('mlp'):
            if hasattr(m, "gate_up_proj"):
                out_channel_groups[m.gate_up_proj] = 2
    
    _is_gqa = model.config.num_attention_heads != model.config.num_key_value_heads
    
    print(f"_is_gqa:{_is_gqa}")
    print(f"num_heads:{model.config.num_attention_heads}")
    print(f"num_key_value_heads:{model.config.num_key_value_heads}")
    
    head_pruning_ratio = args.pruning_ratio
    hidden_size_pruning_ratio = args.pruning_ratio
    
    # vediamo se risolver l'oom
    model.config.use_cache = args.use_cache
    
    ignored_layers = [model.lm_head, model.model.embed_tokens]
    # Ignoriamo i primi 4 e gli ultimi 4
    if args.ignore_first_x_layers > 0:
        ignored_layers.extend(model.model.layers[:args.ignore_first_x_layers])
    if args.ignore_last_x_layers > 0:
        ignored_layers.extend(model.model.layers[-args.ignore_last_x_layers:])

    print("DEBUG")
    print("Ignored layers: ", ignored_layers)

    importance = tp.importance.GroupTaylorImportance()
    # importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean')
    pruner = tp.pruner.BasePruner(
        model, 
        example_inputs=inputs,
        importance=importance,
        global_pruning=args.global_pruning,
        isomorphic=args.isomorphic,
        output_transform=lambda x: x.logits,
        pruning_ratio=hidden_size_pruning_ratio,
        ignored_layers=ignored_layers,
        num_heads=num_heads,
        prune_num_heads=args.prune_num_heads,
        prune_head_dims=args.prune_head_dims, # we do not prune head dims so that we don't need to prune the ROPE (rotary position embedding)
        head_pruning_ratio=head_pruning_ratio,
        out_channel_groups=out_channel_groups,
        round_to=8,
    )

    model.zero_grad()
    example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
    loss = model(example_prompts, labels=example_prompts).loss
    logger.log("Loss = {}".format(loss))
    loss.backward()

    pruner.step()    
    print("Pruning done")
    
    layer_names = [name for name, _ in model.named_modules() if name.endswith(("self_attn", "mlp"))]
    # *2 = self_attn + mlp
    start_idx = 0
    end_idx = len(layer_names)
    if args.ignore_first_x_layers > 0:
        start_idx = args.ignore_first_x_layers * 2
    if args.ignore_last_x_layers > 0:
        end_idx -= args.ignore_last_x_layers * 2
    layers_to_process = layer_names[start_idx:end_idx]
    
    print("DEBUG")
    print("Layers to process:")
    print(layers_to_process)

    # Update model attributes    
    model.config.hidden_size = model.lm_head.in_features
    for name, m in model.named_modules():
        if name in layers_to_process:
            if name.endswith("self_attn"):
                if seperate_qkv:
                    m.hidden_size = m.q_proj.out_features
                else:
                    m.hidden_size = m.qkv_proj.out_features // 3        
                m.num_heads = m.hidden_size // m.head_dim
                model.config.num_attention_heads = m.num_heads
                #m.head_dim = m.q_proj.out_features // m.num_heads
                if not _is_gqa:
                    m.num_key_value_heads = m.num_heads
                    model.config.num_key_value_heads = m.num_heads
                if hasattr(m, "num_key_value_groups"):
                    m.num_key_value_groups = m.num_heads // model.config.num_key_value_heads

            elif name.endswith("mlp"):
                if hasattr(m, "gate_proj"):
                    m.hidden_size = m.gate_proj.in_features
                    model.config.intermediate_size = m.gate_proj.out_features
                elif hasattr(m, "gate_up_proj"):
                    m.hidden_size = m.gate_up_proj.in_features
                    model.config.intermediate_size = m.gate_up_proj.out_features // 2
                else:
                    raise ValueError("Unknown mlp layer")
        
    if not _is_gqa:
        model.config.num_key_value_heads = model.config.num_attention_heads
    tp.utils.print_tool.after_pruning(model, do_print=True)
    print(model.config)
    
    
    print("!"*20)
    print("Pruning done: Printing Model")
    print(model)
    
    
    print(f"num_heads:{model.config.num_attention_heads}")
    print(f"num_key_value_heads:{model.config.num_key_value_heads}")
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.log("After pruning #parameters: {}".format(after_pruning_parameters))

    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None
    
    del pruner
    gc.collect()
    torch.cuda.empty_cache()

    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    if args.save_model:
        model.half()
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(args.eval_device)

                generation_output = model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    
    parser.add_argument('--global_pruning', action='store_true', help='if global pruning')
    parser.add_argument('--isomorphic', action='store_true', help='if isomorphic pruning')
    parser.add_argument('--prune_num_heads', default=True, action='store_true', help='if prune num heads')
    parser.add_argument('--prune_head_dims', default=False, action='store_true', help='if prune head dims')
    parser.add_argument('--round_to', type=int, default=8, help='round to')


    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')
    
    parser.add_argument('--ignore_first_x_layers', type=int, default=4, help='ignore first x layers')
    parser.add_argument('--ignore_last_x_layers', type=int, default=4, help='ignore last x layers')
    parser.add_argument('--use_cache', action='store_true', help='use cache')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
