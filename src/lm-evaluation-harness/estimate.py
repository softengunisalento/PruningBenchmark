import argparse
import json
import lm_eval.models
from transformers import GenerationConfig
import torch
from datasets import load_dataset
from codecarbon import EmissionsTracker
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import json

class PromptDataset(Dataset):
    def __init__(self, user_messages, tokenizer, max_length=512):
        self.user_messages = user_messages
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.user_messages)
    
    def __getitem__(self, idx):
        return self.user_messages[idx]

def collate_fn(batch, tokenizer, device):
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    return inputs

def main(args):
    # Initialize model
    lm = lm_eval.models.get_model("hf-causal-experimental").create_from_arg_string(
        args.model_args, {"batch_size": args.batch_size, "device": args.device}
    )
    lm.model.generation_config.prefill_chunk_size = None
    
    # Load dataset
    dataset = load_dataset("allenai/WildChat-1M", split="train[:100000]")
    dataset = dataset.filter(lambda x: x["language"] == "English" and (x["turn"]) == 1)
    dataset = dataset.shuffle(seed=11).select(range(args.num_samples))
    
    # Format prompts
    special_tokens = {
        'bos_token': getattr(lm.tokenizer, 'bos_token', '$$$'),
        'eos_token': getattr(lm.tokenizer, 'eos_token', '###'),
        'sep_token': getattr(lm.tokenizer, 'sep_token', '\n\n### '),
    }
    
    user_messages = [
        turn['content'] 
        for conv in dataset["conversation"] 
        for turn in conv 
        if turn['role'] == 'user'
    ]
    
    # Prepare DataLoader
    prompt_dataset = PromptDataset(user_messages, lm.tokenizer)
    dataloader = DataLoader(
        prompt_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, lm.tokenizer, lm.device)
    )

    # Initialize metrics
    total_input_tokens = 0
    total_output_tokens = 0
    total_conversations = 0
    
    # Configurazione del tokenizer per evitare warning ripetuti
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    lm.model.generation_config.pad_token_id = lm.tokenizer.pad_token_id
    
    # Start tracking
    tracker = EmissionsTracker(
        project_name="inference",
        output_dir="codecarbon/inference",
        output_file=args.codecarbon_name,
        tracking_mode="process",
        log_level="error", 
        measure_power_secs=1,
        save_to_file=True,
        gpu_ids=[0],
    )
    tracker.start()
    start_time = time.time()

    # Inference loop
    output_data = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", unit="batch")):
            # Count input tokens
            input_lengths = (batch['input_ids'] != lm.tokenizer.pad_token_id).sum(dim=1)
            total_input_tokens += input_lengths.sum().item()
            
            # Generate output
            generation_output = lm.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                do_sample=True,
                top_k=20,
                max_new_tokens=args.max_new_tokens,
                top_p=0.7,
                temperature=0.3,
                early_stopping=True,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                eos_token_id=lm.tokenizer.eos_token_id, 
                # return_dict_in_generate=True,
                # output_scores=True,
                num_beams=4,
            )
            
            # Count output tokens
            output_lengths = (generation_output.sequences != lm.tokenizer.pad_token_id).sum(dim=1)
            total_output_tokens += output_lengths.sum().item()
            total_conversations += len(batch['input_ids'])
            
            # Decode results
            prompts = lm.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            results = [lm.tokenizer.decode(gen, skip_special_tokens=True) for gen in generation_output.sequences]
            
            for prompt, result in zip(prompts, results):
                output_data.append({
                    "prompt": prompt,
                    "response": result,
                    "batch_idx": batch_idx
                })

    # Stop tracking and calculate metrics
    total_time = time.time() - start_time
    emissions = tracker.stop()
    total_tokens = total_input_tokens + total_output_tokens
    new_tokens = total_output_tokens - total_input_tokens
    
    # Calculate metrics
    avg_input_length = total_input_tokens / total_conversations
    avg_output_length = total_output_tokens / total_conversations
    tokens_per_second = total_tokens / total_time
    energy_per_token = tracker._total_energy.kWh * 1000 * 1000 / total_tokens  # mWh per token
    energy_per_conversation = tracker._total_energy.kWh * 1000 / total_conversations  # Wh per conversation
    
    results = {
        "Performance Metrics": {
            "Average input length (tokens)": round(avg_input_length, 1),
            "Average output length (tokens)": round(avg_output_length, 1),
            "Total tokens processed": total_tokens,
            "New tokens generated": new_tokens,
            "Tokens per second": round(tokens_per_second, 1),
            "Energy per token (mWh/token)": round(energy_per_token, 4),
            "Energy per conversation (Wh/conversation)": round(energy_per_conversation, 4),
        },
        "Energy Results": {
            "GPU energy (kWh)": round(tracker._total_gpu_energy.kWh, 4),
            "CPU energy (kWh)": round(tracker._total_cpu_energy.kWh, 4),
            "RAM energy (kWh)": round(tracker._total_ram_energy.kWh, 4),
            "Total energy (kWh)": round(tracker._total_energy.kWh, 4),
            "Duration (s)": round(total_time, 1)
        }
    }
    
    print(results)

    # Save results if specified
    if args.chat_file:
        with open(args.chat_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.chat_file}")
        
    if args.out_file:
        with open(args.out_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run batch inference with energy tracking')
    parser.add_argument('--model_args', type=str, 
                        default="pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help='Model arguments string')
    parser.add_argument('--num_samples', type=int, default=512,
                        help='Number of samples to process')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                    help='Maximum number of new tokens to generate')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--chat_file', type=str, default=None,
                        help='Path to JSON file to chat results (optional)')
    parser.add_argument('--codecarbon_name', type=str, default="inference", 
                        help='Name for CodeCarbon .csv tracking file')
    parser.add_argument('--out_file', type=str, default=None,
                    help='Path to JSON file to save results (optional)')
    
    args = parser.parse_args()
    main(args)