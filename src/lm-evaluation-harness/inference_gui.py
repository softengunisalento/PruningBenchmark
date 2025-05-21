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
import gradio as gr

def main(args):
    # Initialize model
    lm = lm_eval.models.get_model("hf-causal-experimental").create_from_arg_string(
        args.model_args, {"batch_size": 1, "device": "cuda:0"}
    )
    lm.model.generation_config.prefill_chunk_size = None
    lm.model = lm.model.to(torch.bfloat16)

    # Configurazione del tokenizer per evitare warning ripetuti
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    lm.model.generation_config.pad_token_id = lm.tokenizer.pad_token_id
    lm.tokenizer.padding_side = "left"
    
    lm.model.eval()
    
    # Start tracking
    tracker = EmissionsTracker(
        project_name="inference",
        tracking_mode="process",
        log_level="error", 
        measure_power_secs=1,
        save_to_file=False,
        gpu_ids=[0],
    )
    tracker.start()

    def evaluate(
        intput=None,
        temperature=0.3,
        top_p=0.7,
        top_k=20,
        max_new_tokens=512,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
    ):
        # Tokenizza l'input singolo
        inputs = lm.tokenizer(
            intput, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(lm.device)

        with torch.no_grad():
            # Generazione su singolo input
            generation_output = lm.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=True,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                early_stopping=True,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=lm.tokenizer.eos_token_id,
                num_beams=num_beams,
            )

        generated_tokens = generation_output[0][inputs['input_ids'].shape[-1]:]
        result = lm.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield result
        # result = lm.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        # yield result
        
    with gr.Blocks(title="🥷 Mistery ChatBot 🥷") as demo:
        gr.Markdown("## 🥷 Mistery ChatBot 🥷")
        gr.Markdown("Nessuno sa chi sia, ma è qui per aiutarti.")

        # Input / Output
        input_text = gr.Textbox(lines=3, label="Input", placeholder="Inserisci il tuo prompt qui")
        output_text = gr.Textbox(lines=8, label="Output")

        # Opzioni avanzate (inizialmente nascoste)
        temperature = gr.Slider(0, 1, value=0.3, label="Temperature", visible=False)
        top_p = gr.Slider(0, 1, value=0.7, label="Top p", visible=False)
        top_k = gr.Slider(0, 100, step=1, value=20, label="Top k", visible=False)
        max_new_tokens = gr.Slider(1, 2048, step=1, value=512, label="Max new tokens", visible=False)
        num_beams = gr.Slider(1, 10, step=1, value=4, label="Num beams", visible=False)
        no_repeat_ngram_size = gr.Slider(1, 10, step=1, value=3, label="No repeat ngram size", visible=False)
        repetition_penalty = gr.Slider(1.0, 5.0, step=0.1, value=1.5, label="Repetition penalty", visible=False)

        # Bottone principale per generare
        generate_btn = gr.Button("Genera")

        # Bottone discreto per attivare modalità Dev
        toggle_dev = gr.Button("🔧 Dev Mode", elem_id="dev-toggle", visible=False)

        # Callback per mostrare/nascondere gli slider
        def toggle_visibility(visible):
            return [
                gr.update(visible=not visible) for _ in range(7)
            ]

        # Traccia della visibilità corrente
        current_visibility = gr.State(False)

        toggle_dev.click(
            toggle_visibility,
            inputs=[current_visibility],
            outputs=[temperature, top_p, top_k, max_new_tokens, num_beams, no_repeat_ngram_size, repetition_penalty],
            show_progress=False,
        ).then(
            fn=lambda visible: not visible,
            inputs=current_visibility,
            outputs=current_visibility
        )

        # Esegui la generazione
        generate_btn.click(
            fn=evaluate,
            inputs=[
                input_text,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                num_beams,
                no_repeat_ngram_size,
                repetition_penalty,
            ],
            outputs=output_text,
        )

    demo.queue().launch(server_name="0.0.0.0", share=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run batch inference with energy tracking')
    parser.add_argument('--model_args', type=str, 
                        default="pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help='Model arguments string')
    
    args = parser.parse_args()
    main(args)