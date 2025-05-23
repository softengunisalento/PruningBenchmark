from transformers import LlamaForCausalLM

# Carica il modello llama
model = LlamaForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf")

# Stampa i nomi dei moduli
for name, module in model.named_modules():
    print(name)