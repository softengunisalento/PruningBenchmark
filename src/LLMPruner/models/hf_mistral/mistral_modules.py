# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")

# Stampa i nomi dei moduli
for name, module in model.named_modules():
    print(name)