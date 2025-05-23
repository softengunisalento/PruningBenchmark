# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-6B-Chat")
model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-1.5-6B-Chat")

# Stampa i nomi dei moduli
for name, module in model.named_modules():
    print(name)