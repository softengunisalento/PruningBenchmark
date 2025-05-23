# 
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")

# Stampa i nomi dei moduli
for name, module in model.named_modules():
    print(name)