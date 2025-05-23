# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-small-8k-instruct", trust_remote_code=True)

# Stampa i nomi dei moduli
for name, module in model.named_modules():
    print(name)