# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)

# Stampa i nomi dei moduli
for name, module in model.named_modules():
    print(name)