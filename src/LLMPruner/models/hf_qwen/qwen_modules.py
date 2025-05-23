# Load model directly
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

# Stampa i nomi dei moduli
for name, module in model.named_modules():
    print(name)