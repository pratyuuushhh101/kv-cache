from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "Will the interviewer see this? If so, do you think they'll take me?"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs, use_cache=True)

cache = output.past_key_values
print(f"Number of layers cached: {len(cache.layers)}")

key_tensor = cache.layers[0].keys
value_tensor = cache.layers[0].values

print(f"\nKey tensor shape:   {key_tensor.shape}")
print(f"Value tensor shape: {value_tensor.shape}")
print(f"Key tensor dtype:   {key_tensor.dtype}")

total_elements = 0
for layer in cache.layers:
    total_elements += layer.keys.numel() + layer.values.numel()

bytes_per_element = key_tensor.element_size()
total_bytes = total_elements * bytes_per_element
total_mb = total_bytes / (1024 ** 2)

print(f"\nTotal elements in KV cache: {total_elements:,}")
print(f"Bytes per element: {bytes_per_element}")
print(f"Total KV cache size: {total_mb:.4f} MB")