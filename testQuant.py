from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from quantize_utils import quantize_tensor, dequantize_tensor, quantize_tensor_symmetric, dequantize_tensor_symmetric

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "Will the interviewer see this? If so, do you think they'll take me?"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs, use_cache=True)

cache = out.past_key_values
real_key = cache.layers[0].keys  # real K tensor, not synthetic

print(f"Real key tensor stats: mean={real_key.mean().item():.4f}, std={real_key.std().item():.4f}")
print(f"Min={real_key.min().item():.4f}, Max={real_key.max().item():.4f}\n")

print("=== Asymmetric, per-channel ===")
q, s, z = quantize_tensor(real_key, dim=-1)
r = dequantize_tensor(q, s, z)
print(f"Mean abs error: {(real_key - r).abs().mean().item():.6f}")

print("\n=== Symmetric, per-channel ===")
q, s = quantize_tensor_symmetric(real_key, dim=-1)
r = dequantize_tensor_symmetric(q, s)
print(f"Mean abs error: {(real_key - r).abs().mean().item():.6f}")