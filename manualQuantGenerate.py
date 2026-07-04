from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from quantize_utils import quantize_tensor, dequantize_tensor

RECENT_WINDOW = 4

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "Will the interviewer see this? If so, do you think they'll take me?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

max_new_tokens = 30
generated = input_ids

with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    for step in range(max_new_tokens - 1):
        for layer in cache.layers:
            k, v = layer.keys, layer.values
            seq_len = k.shape[2]

            if seq_len > RECENT_WINDOW:
                old_k, recent_k = k[:, :, :-RECENT_WINDOW, :], k[:, :, -RECENT_WINDOW:, :]
                old_v, recent_v = v[:, :, :-RECENT_WINDOW, :], v[:, :, -RECENT_WINDOW:, :]

                qk, sk, zk = quantize_tensor(old_k)
                qv, sv, zv = quantize_tensor(old_v)
                old_k_reconstructed = dequantize_tensor(qk, sk, zk)
                old_v_reconstructed = dequantize_tensor(qv, sv, zv)

                layer.keys = torch.cat([old_k_reconstructed, recent_k], dim=2)
                layer.values = torch.cat([old_v_reconstructed, recent_v], dim=2)
            # else: leave short sequences untouched

        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("Generated text (with simulated quantized cache):")
print(output_text)