from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from quantize_utils import quantize_tensor, dequantize_tensor

RECENT_WINDOW = 4
MAX_NEW_TOKENS = 30
PROMPT = "Will the interviewer see this? If so, do you think they'll take me?"

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()


def cache_size_mb(cache):
    total_bytes = 0
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024 ** 2)


def theoretical_cache_size_mb(cache, recent_window):
    total_bytes = 0
    for layer in cache.layers:
        k = layer.keys
        batch, heads, seq_len, head_dim = k.shape

        recent_tokens = min(seq_len, recent_window)
        old_tokens = seq_len - recent_tokens

        elements_per_token = batch * heads * head_dim

        recent_bytes = recent_tokens * elements_per_token * 4
        old_bytes = old_tokens * elements_per_token * 1 + 8  # +8 bytes overhead for scale/zero_point

        total_bytes += 2 * (recent_bytes + old_bytes)  # x2 for keys AND values

    return total_bytes / (1024 ** 2)


def run_generation(quantize: bool):
    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]
    generated = input_ids

    start = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        for step in range(MAX_NEW_TOKENS - 1):
            if quantize:
                for layer in cache.layers:
                    k, v = layer.keys, layer.values
                    seq_len = k.shape[2]

                    if seq_len > RECENT_WINDOW:
                        old_k, recent_k = k[:, :, :-RECENT_WINDOW, :], k[:, :, -RECENT_WINDOW:, :]
                        old_v, recent_v = v[:, :, :-RECENT_WINDOW, :], v[:, :, -RECENT_WINDOW:, :]

                        qk, sk, zk = quantize_tensor(old_k, dim=-1)
                        qv, sv, zv = quantize_tensor(old_v, dim=2)
                        old_k_r = dequantize_tensor(qk, sk, zk)
                        old_v_r = dequantize_tensor(qv, sv, zv)

                        layer.keys = torch.cat([old_k_r, recent_k], dim=2)
                        layer.values = torch.cat([old_v_r, recent_v], dim=2)

            out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return {
        "elapsed_sec": elapsed,
        "tokens_per_sec": MAX_NEW_TOKENS / elapsed,
        "text": text,
        "cache": cache,
    }


print("Running full-precision baseline...")
baseline = run_generation(quantize=False)

print("Running quantized version...")
quantized = run_generation(quantize=True)

baseline_cache_mb = cache_size_mb(baseline["cache"])
quantized_cache_mb = theoretical_cache_size_mb(quantized["cache"], RECENT_WINDOW)

print("\n=== COMPARISON ===")
print(f"{'Metric':<24}{'Full Precision':<20}{'Quantized':<20}")
print(f"{'Time (s)':<24}{baseline['elapsed_sec']:<20.3f}{quantized['elapsed_sec']:<20.3f}")
print(f"{'Tokens/sec':<24}{baseline['tokens_per_sec']:<20.2f}{quantized['tokens_per_sec']:<20.2f}")
print(f"{'Cache size (MB)':<24}{baseline_cache_mb:<20.4f}{quantized_cache_mb:<20.4f}")

reduction = (1 - quantized_cache_mb / baseline_cache_mb) * 100
print(f"\nTheoretical cache size reduction: {reduction:.1f}%")

print("\n--- Full precision output ---")
print(baseline["text"])
print("\n--- Quantized output ---")
print(quantized["text"])