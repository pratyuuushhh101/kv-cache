from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
from quantize_utils import quantize_tensor, dequantize_tensor

RECENT_WINDOW = 4

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

EVAL_TEXTS = [
    # Passage 1: technology/history
    """The history of computing is marked by a series of transformative innovations, 
    each building upon the last to create increasingly powerful and accessible technology. In the 
    mid-twentieth century, room-sized machines using vacuum tubes performed calculations that would 
    take humans days to complete manually. The invention of the transistor in 1947 set the stage for 
    a dramatic miniaturization of computing hardware, eventually leading to the integrated circuit 
    and, later, the microprocessor.""",

    # Passage 2: nature/science, different style
    """Coral reefs are among the most biodiverse ecosystems on Earth, supporting nearly a quarter 
    of all marine species despite covering less than one percent of the ocean floor. These delicate 
    structures are built over thousands of years by tiny organisms called coral polyps, which secrete 
    calcium carbonate skeletons that gradually accumulate into vast underwater formations. Rising 
    ocean temperatures and acidification, however, have placed immense stress on these ecosystems, 
    leading to widespread bleaching events in recent decades.""",

    # Passage 3: narrative/casual, very different register
    """My grandmother used to tell me that the best stories were the ones nobody believed at first. 
    She would sit on the porch every evening, a cup of tea balanced on her knee, and recount tales 
    of her childhood in a village so small it didn't appear on most maps. I never knew how much of 
    what she said was true and how much was embellished over the years, but it never seemed to matter. 
    What mattered was the way she paused before the good parts, letting the silence do half the work.""",
]


def apply_precision(cache, mode: str):
    if mode == "fp16":
        for layer in cache.layers:
            layer.keys = layer.keys.half().float()
            layer.values = layer.values.half().float()
    elif mode == "int8":
        for layer in cache.layers:
            k, v = layer.keys, layer.values
            cur_len = k.shape[2]
            if cur_len > RECENT_WINDOW:
                old_k, recent_k = k[:, :, :-RECENT_WINDOW, :], k[:, :, -RECENT_WINDOW:, :]
                old_v, recent_v = v[:, :, :-RECENT_WINDOW, :], v[:, :, -RECENT_WINDOW:, :]
                qk, sk, zk = quantize_tensor(old_k, dim=-1)
                qv, sv, zv = quantize_tensor(old_v, dim=2)
                layer.keys = torch.cat([dequantize_tensor(qk, sk, zk), recent_k], dim=2)
                layer.values = torch.cat([dequantize_tensor(qv, sv, zv), recent_v], dim=2)


def compute_perplexity_for_text(text: str, precision_mode: str):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        out = model(input_ids=input_ids[:, 0:1], use_cache=True)
        cache = out.past_key_values

        for pos in range(1, seq_len):
            logits = out.logits[:, -1, :]
            actual_token = input_ids[:, pos]

            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_prob = log_probs[0, actual_token.item()]
            total_nll += -token_log_prob.item()
            total_tokens += 1

            apply_precision(cache, precision_mode)

            out = model(input_ids=input_ids[:, pos:pos+1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values

    return total_nll, total_tokens


def compute_average_perplexity(precision_mode: str):
    all_nll = 0.0
    all_tokens = 0
    per_text_ppl = []

    for text in EVAL_TEXTS:
        nll, tokens = compute_perplexity_for_text(text, precision_mode)
        all_nll += nll
        all_tokens += tokens
        per_text_ppl.append(math.exp(nll / tokens))

    avg_ppl = math.exp(all_nll / all_tokens)
    return avg_ppl, per_text_ppl


print("Computing perplexity across 3 passages, under different cache precisions...\n")

for mode in ["fp32", "fp16", "int8"]:
    avg_ppl, per_text = compute_average_perplexity(mode)
    print(f"{mode.upper()}: overall perplexity = {avg_ppl:.4f}")
    print(f"  Per-passage: {[f'{p:.4f}' for p in per_text]}\n")