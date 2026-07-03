from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import psutil
import os
import csv

# Load model and tokenizer
model_name = "gpt2"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Test prompts of varying length
prompts = {
    "short": "Will the interviewer see this? If so, do you think they'll take me?",
    
    "medium": """My cat has been staring at the wall for exactly eleven minutes now and I am 
    starting to genuinely worry that she knows something I don't. She hasn't blinked. She 
    hasn't moved. The wall, for its part, has not done anything remotely interesting in the 
    six years I have lived in this apartment. I am now considering the possibility that my 
    cat has achieved a level of enlightenment that I, a mere human with a LinkedIn profile 
    and unresolved gym anxiety, will never reach. Someone please explain to me""",
    
    "long": """My cat has been staring at the wall for exactly eleven minutes now and I am 
    genuinely starting to worry that she knows something I don't. She hasn't blinked. She 
    hasn't moved. The wall, for its part, has not done anything remotely interesting in the 
    six years I have lived in this apartment.

    This is not the first time this has happened. Last Tuesday she stared at the microwave 
    for four minutes straight, and I still don't know why, because the microwave was off, 
    and nothing was inside it, and yet she stared at it with the intensity of a man who has 
    just remembered he left the stove on. I have started to suspect that cats operate on a 
    frequency that humans simply cannot perceive, possibly related to Wi-Fi signals, possibly 
    related to ghosts, possibly related to something far more mundane that I am simply too 
    unenlightened to understand.

    In the meantime, I have a technical interview in four days, I have solved approximately 
    43 out of 150 problems I am supposed to know cold, and instead of practicing binary trees 
    I am sitting here writing a fake benchmark prompt about my fake cat's spiritual awakening. If 
    anyone from the hiring committee is somehow reading this, please know that I am, in fact, begging you 
    give me this job."""

}

process = psutil.Process(os.getpid())

def get_memory_mb():
    return process.memory_info().rss / (1024 ** 2)

def run_benchmark(prompt_text, label, max_new_tokens=50):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    mem_before = get_memory_mb()
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start_time
    mem_after = get_memory_mb()

    tokens_generated = output.shape[1] - input_len
    tokens_per_sec = tokens_generated / elapsed

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nGenerated text ({label}):")
    print(generated_text)

    print(f"\n--- {label} (input length: {input_len} tokens) ---")
    print(f"Time taken:      {elapsed:.2f} sec")
    print(f"Tokens/sec:      {tokens_per_sec:.2f}")
    print(f"Memory before:   {mem_before:.1f} MB")
    print(f"Memory after:    {mem_after:.1f} MB")
    print(f"Memory delta:    {mem_after - mem_before:.1f} MB")

    return {
        "label": label,
        "input_len": input_len,
        "tokens_generated": tokens_generated,
        "time_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
    }

# Run benchmark across all prompts
results = []
for label, prompt_text in prompts.items():
    result = run_benchmark(prompt_text, label)
    results.append(result)

print("\n=== SUMMARY ===")
print(f"{'Prompt':<10}{'Input':<10}{'Output':<10}{'Time(s)':<10}{'Tok/s':<10}{'RAM(MB)':<10}")
for r in results:
    print(
        f"{r['label']:<10}"
        f"{r['input_len']:<10}"
        f"{r['tokens_generated']:<10}"
        f"{r['time_sec']:<10.2f}"
        f"{r['tokens_per_sec']:<10.2f}"
        f"{r['mem_after_mb']:<10.1f}"
    )

with open("baseline_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to baseline_results.csv")