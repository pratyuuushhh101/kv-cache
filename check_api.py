# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.eval()

# inputs = tokenizer("hello world", return_tensors="pt")
# with torch.no_grad():
#     out = model(**inputs, use_cache=True)

# cache = out.past_key_values
# print("Type:", type(cache))
# print("\nAttributes:")
# print([a for a in dir(cache) if not a.startswith("_")])
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

inputs = tokenizer("hello world", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, use_cache=True)

cache = out.past_key_values
layer0 = cache.layers[0]

print("Type of layer0:", type(layer0))
print("\nAttributes:")
print([a for a in dir(layer0) if not a.startswith("_")])