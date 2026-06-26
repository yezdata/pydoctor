import torch
from datasets import load_from_disk
import pandas as pd

from src.utils.config_models import TokenizerConfig
from src.utils.tokenizer import get_finetune_tokenizer

config = TokenizerConfig()

tokenizer = get_finetune_tokenizer(config)

folder = "data/tokenized_finetune"
ds = load_from_disk(folder).to_pandas()


print(f"Dataset length: {len(ds)}")

print(f"Avg input_ids length: {ds['input_ids'].apply(len).mean()}")


print(f"P75: {ds['input_ids'].apply(len).quantile(0.75)}")
print(f"P90: {ds['input_ids'].apply(len).quantile(0.90)}")
print(f"P95: {ds['input_ids'].apply(len).quantile(0.95)}")

max_idx = ds['input_ids'].apply(len).idxmax()

max_input_ids = ds['input_ids'].iloc[max_idx]
max_text = tokenizer.decode(max_input_ids)

print(f"Index of longest sample: {max_idx}")
print(f"Length of longest sample: {len(max_input_ids)}")
print("-" * 20)
print("Content:")
print(max_text)



print("Ensure that correct tokenizer is used for decoding. If the decoded text looks incorrect, check the tokenizer configuration.")