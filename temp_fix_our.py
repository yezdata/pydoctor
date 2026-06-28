from pathlib import Path
import json
from datasets import Dataset, Features, Value
from src.utils.tokenize import tokenize_ds
from src.utils.config_models import TokenizerConfig
from src.utils.tokenizer import get_finetune_tokenizer

batches_dir = Path("data/synthetic_batches")

results: dict[int, str] = {}
list_to_save = []

for batch_file in batches_dir.glob("batch_*.json"):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            b_idx = batch_data["batch_idx"]
            b_results = batch_data.get("results", {})
            
            for k, v in b_results.items():
                results[int(k)] = v

def stream_batches(batches_dir):
    for batch_file in batches_dir.glob("batch_*.json"):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            b_results = batch_data.get("results", {})
            
            for k, raw_text in b_results.items():
               yield {"text": raw_text}

features = Features({
    "text": Value("string")
})

final_ds = Dataset.from_generator(
    lambda: stream_batches(batches_dir),
    features=features,
    keep_in_memory=True
)

print(final_ds)
print(final_ds[0])

final_ds.save_to_disk("data/raw/synthetic__our")


tokenizer_config = TokenizerConfig(name="bigcode/starcoder2-15b")
tokenizer = get_finetune_tokenizer(tokenizer_config)


tokenized_ds = tokenize_ds(ds=final_ds, packing=False, tokenizer=tokenizer, num_workers=16, batch_size=256)

tokenized_ds.save_to_disk("data/finetune/tokenized_synthetic_the_stack_libs_starcoder2")