from pathlib import Path
import json
from datasets import Dataset, Features, Value

from pydoctor_model_pipeline.utils.tokenize import tokenize_ds
from pydoctor_model_pipeline.utils.config_models import MainConfig
from pydoctor_model_pipeline.utils.tokenizer import get_finetune_tokenizer

batches_dir = Path("data/synthetic_batches")


config = MainConfig.from_yaml("configs.yaml")

# results: dict[int, str] = {}
# list_to_save = []

# for batch_file in batches_dir.glob("batch_*.json"):
#     with open(batch_file, "r", encoding="utf-8") as f:
#         batch_data = json.load(f)
#         b_idx = batch_data["batch_idx"]
#         b_results = batch_data.get("results", {})

#         for k, v in b_results.items():
#             results[int(k)] = v


def stream_batches(batches_dir):
    for batch_file in batches_dir.glob("batch_*.json"):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            b_results = batch_data.get("results", {})

            for sample in b_results.values():
                final_output = f"{sample['target']}\n{config.tokenizer.spec_tokens.docstring_start_token}{sample['docstring']}{config.tokenizer.eos_token}"

                yield {"text": final_output}


def main() -> None:
    features = Features({"text": Value("string")})

    final_ds = Dataset.from_generator(
        lambda: stream_batches(batches_dir), features=features, keep_in_memory=False
    )

    print(final_ds.to_pandas().head())

    tokenizer = get_finetune_tokenizer(config.tokenizer)

    tokenized_ds = tokenize_ds(
        ds=final_ds, packing=False, tokenizer=tokenizer, num_workers=16, batch_size=256
    )

    tokenized_ds.save_to_disk(
        "data/finetune/tokenized_synthetic_the_stack_libs_starcoder2"
    )
