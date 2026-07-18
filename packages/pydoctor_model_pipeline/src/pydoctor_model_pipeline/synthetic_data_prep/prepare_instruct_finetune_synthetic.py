from pathlib import Path
import json
from datasets import Dataset, Features, Value

from pydoctor_model_pipeline.utils.config_models import MainConfig
from pydoctor_model_pipeline.utils.tokenizer import get_instruct_tokenizer
from pydoctor_model_pipeline.utils.tokenize import tokenize_ds

batches_dir = Path("data/synthetic_batches")

config = MainConfig.from_yaml("configs_kaggle.yaml")

SYSTEM_PROMPT = '''You are a professional Python documentation expert.
Analyze the 'TARGET CODE' and using also the 'CONTEXT' provide a concise docstring containing a high-level summary of its purpose and logic.
Scope Definitions:
1. 'CONTEXT' is the surrounding code that provides additional information about the 'TARGET CODE'.
2. If 'TARGET CODE' is a class, the 'CONTEXT' are signatures of the class methods, if 'TARGET CODE' is a method, the 'CONTEXT' is the constructor of its class.

CRITICAL RULES:
1. Output ONLY the raw docstring text. Do NOT output any other conversation filler. Do not include triple quotes (""").
2. Describe the semantic purpose and architectural role of the TARGET CODE, not its internal state or attributes.
3. Do not include structured sections like Args, Returns, or Raises.
4. Focus on what the TARGET CODE achieves, do NOT describe its implementation details.
5. Do not list initialization variables or internal data structures.
'''

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def stream_batches(batches_dir):
    for batch_file in batches_dir.glob("batch_*.json"):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            b_results = batch_data.get("results", {})

            for sample in b_results.values():
                prompt = (
                    f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
                    f"{IM_START}user\nCONTEXT\n{sample['context']}\n\nTARGET CODE\n{sample['target']}{IM_END}\n"
                    f"{IM_START}assistant\n"
                )
                completion = f"{sample['docstring']}{IM_END}"

                yield {"prompt": prompt, "completion": completion}


def main() -> None:
    features = Features({"prompt": Value("string"), "completion": Value("string")})

    raw_ds = Dataset.from_generator(
        lambda: stream_batches(batches_dir),
        features=features,
        keep_in_memory=False,
    )

    print(raw_ds[0])

    tokenizer = get_instruct_tokenizer(config.tokenizer)

    tokenized_ds = tokenize_ds(
        ds=raw_ds, packing=False, tokenizer=tokenizer, num_workers=16, batch_size=512
    )

    tokenized_ds.save_to_disk(
        "data/finetune/tokenized_synthetic_codeparrot_instruct_smollm2_1_7B_instruct"
    )
