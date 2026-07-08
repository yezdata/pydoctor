from pathlib import Path
import json
from datasets import Dataset, Features, Value

from pydoctor_model_pipeline.utils.config_models import MainConfig
from pydoctor_model_pipeline.utils.tokenizer import get_instruct_tokenizer
from pydoctor_model_pipeline.utils.tokenize import tokenize_ds

batches_dir = Path("data/synthetic_batches")

config = MainConfig.from_yaml("configs_kaggle.yaml")

SYSTEM_PROMPT = '''You are a professional Python docstring generator.
Analyze the code in 'TARGET CODE'. Use 'CONTEXT' only for reference regarding class attributes and structure.
Generate a concise, single-paragraph docstring summarizing ONLY the 'TARGET CODE'.
The docstring must consist only of descriptive sentences.

CRITICAL RULES:
1. Output ONLY the raw docstring text. Do not include triple quotes (""").
2. Do not include structured sections like Args, Returns, or Raises.
3. Do not include conversational filler, explanations, or markdown code blocks.
4. Focus strictly on the functionality of the 'TARGET CODE'.'''


def stream_batches(batches_dir):
    for batch_file in batches_dir.glob("batch_*.json"):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            b_results = batch_data.get("results", {})

            for sample in b_results.values():
                final_output = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\nCONTEXT\n{sample['context']}\n\nTARGET CODE\n{sample['target']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{sample['docstring']}<|im_end|>"
                )
                yield {"text": final_output}


def main() -> None:
    features = Features({"text": Value("string")})

    final_ds = Dataset.from_generator(
        lambda: stream_batches(batches_dir),
        features=features,
        keep_in_memory=False,
    )

    print(final_ds[0])

    tokenizer = get_instruct_tokenizer(config.tokenizer)

    tokenized_ds = tokenize_ds(
        ds=final_ds, packing=False, tokenizer=tokenizer, num_workers=16, batch_size=512
    )

    tokenized_ds.save_to_disk(
        "data/finetune/tokenized_synthetic_the_stack_libs_instruct_smollm2_1_7B_instruct"
    )
