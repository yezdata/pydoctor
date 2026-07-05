from pathlib import Path
import json
from datasets import Dataset, Features, Value

from pydoctor_model_pipeline.utils.config_models import MainConfig
from pydoctor_model_pipeline.utils.tokenizer import get_instruct_tokenizer
from pydoctor_model_pipeline.utils.tokenize import tokenize_ds

batches_dir = Path("data/synthetic_batches")

config = MainConfig.from_yaml("configs_kaggle.yaml")

SYSTEM_PROMPT = f"""You are a professional Python docstring generator.
You will receive Python code block (function, class, class method).
Generate a concise docstring for the code block marked with {config.tokenizer.spec_tokens.docstring_placeholder_token}.
The style of the doctring have to be only sentences summarizing the code block.
Output ONLY the raw docstring text.
Do NOT include Args, Returns, Raises, or any structured sections.
Do NOT output any conversational text or explanations."""


def stream_batches(batches_dir):
    for batch_file in batches_dir.glob("batch_*.json"):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            b_results = batch_data.get("results", {})

            for pair in b_results.values():
                final_output = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{pair['code']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{pair['docstring']}<|im_end|>"
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
