import re
from pathlib import Path
import json
from datasets import Dataset, Features, Value

from src.utils.config_models import TokenizerConfig
from src.utils.tokenizer import get_instruct_tokenizer
from src.utils.tokenize import tokenize_ds

# TODO: zavolat prepare_finetune_data funkci if not os.path.exists

batches_dir = Path("data/synthetic_batches")

results: dict[int, str] = {}
spec_tokens = r"<\|startof(method|class|func)docstring\|>"
list_to_save = []


SYSTEM_PROMPT = f"""You are a professional Python docstring generator.
You will receive Python code block (function, class, class method).
Generate a concise docstring for the code block marked with {config.tokenizer.spec_tokens.docstring_placeholder_token}.
The style of the doctring have to be only sentences summarizing the code block.
Output ONLY the raw docstring text.
Do NOT include Args, Returns, Raises, or any structured sections.
Do NOT output any conversational text or explanations."""


for batch_file in batches_dir.glob("batch_*.json"):
    with open(batch_file, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
        b_idx = batch_data["batch_idx"]
        b_results = batch_data.get("results", {})

        for k, v in b_results.items():
            results[int(k)] = v


def stream_batches(batches_dir, spec_tokens):
    for batch_file in batches_dir.glob("batch_*.json"):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            b_results = batch_data.get("results", {})

            for k, raw_text in b_results.items():
                parts = re.split(spec_tokens, raw_text, maxsplit=1)
                if len(parts) == 3:
                    code, docstring = (
                        parts[0].strip(),
                        parts[2].replace("<|endoftext|>", "").strip(),
                    )
                    final_output = (
                        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                        f"<|im_start|>user\n{code}<|im_end|>\n"
                        f"<|im_start|>assistant\n{docstring}<|im_end|>"
                    )
                    yield {"text": final_output}


features = Features({"text": Value("string")})

final_ds = Dataset.from_generator(
    lambda: stream_batches(batches_dir, spec_tokens),
    features=features,
    keep_in_memory=True,
)

print(final_ds)
print(final_ds[0])

final_ds.save_to_disk("data/raw/synthetic_the_stack_libs_instruct")

tokenizer_config = TokenizerConfig(name="HuggingFaceTB/SmolLM2-1.7B-Instruct")
tokenizer = get_instruct_tokenizer(tokenizer_config)

tokenized_ds = tokenize_ds(
    ds=final_ds, packing=False, tokenizer=tokenizer, num_workers=16, batch_size=256
)

tokenized_ds.save_to_disk(
    "data/tokenized_synthetic_the_stack_libs_instruct_smollm2_1_7B_instruct"
)
