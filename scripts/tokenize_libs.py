import glob
from datasets import Dataset
from transformers import AutoTokenizer
import os

from src.utils.tokenize import tokenize_ds


LIBS_DIR = "data/raw/libs"
TARGET_DIR = "data/tokenized_libs"
MAX_SEQ_LEN = 2048


def py_file_generator(libs_dir):
    for filepath in glob.iglob(f"{libs_dir}/**/*.py", recursive=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                yield {"text": content}
        except Exception:
            continue


tokenizer = AutoTokenizer.from_pretrained(
    "tokenizers/Qwen3-Coder-Next", local_files_only=True
)

ds = Dataset.from_generator(lambda: py_file_generator(LIBS_DIR))
print(len(ds))

final_ds = tokenize_ds(tokenizer, ds, MAX_SEQ_LEN)


final_ds = final_ds.shuffle(seed=100)
final_ds = final_ds.flatten_indices()

split_ds = final_ds.train_test_split(test_size=0.01, seed=42)

data_path = f"{TARGET_DIR}_seq{MAX_SEQ_LEN}"
os.makedirs(data_path, exist_ok=True)

split_ds["train"].save_to_disk(f"{data_path}/train")
split_ds["test"].save_to_disk(f"{data_path}/eval")
