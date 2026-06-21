from datasets import load_dataset, concatenate_datasets
from huggingface_hub import list_repo_files
from transformers import PreTrainedTokenizerFast
from dotenv import load_dotenv
import os

from src.utils.tokenize import tokenize_ds
from src.utils.preprocessing import passes_quality_filter
from src.utils.config_models import TokenizerConfig

load_dotenv()

MAX_SEQ_LEN = int(os.getenv("PRETRAIN_MAX_SEQ_LEN"))
HF_TOKEN = os.getenv("HF_TOKEN")
pretrain_dir = os.getenv("PRETRAIN_TOKENIZED_DS_DIR")
PRETRAIN_DIR = f"{pretrain_dir}{MAX_SEQ_LEN}"


def main():
    config = TokenizerConfig()  # type: ignore

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        f"tokenizers/{config.name}", local_files_only=True
    )

    # ds1_id = "angie-chen55/python-github-code"
    # all_files_ds1 = list_repo_files(ds1_id, repo_type="dataset")
    # safe_files_ds1 = [
    #     f
    #     for f in all_files_ds1
    #     if f.endswith(".parquet") and f != "data/train-00030-of-00108.parquet"
    # ]
    # ds1 = load_dataset(ds1_id, data_files={"train": safe_files_ds1}, data_dir=None)["train"]

    ds2_id = "codeparrot/codeparrot-clean"
    all_files_ds2 = list_repo_files(ds2_id, repo_type="dataset")
    unsafe_files_ds2 = {
        "file-000000000021.json.gz",
        "file-000000000029.json.gz",
        "file-000000000045.json.gz",
        "file-000000000053.json.gz",
    }
    safe_files_ds2 = [
        f for f in all_files_ds2 if f.endswith(".json.gz") and f not in unsafe_files_ds2
    ]
    ds = load_dataset(ds2_id, data_files={"train": safe_files_ds2}, data_dir=None)[
        "train"
    ]
    # ds3 = load_dataset(
    #     "bigcode/the-stack-dedup", data_dir="data/python", token=HF_TOKEN, split="train"
    # )["train"]

    # ds1 = ds1.select_columns(["code"]).rename_column("code", "text")
    ds = ds.select_columns(["content"]).rename_column("content", "text")
    # ds3 = ds3.select_columns(["content"]).rename_column("content", "text")

    # ds = concatenate_datasets([ds1, ds2, ds3])

    num_workers = min(16, os.cpu_count() or 1)
    ds = ds.filter(
        passes_quality_filter,
        num_proc=num_workers,
    )

    ds = tokenize_ds(
        tokenizer,
        ds,
        MAX_SEQ_LEN,
        packing=True,
    )

    # split_ds = ds.train_test_split(test_size=0.01, seed=42)

    data_path = f"{PRETRAIN_DIR}/big_data"
    os.makedirs(data_path, exist_ok=True)

    ds.save_to_disk(data_path)

    # split_ds["train"].save_to_disk(f"{data_path}/train")
    # split_ds["test"].save_to_disk(f"{data_path}/eval")


if __name__ == "__main__":
    main()
