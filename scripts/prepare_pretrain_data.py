from datasets import load_dataset, concatenate_datasets
from huggingface_hub import list_repo_files
import os

from src.utils.tokenizer import get_pretrain_tokenizer
from src.utils.tokenize import tokenize_ds
from src.utils.preprocessing import passes_quality_filter
from src.utils.config_models import TokenizerConfig, PretrainConfig


def main():
    tokenizer_config = TokenizerConfig()  # type: ignore

    tokenizer = get_pretrain_tokenizer(tokenizer_config)

    train_config = PretrainConfig()  # type: ignore
    PRETRAIN_DIR = f"{train_config.tokenized_ds_dir}{train_config.max_seq_len}"

    ds1_id = "angie-chen55/python-github-code"
    all_files_ds1 = list_repo_files(ds1_id, repo_type="dataset")
    safe_files_ds1 = [
        f
        for f in all_files_ds1
        if f.endswith(".parquet") and f != "data/train-00030-of-00108.parquet"
    ]
    ds1 = load_dataset(ds1_id, data_files={"train": safe_files_ds1}, data_dir=None)[
        "train"
    ]

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
    ds2 = load_dataset(ds2_id, data_files={"train": safe_files_ds2}, data_dir=None)[
        "train"
    ]

    ds1 = ds1.select_columns(["code"]).rename_column("code", "text")
    ds2 = ds2.select_columns(["content"]).rename_column("content", "text")

    ds = concatenate_datasets([ds1, ds2])

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

    data_path = f"{PRETRAIN_DIR}/big_data"
    os.makedirs(data_path, exist_ok=True)

    ds.save_to_disk(data_path)


if __name__ == "__main__":
    main()
