from datasets import load_dataset
from huggingface_hub import list_repo_files
import os

from src.utils.tokenizer import get_pretrain_tokenizer
from src.utils.tokenize import tokenize_ds
from src.utils.preprocessing import passes_quality_filter
from src.utils.config_models import MainConfig


def main() -> None:
    config = MainConfig.from_yaml("configs.yaml")

    tokenizer = get_pretrain_tokenizer(config.tokenizer)

    PRETRAIN_DIR = f"{config.pretrain.tokenized_ds_dir}{config.pretrain.max_seq_len}"

    ds_id = "codeparrot/codeparrot-clean"
    all_files_ds = list_repo_files(ds_id, repo_type="dataset")
    unsafe_files_ds = {
        "file-000000000021.json.gz",
        "file-000000000029.json.gz",
        "file-000000000045.json.gz",
        "file-000000000053.json.gz",
    }
    safe_files_ds = [
        f for f in all_files_ds if f.endswith(".json.gz") and f not in unsafe_files_ds
    ]
    ds = load_dataset(ds_id, data_files={"train": safe_files_ds}, data_dir=None)[
        "train"
    ]
    ds = ds.select_columns(["content"]).rename_column("content", "text")

    num_workers = min(16, os.cpu_count() or 1)
    ds = ds.filter(
        passes_quality_filter,
        num_proc=num_workers,
    )

    ds = tokenize_ds(
        tokenizer,
        ds,
        config.pretrain.max_seq_len,
        packing=True,
    )

    os.makedirs(PRETRAIN_DIR, exist_ok=True)

    ds.save_to_disk(PRETRAIN_DIR)


if __name__ == "__main__":
    main()
