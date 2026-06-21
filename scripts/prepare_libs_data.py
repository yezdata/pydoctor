import glob
import libcst as cst
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import os
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures

from src.utils.preprocessing import download_and_extract_py
from src.utils.tokenize import tokenize_ds
from src.utils.config_models import TokenizerConfig
from src.cst.code_extractor import CodeExtractor

load_dotenv()

MAX_SEQ_LEN = int(os.getenv("PRETRAIN_MAX_SEQ_LEN"))
pretrain_dir = os.getenv("PRETRAIN_TOKENIZED_DS_DIR")
PRETRAIN_DIR = f"{pretrain_dir}{MAX_SEQ_LEN}"


LIBS_DIR = "data/raw/libs"
FINETUNE_DIR = "data/tokenized_finetune"


REPOS = [
    "python/cpython",
    "pytorch/pytorch",
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "numpy/numpy",
    "huggingface/transformers",
    "django/django",
    "matplotlib/matplotlib",
    "jax-ml/jax",
    "home-assistant/core",
    "scipy/scipy",
    "pallets/flask",
    "tiangolo/fastapi",
    "psf/requests",
    "encode/httpx",
]


def read_file(filepath: str) -> str | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def process_file(filepath: str) -> list[dict]:
    import sys

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(5000)

    try:
        content = read_file(filepath)
        if not content or not content.strip():
            return []

        cst_tree = cst.parse_module(content)
        extractor = CodeExtractor(cst_tree)
        cst_tree.visit(extractor)
        return extractor.extracted_pairs
    except (cst.ParserSyntaxError, Exception) as e:
        print(f"Error processing {filepath}: {e}")
        return []
    finally:
        sys.setrecursionlimit(old_limit)


def main():
    num_workers = min(16, os.cpu_count() or 1)
    print(f"Workers: {num_workers}")

    config = TokenizerConfig()  # type: ignore

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        f"tokenizers/{config.name}", local_files_only=True
    )

    if not os.path.exists(LIBS_DIR):
        os.makedirs(LIBS_DIR, exist_ok=True)
        for repo in REPOS:
            download_and_extract_py(repo, LIBS_DIR)

    file_paths = [
        fp
        for fp in glob.iglob(f"{LIBS_DIR}/**/*.py", recursive=True)
        if os.path.isfile(fp)
    ]

    # PRETRAIN DATASET
    pretrain_list = []
    for path in file_paths:
        txt = read_file(path)
        if txt:
            pretrain_list.append({"text": txt})

    pretrain_ds = Dataset.from_list(pretrain_list)
    del pretrain_list
    print(f"Pretrain data len: {len(pretrain_ds)}")

    pretrain_ds_tokenized = tokenize_ds(
        tokenizer,
        pretrain_ds,
        MAX_SEQ_LEN,
        tokenizer.eos_token_id,
        packing=True,
        num_workers=num_workers,
        batch_size=5000,
    )

    split_ds = pretrain_ds_tokenized.train_test_split(test_size=0.01, seed=42)

    data_path = f"{PRETRAIN_DIR}/libs"
    os.makedirs(data_path, exist_ok=True)
    split_ds["train"].save_to_disk(f"{data_path}/train")
    split_ds["test"].save_to_disk(f"{data_path}/eval")

    # FINETUNE DATASET
    # TODO CODESEARCHNET integrace do FINETUNE preparation
    all_extracted_pairs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in file_paths}
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(file_paths)
        ):
            try:
                result = future.result(timeout=30)
                if result:
                    all_extracted_pairs.extend(result)
            except concurrent.futures.TimeoutError:
                print(f"Timeout: {futures[future]}")
            except Exception as e:
                print(f"Worker error: {e}")

    print(f"Total parsed pairs: {len(all_extracted_pairs)}")
    finetune_ds = Dataset.from_list(all_extracted_pairs)
    del all_extracted_pairs

    finetune_ds_tokenized = tokenize_ds(
        tokenizer,
        finetune_ds,
        MAX_SEQ_LEN,
        packing=False,
        num_workers=num_workers,
        batch_size=5000,
    )

    # TODO: train/eval split
    os.makedirs(FINETUNE_DIR, exist_ok=True)
    finetune_ds_tokenized.save_to_disk(FINETUNE_DIR)


if __name__ == "__main__":
    main()
