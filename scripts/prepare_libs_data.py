import glob
import libcst as cst
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import os
from tqdm import tqdm
import concurrent.futures

from src.utils.libs_data import download_and_extract_py
from src.utils.tokenize import tokenize_ds
from src.utils.config_models import TokenizerConfig
from src.cst.code_extractor import CodeExtractor


LIBS_DIR = "data/raw/libs"
PRETRAIN_DIR = "data/tokenized_pretrain"
FINETUNE_DIR = "data/tokenized_finetune"
MAX_SEQ_LEN = 2048


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
    content = read_file(filepath)
    if not content or not content.strip():
        return []
    try:
        cst_tree = cst.parse_module(content)
        extractor = CodeExtractor(cst_tree)
        cst_tree.visit(extractor)
        return extractor.extracted_pairs
    except (cst.ParserSyntaxError, Exception):
        return []


config = TokenizerConfig()  # type: ignore

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    f"tokenizers/{config.name}", local_files_only=True
)

if not os.path.exists(LIBS_DIR):
    os.makedirs(LIBS_DIR, exist_ok=True)
    for repo in REPOS:
        download_and_extract_py(repo, LIBS_DIR)


file_paths = [
    fp for fp in glob.iglob(f"{LIBS_DIR}/**/*.py", recursive=True) if os.path.isfile(fp)
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

pretrain_ds_tokenized = tokenize_ds(tokenizer, pretrain_ds, MAX_SEQ_LEN, packing=True)
del pretrain_ds

data_path = f"{PRETRAIN_DIR}_seq{MAX_SEQ_LEN}"
os.makedirs(data_path, exist_ok=True)
pretrain_ds_tokenized.save_to_disk(data_path)
del pretrain_ds_tokenized


# FINETUNE DATASET
all_extracted_pairs = []

max_workers = os.cpu_count()
print(f"Workers: {max_workers}")

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(
        tqdm(
            executor.map(process_file, file_paths, chunksize=50),
            total=len(file_paths),
        )
    )

for file_results in results:
    if file_results:
        all_extracted_pairs.extend(file_results)

print(f"Total parsed pairs: {len(all_extracted_pairs)}")
finetune_ds = Dataset.from_list(all_extracted_pairs)
del all_extracted_pairs

finetune_ds_tokenized = tokenize_ds(tokenizer, finetune_ds, MAX_SEQ_LEN, packing=False)
del finetune_ds

os.makedirs(FINETUNE_DIR, exist_ok=True)
finetune_ds_tokenized.save_to_disk(FINETUNE_DIR)
