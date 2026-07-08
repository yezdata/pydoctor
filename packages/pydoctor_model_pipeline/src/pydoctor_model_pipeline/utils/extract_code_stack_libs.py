import glob
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import libcst as cst
from libcst.metadata import MetadataWrapper
from datasets import Dataset, load_dataset
import os
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures

from pydoctor_model_pipeline.utils.preprocessing import (
    download_and_extract_py,
    passes_quality_filter,
)
from pydoctor_shared_cst.code_extractor import CodeExtractor


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

LIBS_DIR = "data/raw/libs"

BATCH_SIZE = 50_000
THE_STACK_SAMPLES = 1_000_000


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
    "spyder-ide/spyder",
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


def parse_code(content: str) -> list[dict]:
    import sys

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(5000)

    try:
        if not content or not content.strip():
            return []

        cst_tree = cst.parse_module(content)
        wrapper = MetadataWrapper(cst_tree)

        extractor = CodeExtractor(
            extraction_options="with_docstring",
        )
        wrapper.visit(extractor)

        return extractor.extracted_blocks.values()
    except (cst.ParserSyntaxError, Exception):
        print("Error parsing code content. Skipping this file.")
        return []
    finally:
        sys.setrecursionlimit(old_limit)


def extract_code(save_path: str) -> None:

    def process_batch_parallel(batch: list[str]) -> list[dict]:
        results = []
        futures = [
            executor.submit(
                parse_code,
                content,
            )
            for content in batch
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=30)
                if result:
                    results.extend(result)
            except concurrent.futures.TimeoutError:
                print("Timeout error while parallel processing")
            except Exception as e:
                print(f"Worker error: {e}")
        return results

    num_workers = 16
    print(f"Workers: {num_workers}")

    if not os.path.exists(LIBS_DIR):
        os.makedirs(LIBS_DIR, exist_ok=True)
        for repo in REPOS:
            download_and_extract_py(repo, LIBS_DIR)

    file_paths = [
        fp
        for fp in glob.iglob(f"{LIBS_DIR}/**/*.py", recursive=True)
        if os.path.isfile(fp)
        and "test" not in os.path.basename(fp).lower()
        and "test" not in fp.lower()
        and "init" not in fp.lower()
    ]

    all_extracted_blocks = []
    batch = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for fp in tqdm(file_paths, total=len(file_paths), desc="Processing libs files"):
            content = read_file(fp)

            if content:
                batch.append(content)

            if len(batch) >= BATCH_SIZE:
                all_extracted_blocks.extend(process_batch_parallel(batch))
                batch.clear()

        if batch:
            all_extracted_blocks.extend(process_batch_parallel(batch))
            batch.clear()

        thestack = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            token=HF_TOKEN,
            split="train",
            streaming=True,
        )
        parsed_samples = 0
        for item in tqdm(
            iter(thestack), total=THE_STACK_SAMPLES, desc="Processing the-stack-dedup"
        ):
            if parsed_samples >= THE_STACK_SAMPLES:
                break

            content = item.get("content", "")

            if not passes_quality_filter(content):
                continue

            batch.append(content)
            parsed_samples += 1

            if len(batch) >= BATCH_SIZE:
                all_extracted_blocks.extend(process_batch_parallel(batch))
                batch.clear()

        if batch:
            all_extracted_blocks.extend(process_batch_parallel(batch))
            batch.clear()

        print(f"Total parsed blocks: {len(all_extracted_blocks)}")

    ds = Dataset.from_list(all_extracted_blocks)
    del all_extracted_blocks

    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
