import os
from src.utils.libs_data import download_and_extract_py

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

TARGET_DIR = "data/raw/libs"


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for repo in REPOS:
        download_and_extract_py(repo)


if __name__ == "__main__":
    main()
