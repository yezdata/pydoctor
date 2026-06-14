import io
import os
import zipfile
import requests

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

TARGET_DIR = "data/libs"


def download_and_extract_py(repo):
    url = f"https://api.github.com/repos/{repo}/zipball"
    print(f"Downloading: {repo}")

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(
                f"Error: Failed to download {repo}. Status code: {response.status_code}"
            )
            return

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file_info in z.infolist():
                if file_info.filename.endswith(".py") and not file_info.is_dir():
                    parts = file_info.filename.split("/", 1)
                    relative_path = (
                        parts[1] if len(parts) > max(1, 0) else file_info.filename
                    )

                    dest_path = os.path.join(
                        TARGET_DIR, repo.replace("/", "_"), relative_path
                    )
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    with (
                        z.open(file_info) as source_file,
                        open(dest_path, "wb") as target_file,
                    ):
                        target_file.write(source_file.read())

    except Exception as e:
        print(f"Error: An exception occurred while processing {repo}. Exception: {e}")


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for repo in REPOS:
        download_and_extract_py(repo)


if __name__ == "__main__":
    main()
