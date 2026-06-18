import requests
import zipfile
import io
import numpy as np
import os


def compute_line_stats(code: str) -> dict:
    """Compute quality metrics for a code file."""
    lines = code.strip().split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    if not non_empty_lines:
        return {
            "avg_line_len": 0,
            "alpha_ratio": 0,
            "unique_ratio": 0,
            "num_lines": 0,
        }

    avg_line_len = np.mean([len(l) for l in non_empty_lines])
    all_chars = "".join(non_empty_lines)
    alpha_ratio = sum(c.isalpha() for c in all_chars) / max(len(all_chars), 1)
    unique_ratio = len(set(non_empty_lines)) / len(non_empty_lines)

    return {
        "avg_line_len": avg_line_len,
        "alpha_ratio": alpha_ratio,
        "unique_ratio": unique_ratio,
        "num_lines": len(non_empty_lines),
    }


def passes_quality_filter(code: str | dict) -> bool:
    """
    Return (passes, reason). Applies basic code quality heuristics.
    """
    if isinstance(code, dict):
        code = code["text"]
    stats = compute_line_stats(code)

    # Too few lines
    if stats["num_lines"] < 3:
        return False

    # Likely minified (very long lines)
    if stats["avg_line_len"] > 150:
        return False

    # Very low alphabetic ratio (likely binary or machine-generated)
    if stats["alpha_ratio"] < 0.15:
        return False

    # Very low unique line ratio (repetitive boilerplate)
    if stats["unique_ratio"] < 0.1:
        return False

    return True


def download_and_extract_py(repo: str, save_dir: str):
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
                        save_dir, repo.replace("/", "_"), relative_path
                    )
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    with z.open(file_info) as source_file:
                        raw_data = source_file.read()

                        try:
                            code_text = raw_data.decode("utf-8")
                        except Exception as e:
                            print(
                                f"Skipping {file_info.filename} - decoding failed: {e}"
                            )
                            continue

                        ok = passes_quality_filter(code_text)
                        if not ok:
                            continue

                        with open(dest_path, "wb") as target_file:
                            target_file.write(raw_data)

    except Exception as e:
        print(f"Error: An exception occurred while processing {repo}. Exception: {e}")
