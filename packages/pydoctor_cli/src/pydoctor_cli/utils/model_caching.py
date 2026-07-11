from pathlib import Path
import os
import logging
import urllib.request


def get_model_path(repo_id: str, filename: str) -> Path:
    """
    Downloads model into cache directory if not already present and returns the path to the model file.
    """
    if os.name == "nt":
        base_cache = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base_cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))

    cache_dir = base_cache / "pydoctor"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / filename

    if model_path.exists():
        return model_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    logging.info("Downloading pydoctor_model...")

    tmp_model_path = model_path.with_suffix(".download")
    try:
        with (
            urllib.request.urlopen(url) as response,
            open(tmp_model_path, "wb") as out_file,
        ):
            total_size = response.getheader("Content-Length")
            if total_size is not None:
                total_size = int(total_size)

            downloaded = 0
            block_size = 1024 * 1024  # 1MB

            while True:
                block = response.read(block_size)
                if not block:
                    break
                out_file.write(block)
                downloaded += len(block)

                if total_size:
                    percent = downloaded / total_size
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)

                    bar_length = 40
                    filled_length = int(round(bar_length * percent))
                    bar = "█" * filled_length + "-" * (bar_length - filled_length)

                    sys.stdout.write(
                        f"\rDownloading pydoctor_model: |{bar}| {percent:.1%} ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                    )
                    sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.flush()

        tmp_model_path.replace(model_path)
        return model_path

    except Exception:
        sys.stdout.write("\n")
        if tmp_model_path.exists():
            try:
                tmp_model_path.unlink()
            except OSError:
                pass
        raise
