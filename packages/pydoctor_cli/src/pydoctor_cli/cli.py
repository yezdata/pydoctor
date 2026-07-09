import libcst as cst
from libcst.metadata import MetadataWrapper
from llama_cpp import Llama
from argparse import ArgumentParser
import logging
import os
import sys
from typing import Literal
from pathlib import Path
import urllib.request

from pydoctor_cli.inference import generate_docstring
from pydoctor_cli.logging_utils import setup_logging
from pydoctor_shared_cst.docstring_transformer import DocstringTransformer
from pydoctor_shared_cst.code_extractor import CodeExtractor


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


def process_single_file(
    file_path: Path,
    llm: Llama,
    extraction_option: Literal["with_docstring", "without_docstring", "all"],
    is_dry_run: bool,
) -> int:
    tmp_file_path = file_path.with_suffix(file_path.suffix + ".tmp")

    if tmp_file_path.exists():
        try:
            tmp_file_path.unlink()
        except OSError:
            logging.error(f"Could not remove stale tmp file: {tmp_file_path}")
            return 0

    try:
        logging.debug(f"Parsing source code from: {file_path}")

        source_code = file_path.read_text(encoding="utf-8")

        cst_tree = cst.parse_module(source_code)
        wrapper = MetadataWrapper(cst_tree)

        extractor = CodeExtractor(extraction_options=extraction_option)
        wrapper.visit(extractor)

        if not extractor.extracted_blocks:
            return 0

        extracted_blocks_count = len(extractor.extracted_blocks)

        logging.debug(
            f"Extracted {extracted_blocks_count} code blocks from {file_path}"
        )

        llm_responses = {}
        for k, v in extractor.extracted_blocks.items():
            docstring = generate_docstring(llm, v).strip().replace('"""', "'''")
            logging.debug(
                f"Generated docstring in {file_path} at {k.split('_')[0]}:\n{docstring}\n"
            )

            llm_responses[k] = f'"""{docstring}"""'

        transformer = DocstringTransformer(generated_docstrings=llm_responses)
        modified_tree = wrapper.visit(transformer)

        if is_dry_run:
            for key, old_docstring in transformer.old_docstrings.items():
                new_docstring = llm_responses.get(key, "")
                logging.diff(
                    str(file_path.name),
                    old_docstring,
                    new_docstring,
                )
        else:
            tmp_file_path.write_text(modified_tree.code, encoding="utf-8")

            tmp_file_path.replace(file_path)

        return transformer.transformed_blocks

    except cst.ParserSyntaxError as e:
        logging.error(f"Syntax error in file {file_path}: {e}")
        return 0
    except Exception:
        logging.exception(f"Unexpected error processing file: {file_path}")
        return 0

    finally:
        if tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
            except OSError:
                logging.error(f"Could not remove stale tmp file: {tmp_file_path}")


def main() -> None:
    parser = ArgumentParser(description="Process Python files and add docstrings.")
    parser.add_argument(
        "path",
        type=str,
        help="Python file or directory containing Python files to process.",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--replace",
        action="store_const",
        dest="extraction_option",
        const="with_docstring",
        help="Only replace existing docstrings.",
    )
    group.add_argument(
        "--add",
        action="store_const",
        dest="extraction_option",
        const="without_docstring",
        help="Only add new docstrings (default).",
    )
    group.add_argument(
        "--all",
        action="store_const",
        dest="extraction_option",
        const="all",
        help="Process all functions / classes.",
    )

    parser.set_defaults(extraction_option="without_docstring")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without modifying any files and show changes (default: False)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Turn on verbose output (default: False)",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    target_path = Path(args.path).resolve()

    files_to_process = []
    if target_path.is_file():
        if target_path.suffix == ".py":
            files_to_process.append(target_path)
        else:
            logging.error(f"File {target_path} is not a Python file.")
            return
    elif target_path.is_dir():
        files_to_process.extend(target_path.rglob("*.py"))
    else:
        logging.error(f"Path {target_path} does not exist.")
        return

    if not files_to_process:
        logging.info("No Python files found to process.")
        return

    # TODO: set provider (GPU / CPU)
    REPO_ID = "yezdata/pydoctor_model"
    MODEL_FILE = "smollm2_1_7b_instruct_merged-q8_0.gguf"

    try:
        model_path = get_model_path(REPO_ID, MODEL_FILE)
    except Exception as e:
        logging.critical(
            f"Initialization failed: Could not download model. Details: {e}"
        )
        sys.exit(1)

    logging.debug("Loading pydoctor_model...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=(os.cpu_count() or 2) // 2,
        n_gpu_layers=-1,
        seed=42,
        verbose=False,
    )
    logging.debug("pydoctor_model loaded successfully.")

    transformed_blocks_total = 0
    for file in files_to_process:
        transformed_blocks = process_single_file(
            file, llm, args.extraction_option, args.dry_run
        )
        transformed_blocks_total += transformed_blocks

    transformation_map = {
        "with_docstring": "Replaced",
        "without_docstring": "Added new",
        "all": "Processed",
    }
    action = transformation_map.get(args.extraction_option, "Updated")

    if transformed_blocks_total == 0:
        logging.info("No docstrings to update in any code blocks.")
        return
    if len(files_to_process) > 1:
        final_msg = (
            f"{action} docstrings in {transformed_blocks_total} code blocks "
            f"across {len(files_to_process)} source files."
        )
    else:
        single_file = files_to_process[0]
        try:
            relative_path = single_file.relative_to(Path.cwd())
        except ValueError:
            # use absolute path if the file is not in the cwd
            relative_path = single_file
        final_msg = f"{action} docstrings in {transformed_blocks_total} code blocks in {relative_path}."

    logging.success(final_msg)


if __name__ == "__main__":
    main()
