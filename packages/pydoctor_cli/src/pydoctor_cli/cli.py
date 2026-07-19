from llama_cpp import Llama
import logging
import os
import sys
import threading
import queue
from pathlib import Path

from pydoctor_cli.utils.file_processing import (
    get_files_to_process,
    parser_producer,
    inference_consumer,
)
from pydoctor_cli.utils.spinner import CLIProgress
from pydoctor_cli.utils.model_caching import get_model_path
from pydoctor_cli.utils.logging_setup import setup_logging
from pydoctor_cli.utils.args_parser import get_argparser


DEFAULT_IGNORE = {
    ".venv",
    "venv",
    ".egg-info",
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
}


def main() -> None:
    parser = get_argparser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    target_path = Path(args.path).resolve()

    if not target_path.exists():
        logging.error(f"Path {target_path} does not exist.")
        return

    files_to_process = get_files_to_process(target_path, DEFAULT_IGNORE)

    if not files_to_process:
        logging.info("No Python files found to process.")
        return

    REPO_ID = "yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator"
    MODEL_FILE = "smollm2_1_7b_instruct_merged-q8_0.gguf"

    model_name = REPO_ID.split("/")[-1]

    try:
        model_path = get_model_path(REPO_ID, MODEL_FILE)
    except Exception as e:
        logging.critical(
            f"Initialization failed: Could not download model. Details: {e}"
        )
        sys.exit(1)

    logging.debug(f"Loading model: {model_name}...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=(os.cpu_count() or 2) // 2,
        n_gpu_layers=-1,
        seed=42,
        verbose=False,
    )
    logging.debug(f"Model: {model_name} loaded successfully.")

    task_queue = queue.Queue(maxsize=20)
    result_counter = [0]

    producer_thread = threading.Thread(
        target=parser_producer,
        args=(files_to_process, args.extraction_option, task_queue),
        daemon=True,
    )
    producer_thread.start()

    if not args.verbose:
        progress = CLIProgress()
        progress.start()
    else:
        progress = None

    inference_consumer(llm, args.dry_run, task_queue, result_counter, progress)

    if progress:
        progress.stop()

    producer_thread.join()

    transformed_blocks_total = result_counter[0]

    if transformed_blocks_total == 0:
        logging.info("No docstrings to update in any code blocks.")
        return

    if not args.dry_run:
        transformation_map = {
            "with_docstring": "Replaced",
            "without_docstring": "Added new",
            "all": "Processed",
        }
        action = transformation_map.get(args.extraction_option, "Updated")

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
