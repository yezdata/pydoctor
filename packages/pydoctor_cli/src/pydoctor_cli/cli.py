import libcst as cst
from libcst.metadata import MetadataWrapper
from llama_cpp import Llama
import logging
import os
import sys
from typing import Literal
from pathlib import Path

from pydoctor_cli.inference import generate_docstring
from pydoctor_cli.model_cache_utils import get_model_path
from pydoctor_cli.logging_utils import setup_logging
from pydoctor_cli.argparse_utils import get_argparser
from pydoctor_shared_cst.docstring_transformer import DocstringTransformer
from pydoctor_shared_cst.code_extractor import CodeExtractor


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

        new_docstrings = {}
        for k, v in extractor.extracted_blocks.items():
            node_name = v.pop("name", "unknown")

            docstring = generate_docstring(llm, v).strip().replace('"""', "'''")
            if not docstring:
                logging.warning(
                    f"Model did not successfully generate docstring for {file_path.name}::{node_name}"
                )
                continue

            logging.debug(
                f"Generated docstring for {file_path.name}::{node_name}:\n{docstring}\n"
            )

            new_docstrings[k] = {"docstring": f'"""{docstring}"""', "name": node_name}

        transformer = DocstringTransformer(new_docstrings=new_docstrings)
        modified_tree = wrapper.visit(transformer)

        if is_dry_run:
            for key, old_docstring in transformer.old_docstrings.items():
                docstring_info = new_docstrings.get(key, "")
                new_docstring = docstring_info.get("docstring", "")
                node_name = docstring_info.get("name", "unknown")
                target_identifier = f"{file_path.name}::{node_name}"

                logging.diff(
                    target_identifier,
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
        sys.exit(1)

    finally:
        if tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
            except OSError:
                logging.error(f"Could not remove stale tmp file: {tmp_file_path}")


def main() -> None:
    parser = get_argparser()
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
