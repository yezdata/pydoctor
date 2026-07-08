import libcst as cst
from libcst.metadata import MetadataWrapper
from llama_cpp import Llama
from argparse import ArgumentParser
import logging
import os
from typing import Literal
from pathlib import Path

from pydoctor_cli.inference import generate_docstring
from pydoctor_shared_cst.docstring_transformer import DocstringTransformer
from pydoctor_shared_cst.code_extractor import CodeExtractor


def process_single_file(
    file_path: Path,
    llm: Llama,
    extraction_option: Literal["with_docstring", "without_docstring", "all"],
) -> None:
    tmp_file_path = f"{file_path}.tmp"

    if os.path.exists(tmp_file_path):
        try:
            os.remove(tmp_file_path)
        except OSError:
            logging.error(f"Could not remove stale tmp file: {tmp_file_path}")
            return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        cst_tree = cst.parse_module(source_code)
        wrapper = MetadataWrapper(cst_tree)

        extractor = CodeExtractor(extraction_options=extraction_option)
        wrapper.visit(extractor)

        if not extractor.extracted_blocks:
            return

        llm_responses = {}
        for k, v in extractor.extracted_blocks.items():
            docstring = generate_docstring(llm, v).strip().replace('"""', "'''")
            llm_responses[k] = f'"""{docstring}"""'

        transformer = DocstringTransformer(generated_docstrings=llm_responses)
        modified_tree = wrapper.visit(transformer)

        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(modified_tree.code)

        os.replace(tmp_file_path, file_path)

    except cst.ParserSyntaxError as e:
        logging.error(f"Syntax error in file {file_path}: {e}")
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    except Exception as e:
        logging.exception(f"Error processing file: {file_path}, error: {e}")
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


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

    args = parser.parse_args()

    target_path = Path(args.path).resolve()

    files_to_process = []
    if os.path.isfile(args.path):
        if args.path.endswith(".py"):
            files_to_process.append(args.path)
        else:
            logging.error(f"File {args.path} is not a Python file.")
            return
    elif os.path.isdir(args.path):
        files_to_process.extend(target_path.rglob("*.py"))
    else:
        logging.error(f"Path {args.path} does not exist.")
        return

    if not files_to_process:
        logging.info("No Python files found to process.")
        return

    # TODO: set provider (GPU / CPU)
    model_dir = "models/gguf"
    llm = Llama(
        model_path=f"{model_dir}/smollm2_1_7b_instruct_merged-q8_0.gguf",
        n_ctx=2048,
        n_threads=6,
        verbose=False,
    )

    for file in files_to_process:
        process_single_file(file, llm, args.extraction_option)
