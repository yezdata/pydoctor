import sys
from pathlib import Path
from typing import Literal
import libcst as cst
from libcst.metadata import MetadataWrapper
from llama_cpp import Llama
import pathspec
import logging
import os

from pydoctor_shared_cst.code_extractor import CodeExtractor
from pydoctor_shared_cst.docstring_transformer import DocstringTransformer
from pydoctor_cli.inference import generate_docstring


def load_gitignore(target_path: Path) -> pathspec.PathSpec:
    gitignore_path = target_path / ".gitignore"
    if gitignore_path.is_file():
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                return pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
        except OSError:
            pass
    return pathspec.PathSpec.from_lines("gitwildmatch", [])


def get_files_to_process(target_path: Path, default_ignore: set) -> list[Path]:
    if target_path.is_file():
        if target_path.suffix != ".py":
            logging.error(f"File {target_path} is not a Python file.")
            return []
        return [target_path]

    spec = load_gitignore(target_path)

    files_to_process = []
    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if d not in default_ignore]
        for file in files:
            if file in default_ignore or not file.endswith(".py"):
                continue
            file_path = Path(root) / file
            try:
                rel_path = file_path.relative_to(target_path)
                if not spec.match_file(str(rel_path)):
                    files_to_process.append(file_path)
            except ValueError:
                files_to_process.append(file_path)
    return files_to_process


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
