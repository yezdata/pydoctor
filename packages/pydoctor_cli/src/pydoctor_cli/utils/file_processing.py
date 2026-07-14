from pathlib import Path
from typing import Literal
import libcst as cst
from libcst.metadata import MetadataWrapper
from llama_cpp import Llama
import pathspec
import logging
import os
import queue

from pydoctor_shared_cst.code_extractor import CodeExtractor
from pydoctor_shared_cst.docstring_transformer import DocstringTransformer
from pydoctor_cli.utils.inference import generate_docstring


def find_project_root(target_path: Path) -> Path:
    """
    Find the project root by traversing up the directory structure
        - Stops at the first directory containing .git, .gitignore, or .pydoctor_ignore.
    If none is found, returns the original target_path.
    """
    current = target_path.resolve()

    for parent in [current] + list(current.parents):
        if (
            (parent / ".git").exists()
            or (parent / ".gitignore").exists()
            or (parent / ".pydoctor_ignore").exists()
        ):
            return parent
    return current


def load_ignore_file(project_root: Path, filename: str) -> list[str]:
    ignore_path = project_root / filename

    if ignore_path.is_file():
        try:
            with open(ignore_path, "r", encoding="utf-8") as f:
                return [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        except OSError:
            logging.warning(f"Could not read ignore file: {ignore_path}")
    return []


def get_files_to_process(target_path: Path, default_ignore: set) -> list[Path]:
    if target_path.is_file():
        if target_path.suffix != ".py":
            logging.error(f"File {target_path} is not a Python file.")
            return []
        return [target_path]

    project_root = find_project_root(target_path)

    ignore_lines = []
    ignore_lines.extend(default_ignore)
    ignore_lines.extend(load_ignore_file(project_root, ".gitignore"))
    ignore_lines.extend(load_ignore_file(project_root, ".pydoctor_ignore"))

    spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)

    files_to_process = []

    for root, dirs, files in os.walk(target_path):
        current_dir_path = Path(root).resolve()

        active_dirs = []
        for d in dirs:
            dir_rel_path = str((current_dir_path / d).relative_to(project_root)) + "/"
            if not spec.match_file(dir_rel_path):
                active_dirs.append(d)
        dirs[:] = active_dirs

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = current_dir_path / file
            rel_path = file_path.relative_to(project_root)
            if not spec.match_file(str(rel_path)):
                files_to_process.append(file_path)

    return files_to_process


def parser_producer(
    files_to_process: list[Path],
    extraction_option: Literal["with_docstring", "without_docstring", "all"],
    task_queue: queue.Queue,
) -> None:
    """
    Background thread
    Continuously load and parse LibCST ASTs
    Put results into a queue with backpressure
    """
    for file_path in files_to_process:
        try:
            logging.debug(f"Parsing source code: {file_path}")

            source_code = file_path.read_text(encoding="utf-8")

            cst_tree = cst.parse_module(source_code)
            wrapper = MetadataWrapper(cst_tree)

            extractor = CodeExtractor(extraction_options=extraction_option)
            wrapper.visit(extractor)

            if not extractor.extracted_blocks:
                logging.debug(f"No target blocks extracted from: {file_path}")
                continue

            logging.debug(
                f"Extracted {len(extractor.extracted_blocks)} code blocks from {file_path}"
            )

            task_queue.put((file_path, wrapper, extractor.extracted_blocks), block=True)

        except cst.ParserSyntaxError as e:
            logging.error(f"Syntax error in file {file_path}: {e}")
        except Exception:
            logging.exception(f"Unexpected error when parsing {file_path}")

    task_queue.put((None, None, None))


def inference_consumer(
    llm: Llama, is_dry_run: bool, task_queue: queue.Queue, result_counter: list[int]
) -> None:
    """
    Running in main thread
    Takes tasks from the queue, performs inference, and writes results to files
    """
    transformed_blocks_total = 0

    while True:
        file_path, wrapper, extracted_blocks = task_queue.get(block=True)

        # parser indicate finish
        if file_path is None:
            task_queue.task_done()
            break

        tmp_file_path = file_path.with_suffix(file_path.suffix + ".tmp")
        if tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
            except OSError:
                logging.error(f"Could not remove stale tmp file: {tmp_file_path}")
                task_queue.task_done()
                continue

        try:
            new_docstrings = {}
            for k, v in extracted_blocks.items():
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
                new_docstrings[k] = {"docstring": docstring, "name": node_name}

            transformer = DocstringTransformer(new_docstrings=new_docstrings)
            modified_tree = wrapper.visit(transformer)

            if is_dry_run:
                for key, old_docstring in transformer.old_docstrings.items():
                    docstring_info = new_docstrings.get(key, {})
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

            transformed_blocks_total += transformer.transformed_blocks

        except Exception:
            logging.exception(
                f"Unexpected error processing file in consumer: {file_path}"
            )
            if tmp_file_path.exists():
                try:
                    tmp_file_path.unlink()
                except OSError:
                    logging.error(f"Could not remove stale tmp file: {tmp_file_path}")
        finally:
            task_queue.task_done()

    result_counter[0] = transformed_blocks_total
