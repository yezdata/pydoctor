import libcst as cst
import onnxruntime as ort
from tokenizers import Tokenizer
import glob
from argparse import ArgumentParser
import logging
import os

from pydoctor_cli.inference import generate_docstring
from pydoctor_shared_cst.docstring_transformer import DocstringTransformer
from pydoctor_shared_cst.code_extractor import CodeExtractor


def main() -> None:
    parser = ArgumentParser(description="Process Python files and add docstrings.")
    parser.add_argument(
        "path", type=str, help="Directory containing Python files to process."
    )

    args = parser.parse_args()

    tokenizer_path = "tokenizers/instruct_finetune/HuggingFaceTB/SmolLM2-1.7B-Instruct/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # TODO: set provider (GPU / CPU)
    onnx_model_dir = "models/onnx/smollm2_1_7b_instruct_merged"
    opts = ort.SessionOptions()
    session = ort.InferenceSession(f"{onnx_model_dir}/model.onnx", opts)

    is_directory = os.path.isdir(args.path)

    if not is_directory:
        transformer = DocstringTransformer()
        try:
            with open(args.path, "r", encoding="utf-8") as f:
                source_code = f.read()

            cst_tree = cst.parse_module(source_code)

            extractor = CodeExtractor(
                extraction_options="all", docstring_token='"""<docstring_place>"""'
            )
            cst_tree.visit(extractor)

            print("Extracted code snippets:")
            for snippet in extractor.extracted_blocks:
                docstring = generate_docstring(session, tokenizer, snippet["code"])
                # print(snippet["code"])
                print("-" * 20)
                print(docstring)

            modified_cst = cst_tree.visit(transformer)

            return
        except Exception:
            logging.exception(f"Error processing file: {args.path}")
            return

    transformer = DocstringTransformer()
    for file in glob.iglob(f"{args.path}/**/*.py", recursive=True):
        with open(file, "r", encoding="utf-8") as f:
            try:
                source_code = f.read()

                cst_tree = cst.parse_module(source_code)

                modified_cst = cst_tree.visit(transformer)

            except Exception:
                logging.exception(f"Error processing file: {file}")
                continue
