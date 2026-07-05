import libcst as cst
import torch
import glob
from argparse import ArgumentParser
import logging
import os
from optimum.onnxruntime import ORTModelForCausalLM
from tokenizers import Tokenizer

from src.cst.docstring_transformer import DocstringTransformer, CodeExtractor

# TODO: presun inference do /inference.py

SYSTEM_PROMPT = """
You are a professional Python docstring generator
You will receive Python code blocks (functions, methods, or classes)
Generate a concise docstring for the provided code block
The style of the doctring have to be only sentences summarizing the code block
Output ONLY the raw docstring text
Do NOT include Args, Returns, Raises, or any structured sections
Do NOT output any conversational text or explanations.
"""


def get_chat_template(user_prompt: str) -> str:
    final_output = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return final_output


def main():
    parser = ArgumentParser(description="Process Python files and add docstrings.")
    parser.add_argument(
        "path", type=str, help="Directory containing Python files to process."
    )

    args = parser.parse_args()

    tokenizer_path = "tokenizers/instruct_finetune/HuggingFaceTB/SmolLM2-1.7B-Instruct/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    onnx_model_dir = "models/onnx_int8/smollm2_1_7b_instruct_merged"
    model = ORTModelForCausalLM.from_pretrained(
        onnx_model_dir,
        provider="CPUExecutionProvider",
        file_name="model_quantized.onnx",
    )

    is_directory = os.path.isdir(args.path)

    if not is_directory:
        transformer = DocstringTransformer()
        try:
            with open(args.path, "r", encoding="utf-8") as f:
                source_code = f.read()

            cst_tree = cst.parse_module(source_code)

            extractor = CodeExtractor(cst_tree)
            cst_tree.visit(extractor)

            print("Extracted code snippets:")
            for snippet in extractor.extracted_blocks:
                print(snippet)
                print("-" * 20)

                prompt = get_chat_template(snippet)
                encoded = tokenizer.encode(prompt)
                input_ids = [encoded.ids]
                mask = [encoded.attention_mask]

                outputs = model.generate(
                    input_ids=torch.tensor(input_ids),
                    attention_mask=torch.tensor(mask),
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

                prompt_length = len(encoded.ids)
                new_tokens = outputs[0][prompt_length:].tolist()
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                print("Generated docstring:")
                print(generated_text)
                print("-" * 120)

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


if __name__ == "__main__":
    main()
