# /// script
# dependencies = [
#  "transformers==4.53.3",
# "optimum-onnx[onnxruntime]>=0.1.0",
# ]
# ///


import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import subprocess

base = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.float32
)
model = PeftModel.from_pretrained(
    base, "models/smollm2_1_7b_instruct/finetune_instruct/epoch_3"
)
model = model.merge_and_unload()
model.save_pretrained("models/smollm2_1_7b_instruct_merged")


onnx_fp32_path = "models/onnx/smollm2_1_7b_instruct_merged"
subprocess.run(
    [
        "uv",
        "run",
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        "models/smollm2_1_7b_instruct_merged",
        "--task",
        "text-generation-with-past",
        onnx_fp32_path,
        "--no-post-process",
    ]
)


onnx_int8_path = "models/onnx_int8/smollm2_1_7b_instruct_merged"
subprocess.run(
    [
        "uv",
        "run",
        "optimum-cli",
        "onnxruntime",
        "quantize",
        "--onnx_model",
        onnx_fp32_path,
        "-o",
        onnx_int8_path,
        "--arm64",
    ]
)
