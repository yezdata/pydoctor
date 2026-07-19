set -euo pipefail

LORA_MODEL_PATH="models/smollm2_1_7b_instruct_FINAL/epoch_1"
MERGED_PATH="models/smollm2_1_7b_instruct_merged"
GGUF_OUT_DIR="models/gguf"


uv run merge_lora --lora_model_path "$LORA_MODEL_PATH" --merged_model_path "$MERGED_PATH"


if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

uv venv --no-workspace --clear --no-config .venv

source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r requirements.txt
uv pip install -U transformers tokenizers

mkdir -p "../$GGUF_OUT_DIR"

python convert_hf_to_gguf.py ../"$MERGED_PATH" \
  --outfile "../$GGUF_OUT_DIR/smollm2_1_7b_instruct_merged-q8_0.gguf" \
  --outtype q8_0

deactivate
cd ..
