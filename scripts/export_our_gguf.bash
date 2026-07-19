set -euo pipefail


BASE_MODEL_PATH="models/v3/finetune"
LLAMA_MODEL_PATH="models/pydoctor_model_hf_llama"
GGUF_OUT_DIR="models/gguf"


uv run create_hf_llama --base_model_path "$BASE_MODEL_PATH" \
  --output_dir "$LLAMA_MODEL_PATH"


if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

uv venv --no-workspace --clear --no-config .venv

source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r requirements.txt
uv pip install -U transformers tokenizers

mkdir -p "../$GGUF_OUT_DIR"

python convert_hf_to_gguf.py ../"$LLAMA_MODEL_PATH" \
  --outfile "../$GGUF_OUT_DIR/pydoctor_model-q8_0.gguf" \
  --outtype q8_0

deactivate
cd ..
