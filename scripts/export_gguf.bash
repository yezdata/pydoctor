if [ ! -d "models/smollm2_1_7b_instruct_merged" ]; then
    uv run merge_lora
    deactivate
fi


if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

uv venv --no-workspace --clear --no-config .venv

source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r requirements.txt
uv pip install -U transformers tokenizers

mkdir -p ../models/gguf

python convert_hf_to_gguf.py ../models/smollm2_1_7b_instruct_merged \
  --outfile ../models/gguf/smollm2_1_7b_instruct_merged-q8_0.gguf \
  --outtype q8_0

deactivate
cd ..