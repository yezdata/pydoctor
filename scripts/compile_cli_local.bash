set -euo pipefail


OUTPUT_DIR="dist"
mkdir -p "$OUTPUT_DIR"

BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

uv venv "$BUILD_DIR/.venv" --no-config --python 3.12
source "$BUILD_DIR/.venv/bin/activate"

uv export --package pydoctor-cli --no-hashes --no-editable --output-file "$BUILD_DIR/requirements.txt"


uv pip install "Nuitka[onefile]"

if [[ "$OSTYPE" == "darwin"* ]]; then
    export CMAKE_ARGS="-DGGML_METAL=on"
    uv pip install --no-binary llama-cpp-python -r "$BUILD_DIR/requirements.txt"

else
    uv pip install -r "$BUILD_DIR/requirements.txt" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
fi

uv pip install --no-deps ./packages/pydoctor_shared_cst
uv pip install --no-deps ./packages/pydoctor_cli


python -m nuitka \
    --standalone \
    --onefile \
    --enable-plugin=anti-bloat \
    --output-dir="$OUTPUT_DIR" \
    --report=report.xml \
    packages/pydoctor_cli/src/pydoctor_cli/cli.py
