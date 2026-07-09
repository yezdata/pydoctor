set -euo pipefail

OUTPUT_DIR="dist"
mkdir -p "$OUTPUT_DIR"

BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

uv venv "$BUILD_DIR/.venv" --isolated --python 3.12

source "$BUILD_DIR/.venv/bin/activate"

uv pip install \
    "Nuitka[onefile]" \
    "libcst" \
    "llama-cpp-python"

PYTHONPATH="packages/pydoctor_cli/src:packages/pydoctor_shared_cst/src" \
python -m nuitka \
    --standalone \
    --onefile \
    --enable-plugin=anti-bloat \
    --output-dir="$OUTPUT_DIR" \
    --report=report.xml \
    packages/pydoctor_cli/src/pydoctor_cli/cli.py