#!/bin/sh
set -e

REPO="yezdata/pydoctor"
MODEL_REPO="yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator"
MODEL_FILE="smollm2_1_7b_instruct_merged-q8_0.gguf"

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Darwin" ]; then
    PLATFORM="macos"
    if [ "$ARCH" = "arm64" ]; then
        BINARY="pydoctor-macos-arm64"
    else
        echo "Error: Only M-Series Macs - Apple Silicon (arm64) is supported."
        exit 1
    fi
elif [ "$OS" = "Linux" ]; then
    PLATFORM="linux"
    if [ "$ARCH" = "x86_64" ]; then
        BINARY="pydoctor-linux-x86_64"
    else
        echo "Error: Only x86_64 is supported on Linux."
        exit 1
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

INSTALL_DIR="$HOME/.local/bin"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/pydoctor"

mkdir -p "$INSTALL_DIR"
mkdir -p "$CACHE_DIR"

echo "Downloading PyDoctor CLI for $OS ($ARCH)..."
URL="https://github.com/${REPO}/releases/latest/download/${BINARY}"
curl -L "$URL" -o "$INSTALL_DIR/pydoctor"

chmod +x "$INSTALL_DIR/pydoctor"

if [ "$OS" = "Darwin" ]; then
    xattr -d com.apple.quarantine "$INSTALL_DIR/pydoctor" 2>/dev/null || true
fi

MODEL_PATH="$CACHE_DIR/$MODEL_FILE"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading PyDoctor model..."
    MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"

    curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PATH.download"
    mv "$MODEL_PATH.download" "$MODEL_PATH"
    echo "Model successfully cached."
fi

echo "--------------------------------------------------------"
echo "PyDoctor successfully installed to: $INSTALL_DIR/pydoctor"

# checks if install_dir is in the path
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    echo "WARNING: $INSTALL_DIR is not in your PATH."
    echo "Please add it to your profile (e.g., ~/.zshrc or ~/.bashrc):"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
