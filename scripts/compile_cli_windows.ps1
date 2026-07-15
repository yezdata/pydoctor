$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$OUTPUT_DIR = "dist"
$null = New-Item -ItemType Directory -Force -Path $OUTPUT_DIR

$tempParent = [System.IO.Path]::GetTempPath()
$BUILD_DIR = Join-Path $tempParent ([System.IO.Path]::GetRandomFileName())
$null = New-Item -ItemType Directory -Path $BUILD_DIR

try {
    uv venv "$BUILD_DIR/.venv" --no-config --python 3.12
    
    . (Join-Path $BUILD_DIR ".venv/Scripts/Activate.ps1")

    uv export --package pydoctor-cli --no-hashes --no-editable --output-file "$BUILD_DIR/requirements.txt"

    uv pip install "Nuitka[onefile]"

    uv pip install -r "$BUILD_DIR/requirements.txt" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

    uv pip install --no-deps ./packages/pydoctor_shared_cst
    uv pip install --no-deps ./packages/pydoctor_cli


    $env:CCFLAGS = "-O1"

    python -m nuitka `
        --standalone `
        --onefile `
        --lto=no `
        --mingw64 `
        --enable-plugin=anti-bloat `
        --output-dir="$OUTPUT_DIR" `
        --report=report.xml `
        packages/pydoctor_cli/src/pydoctor_cli/cli.py

} finally {
    Write-Host "Cleaning up temporary build directory..."
    if (Test-Path $BUILD_DIR) {
        Remove-Item -Recurse -Force $BUILD_DIR
    }
}
