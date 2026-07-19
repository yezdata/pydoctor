# PyDoctor
> Doctor for your python docstrings.

![Release](https://img.shields.io/github/v/release/yezdata/pydoctor)
![Python](https://img.shields.io/badge/python-3.12-blue)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator)
![License](https://img.shields.io/github/license/yezdata/pydoctor)

PyDoctor is a fully local, LLM-powered CLI tool that automatically writes and manages docstrings in your Python codebase. It parses your source files using **LibCST**, precisely extracting every function, method, and class that needs documentation while preserving your code's exact formatting. Each extracted block is fed — with its surrounding class context — into a fine-tuned **SmolLM2-1.7B-Instruct** model running locally via **llama.cpp**. The model generates a concise, single-paragraph docstring and the result is written back atomically, so your source is never left in a broken state.

PyDoctor aims at providing concise, english and descriptive summarization docstrings. Thus, the style of generated docstrings is more of a summary of target code block. Specific sections such as Args: or Returns: are not included on purpose, as heuristic / deterministic tools are more reliable in generating those. 

On first run (or when using the [installation script](#installation)) PyDoctor downloads the GGUF model from HuggingFace into a local cache (`~/.cache/pydoctor/` on Linux/macOS, `%LOCALAPPDATA%\pydoctor\` on Windows) and reuses it on every subsequent run.

Ignore rules are collected from `.gitignore`, an optional `.pydoctor_ignore` file, and inline `# pydoctor: ignore` comments, so you always stay in control of what gets touched.


## Quick Start

```bash
# 1. Download (Linux / macOS example)
curl -fsSL https://github.com/yezdata/pydoctor/releases/latest/download/install.sh | sh

# 2. Preview what would change — no files are modified
pydoctor ~/my_project --dry-run

# 3. Add docstrings to every undocumented function, method and class
pydoctor ~/my_project
```

> [!TIP]
> Run with `--dry-run` first on an existing codebase so you can review the generated docstrings before they are written.

## Installation

To download the PyDoctor CLI, add `pydoctor` command to your path, and pre-cache the GGUF model, run the installer for your platform:

### Linux / macOS
```bash
curl -fsSL https://github.com/yezdata/pydoctor/releases/latest/download/install.sh | sh
```

### Windows (PowerShell)
```powershell
irm https://github.com/yezdata/pydoctor/releases/latest/download/install.ps1 | iex
```

Or download the pre-built binary for your platform from the [GitHub Releases](../../releases/latest) page.

> [!WARNING]
> **macOS — Gatekeeper / Security Block**
> macOS will block manually downloaded unsigned binaries by default. To allow execution, run:
> ```bash
> xattr -d com.apple.quarantine ./pydoctor-macos-arm64
> ```
> Alternatively, go to **System Settings → Privacy & Security** and click **"Allow Anyway"** after the first blocked launch.
> If you prefer not to bypass Gatekeeper, see [local compilation](#alternatively-compile-locally) below.

### Alternatively compile locally
> Using the prepared `scripts/compile_cli_unix.bash` or `scripts/compile_cli_windows.ps1` script.

**The script uses [uv](https://docs.astral.sh/uv/) as python and venv manager by default**

**Requirements:** uv, git, and a C compiler (GCC / Clang / MSVC) required by Nuitka. Python 3.12 is installed automatically by uv.

**Example usage:**
```bash
git clone https://github.com/yezdata/pydoctor.git
cd pydoctor

bash ./scripts/compile_cli_unix.bash
```


## Usage
```
pydoctor <path> [--replace | --all] [--dry-run] [-v]
```

> **Default:** PyDoctor adds docstrings only where **missing**

| Argument | Description |
|---|---|
| `path` | Python file or directory to process |
| `--replace` | Replace only **existing** docstrings |
| `--all` | Process **all** code blocks (add + replace) |
| `--dry-run` | Preview changes without modifying files |
| `-v`, `--verbose` | Enable verbose output |


## Ignoring Files and Code Blocks

PyDoctor merges ignore rules from three sources:

| Method | Effect |
|--------|--------|
| `.gitignore` entry | Skips matching files and directories |
| `.pydoctor_ignore` entry | Same gitignore syntax, pydoctor-specific |
| `# pydoctor: ignore` comment | Skips that individual function, class or method only |

**Inline ignore example:**
```python
# pydoctor: ignore
def _internal_helper(x):
    ...
```

*Or like this:*
```python
def _internal_helper(x): # pydoctor: ignore
    ...
```


**Ignore only part of code block:**
```python
class PublicAPI:
    # pydoctor: ignore
    def _skip_this_method(self):
        ...

    def document_this_method(self):
        ...  # ← this one will get a docstring
```

**`.pydoctor_ignore` example** (gitignore syntax):
```
tests/
scripts/
**/migrations/*.py
```


## Model

PyDoctor ships with a custom fine-tuned **[SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)** (HuggingFaceTB) quantised to **Q8_0 GGUF** and hosted at [`yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator`](https://huggingface.co/yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator). The model uses the **ChatML** prompt format (`<|im_start|>` / `<|im_end|>`) and is trained + instructed to output only the raw docstring text.

### Fine-tuning pipeline

The model was instruction-fine-tuned in two stages:

1. **Synthetic dataset generation** — Code blocks (functions, methods, classes) were extracted from [codeparrot](https://huggingface.co/datasets/codeparrot/codeparrot-clean) using the same LibCST `CodeExtractor` used by the CLI. Docstrings were then generated by calling **DeepSeek V4 Flash with precise instuctions on how to generate the docstrings**.

2. **Instruct fine-tune** — `SmolLM2-1.7B-Instruct` was loaded in **4-bit NF4** quantisation and fine-tuned with **LoRA** (r=32, α=64) targeting all attention and MLP projection layers (`q/k/v/o_proj`, `gate/up/down_proj`). Training used **AdamW 8-bit**, a cosine LR schedule with warmup.

### EXPERIMENTAL: Custom decoder (research track)

The repo also contains `pydoctor_model_pipeline/model/` — a from-scratch **decoder-only Transformer** implementation built with PyTorch, featuring RMSNorm, SwiGLU FFN and RoPE positional embeddings and using the [StarCoder2](https://huggingface.co/bigcode/starcoder2-15b/tree/main) tokenizer. This architecture (768d, 12 heads, 12 layers - 130M params) was pre-trained using CLM task on code corpus and then fine-tuned on the same synthetic docstring dataset. It serves as a future / experimental baseline; the released binary (at least for now) uses the SmolLM2 instruct fine-tune for better out-of-the-box quality.
