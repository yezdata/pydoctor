---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- code
- python
- docstring
- documentation
- code-generation
- lora
- qlora
- smollm2
- instruct
- causal-lm
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
pipeline_tag: text-generation
model-index:
- name: SmolLM2-1.7B-Instruct-DocstringGenerator
  results: []
datasets:
- codeparrot/codeparrot-clean
---

# SmolLM2-1.7B-Instruct · DocstringGenerator

> A fine-tuned **[SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)** specialised in writing **concise, high-level Python docstrings** for functions, methods and classes.
> This model is the backbone of the **[PyDoctor](https://github.com/yezdata/pydoctor)** CLI — a fully local, LLM-powered tool that automatically writes and manages docstrings in your Python codebase.

[![GitHub](https://img.shields.io/badge/GitHub-yezdata%2Fpydoctor-black?logo=github)](https://github.com/yezdata/pydoctor)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/)
[![Base Model](https://img.shields.io/badge/base-SmolLM2--1.7B--Instruct-yellow?logo=huggingface)](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

---

## Intended Use

The model generates **summary-style docstrings** — single-paragraph, plain-English descriptions of a Python code block's purpose and architectural role. It does **not** produce `Args:`, `Returns:`, or `Raises:` sections by design.

**Suitable for:**
- Automated docstring generation in CI/CD pipelines
- Interactive IDE plugins
- Local, privacy-preserving documentation workflows via llama.cpp / GGUF

**Not suitable for:**
- General-purpose code generation
- Generating full NumPy/Google-style docstrings with parameter tables (explicitly omitted)
- Non-Python languages

---

## Quick Start
### With llama.cpp (GGUF · recommended for local use)

```bash
# Download the Q8_0 GGUF
huggingface-cli download \
  yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator \
  smollm2_1_7b_instruct_merged-q8_0.gguf \
  --local-dir ./models

# Run inference
llama-cli \
  -m ./models/smollm2_1_7b_instruct_merged-q8_0.gguf \
  --chat-template chatml \
  -p "..."
```

> **Tip:** The [PyDoctor CLI](https://github.com/yezdata/pydoctor) handles prompt construction, parsing, and atomic file rewrites out of the box.

---

## Prompt Format (ChatML)

The model uses the **ChatML** template native to SmolLM2-Instruct:

```
<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
CONTEXT
{context_code}

TARGET CODE
{target_code}<|im_end|>
<|im_start|>assistant
```

The model then generates only the raw docstring text, terminated by `<|im_end|>`.

**Context definition:**
- **function** target -> context = "Independent code block"
- **method** target → context = `__init__` signature of its enclosing class
- **class** target → context = signatures of its methods

---

## Training Pipeline

### Stage 1 — Code Extraction

Raw Python source files were streamed from **[codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)** (~200 k samples). Each file passed a quality filter that rejected:

| Filter | Threshold |
|---|---|
| Too few lines | < 3 non-empty lines |
| Minified code | avg line length > 150 chars |
| Low alphabetic ratio | < 15 % (binary / machine-generated) |
| Repetitive boilerplate | unique line ratio < 10 % |
| Oversized files | > 50 000 characters |

Surviving files were parsed with **[LibCST](https://libcst.readthedocs.io/)** producing `(target, context)` pairs.

### Stage 2 — Synthetic Docstring Generation

`(target, context)` pairs were labelled in parallel using **DeepSeek V4 Flash** (via OpenRouter):

The teacher-model system prompt enforced:
1. Describe semantic purpose and architectural role, not implementation details
2. Use context to disambiguate class membership

### Stage 3 — Instruct Data Preparation & Tokenisation

Synthetic batches were assembled into ChatML prompt/completion pairs:

```python
prompt = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\nCONTEXT\n{context}\n\nTARGET CODE\n{target}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
completion = f"{docstring}<|im_end|>"
```

Labels were constructed so that **only completion tokens** are trained on — prompt tokens are masked from cross-entropy loss.

### Stage 4 — QLoRA Fine-tuning

Fine-tuning was performed on Kaggle kernels (`instruct_finetune.py`):

| Hyperparameter | Value |
|---|---|
| Quantisation | 4-bit NF4, double quant, fp16 compute |
| LoRA rank `r` | 32 |
| LoRA alpha `α` | 64 |
| LoRA dropout | 0.2 |
| LoRA bias | none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | AdamW 8-bit (bitsandbytes) |
| Learning rate | 2e-4 |
| LR schedule | Cosine with 5 % warmup |
| Weight decay | 0.01 |
| Batch size | 8 per device |
| Gradient accumulation | 8 steps → effective batch 64 |
| Epochs | 1 |
| Max sequence length | 1 024 tokens (95th-pct filter) |
| Validation split | 1 % held-out, evaluated each epoch |
| Seed | 1337 |

Loss = next-token cross-entropy, **prompt tokens ignored** via label mask.

### Stage 5 — LoRA Merge & GGUF Export

After training, LoRA adapters were merged back into the base model weights and converted to **Q8_0 GGUF** using `llama.cpp`:

```
LoRA adapter (epoch 1, safetensors)
        │
        ▼  merge_and_unload()
        │
merged fp16 safetensors
        │
        ▼  llama.cpp convert_hf_to_gguf.py --outtype q8_0
        ▼
smollm2_1_7b_instruct_merged-q8_0.gguf
```

---

## Files

| File | Description |
|---|---|
| `smollm2_1_7b_instruct_merged-q8_0.gguf` | Q8_0 GGUF for llama.cpp — recommended for local use |
| `safetensors/model.safetensors` | Merged fp16 weights |
| `safetensors/config.json` | HuggingFace model configuration |
| `safetensors/tokenizer.json` / `safetensors/tokenizer_config.json` | SmolLM2-1.7B-Instruct tokenizer |

---

## Limitations & Bias

- **Summary-only style:** the model is trained to output a single-paragraph summary. It will not produce `Args:` / `Returns:` sections.
- **Python only:** trained exclusively on Python source code from codeparrot-clean.
- **Context dependency:** quality improves when the correct context string is provided. Passing an empty context for class methods may reduce coherence.
- **Teacher model bias:** docstring style reflects DeepSeek V4 Flash's preferences filtered through the strict prompt rules. Unusual code idioms may yield generic descriptions.
- **Not a general assistant:** the model is heavily specialised and will likely perform poorly on tasks other than docstring generation.

---

## Citation

```bibtex
@misc{pydoctor2026,
  author       = {yezdata},
  title        = {PyDoctor: Local LLM-powered Python Docstring Generator},
  year         = {2026},
  howpublished = {\url{https://github.com/yezdata/pydoctor}},
  note         = {Fine-tuned SmolLM2-1.7B-Instruct model available at
                  \url{https://huggingface.co/yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator}}
}
```

---

## License

This model is released under the **Apache 2.0** license, matching the base `SmolLM2-1.7B-Instruct` model.  
Training data originates from `codeparrot/codeparrot-clean` (MIT)
