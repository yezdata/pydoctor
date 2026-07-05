import torch
from transformers import Gemma3Config, Gemma3ForCausalLM
import os
from dotenv import load_dotenv

from src.utils.tokenizer import get_finetune_tokenizer
from src.utils.config_models import TokenizerConfig

# ==========================================
# KONFIGURACE A MANUÁLNÍ PROMĚNNÉ
# ==========================================
MODEL_DIR = "models/gemma_finetuned"  # Adresář obsahující config.json a tokenizer
SAFETENSORS_PATH = f"{MODEL_DIR}/model.safetensors"  # Cesta k safetensors souboru
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Manuální vstupní proměnná
USER_INPUT = """
def compute_loss(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_token_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = input_ids[..., 1:]

    start_mask = torch.isin(input_ids[..., :-1], start_token_ids)

    is_docstring = torch.cumsum(start_mask, dim=-1) > 0

    is_not_pad = shift_targets != pad_token_id

    loss_mask = is_docstring & is_not_pad

    shift_targets = torch.where(loss_mask, shift_targets, -100)

    B, S, V = shift_logits.shape
    return criterion(shift_logits.view(B * S, V), shift_targets.view(B * S))
"""

# ==========================================
# 1. NAČTENÍ TOKENIZERU A KONFIGURACE
# ==========================================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("Načítám tokenizer a konfiguraci...")
tokenizer = get_finetune_tokenizer(
    TokenizerConfig(name="google/gemma-3-270m"), token=hf_token
)  # type: ignore
config = Gemma3Config.from_pretrained(MODEL_DIR)

# ==========================================
# 2. INICIALIZACE MODELU A NAČTENÍ VAH (SAFETENSORS)
# ==========================================
print(f"Inicializuji model Gemma 3 na zařízení: {DEVICE}...")
# Inicializace prázdného modelu podle konfigurace
with torch.device(DEVICE):
    model = Gemma3ForCausalLM.from_pretrained(MODEL_DIR)

print(f"Načítám váhy ze safetensors souboru: {SAFETENSORS_PATH}...")
# Manuální načtení tensorů a implementace do modelu
model.eval()  # Přepnutí do eval módu pro vypnutí dropoutu atd.

# Pokud máte dostatek VRAM, doporučuje se přetypovat na bfloat16/float16 pro Gemma 3
if DEVICE == "cuda":
    model = model.to(dtype=torch.bfloat16)

# ==========================================
# 3. PŘÍPRAVA VSTUPU A INFERENCE
# ==========================================
# Aplikace chat template (Gemma 3 vyžaduje správné formátování role)
# Tokenizace stringu do tensorů


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    if input_ids.numel() == 0:
        input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

    generated = input_ids

    for _ in range(max_new_tokens):
        # input_pos MUSÍ přesně odpovídat aktuální délce celého generovaného tensoru,
        # protože model v každém kroku přepočítává úplně všechno od nuly.
        seq_len = generated.shape[1]
        input_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

        # Dopředný průchod celou sekvencí
        logits = model(generated, input_pos=input_pos).logits

        # Extrakce logitů POUZE pro poslední vygenerovaný token
        next_token_logits = logits[:, -1, :]

        # Greedy search (argmax)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Přilepení nového tokenu na konec
        generated = torch.cat([generated, next_token_id], dim=1)

        # Kontrola ukončení
        if (
            tokenizer.eos_token_id is not None
            and next_token_id.item() == tokenizer.eos_token_id
        ):
            break

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


print("\nGeneruji odpověď...")
with torch.no_grad():
    outputs = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=USER_INPUT,
        device=DEVICE,
        max_new_tokens=512,
    )

# ==========================================
# 4. DEKÓDOVÁNÍ VÝSTUPU
# ==========================================
# Ořízneme vstupní tokeny, abychom vypsali pouze nově generovaný text
print("\n--- VÝSTUP MODELU ---")
print(outputs)
