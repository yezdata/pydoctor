from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from src.model.decoder_arch import DecoderModel
from src.utils.config_models import DecoderConfig, TokenizerConfig
from src.utils.tokenizer import get_pretrain_tokenizer, get_finetune_tokenizer


CHECKPOINT_DIR = Path("models/v1/finetune/epoch_1")

INPUT_TEXT = """
@api.post("/query", response_class=StreamingResponse)
def get_response(data: Query):
    user_query = data.prompt
    doc_name = data.doc_name
    context = data.context[-HISTORY_LEN:]

    logger.debug(context)

    try:
        query_emb = convert_embedding_batch([user_query], client)[0]
    except Exception:
        logger.exception("Error during query embedding creation")
        return StreamingResponse(
            format_error("Could not create embedding for the query.")
        )

    try:
        sim_embeddings = search_similar(query_emb, doc_name, k=SEARCH_K_EMBEDDINGS)
    except Exception:
        logger.exception("Error during search for similar embeddings")
        return StreamingResponse(format_error("Could not find similar embeddings."))

    logger.debug(f"Similar embeddings: {sim_embeddings}")

    return StreamingResponse(
        get_llm_res(user_query, sim_embeddings, context), media_type="text/event-stream"
    )\n\n<|startoffuncdocstring|>
"""
MAX_NEW_TOKENS = 500


def load_model_and_tokenizer() -> tuple[
    DecoderModel, PreTrainedTokenizerFast, torch.device
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with (CHECKPOINT_DIR / "config.json").open("r", encoding="utf-8") as f:
        model_config = DecoderConfig.model_validate_json(f.read())

    tok_config = TokenizerConfig()  # type: ignore
    if "finetune" in str(CHECKPOINT_DIR):
        tokenizer = get_finetune_tokenizer(tok_config)
    else:
        tokenizer = get_pretrain_tokenizer(tok_config)

    model = DecoderModel(
        config=model_config,
        eos_token_id=tokenizer.eos_token_id,
    )

    state_dict = load_file(CHECKPOINT_DIR / "model.safetensors", device="cpu")

    if "token_embedding.weight" not in state_dict and "head.weight" in state_dict:
        state_dict["token_embedding.weight"] = state_dict["head.weight"]
    elif "head.weight" not in state_dict and "token_embedding.weight" in state_dict:
        state_dict["head.weight"] = state_dict["token_embedding.weight"]

    model.load_state_dict(state_dict, strict=True)
    assert model.token_embedding.weight.data_ptr() == model.head.weight.data_ptr(), (
        "Weight tying is broken! Head and Embedding point to different memory locations."
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device


@torch.inference_mode()
def generate_text(
    model: DecoderModel,
    tokenizer: PreTrainedTokenizerFast,
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
        logits = model(generated, input_pos=input_pos)

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


def main() -> None:
    model, tokenizer, device = load_model_and_tokenizer()
    output_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=INPUT_TEXT,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print("INPUT:")
    print(INPUT_TEXT)
    print()
    print("OUTPUT:")
    print(output_text)


if __name__ == "__main__":
    main()
