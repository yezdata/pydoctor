import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np


SYSTEM_PROMPT = """You are a professional Python docstring generator.
You will receive Python code block (function, class, class method).
Generate a concise docstring for the code block marked with <docstring_place>.
The style of the doctring have to be only sentences summarizing the code block.
Output ONLY the raw docstring text.
Do NOT include Args, Returns, Raises, or any structured sections.
Do NOT output any conversational text or explanations."""


def get_chat_template(user_prompt: str) -> str:
    final_output = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return final_output


# def generate_docstring(
#     session: ort.InferenceSession, tokenizer: Tokenizer, snippet: str
# ) -> str:

#     prompt = get_chat_template(snippet)
#     encoded = tokenizer.encode(prompt)
#     input_ids = [encoded.ids]
#     mask = [encoded.attention_mask]

#     outputs = session.run(
#         None,
#         {
#             "input_ids": np.array(input_ids, dtype=np.int64),
#             "attention_mask": np.array(mask, dtype=np.int64),
#         },
#     )
#     next_token_id = np.argmax(outputs[0][:, -1, :], axis=-1)

#     while (
#         next_token_id
#         not in (tokenizer.token_to_id("<|im_end|>"), tokenizer.eos_token_id)
#         and len(input_ids[0]) < 512
#     ):
#         input_ids[0].append(int(next_token_id))
#         mask[0].append(1)

#         outputs = session.run(
#             None,
#             {
#                 "input_ids": np.array([next_token_id], dtype=np.int64),
#                 "attention_mask": np.array([[1]], dtype=np.int64),
#                 "past_key_values": [np.array(p, dtype=np.float32) for p in outputs[1:]],
#             },
#         )
#         next_token_id = np.argmax(outputs[0][:, -1, :], axis=-1)

#     prompt_length = len(encoded.ids)
#     new_tokens = outputs[0][prompt_length:].tolist()
#     generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
#     print("Generated docstring:")
#     print(generated_text)
#     print("-" * 120)

#     return generated_text


def generate_docstring(
    session: ort.InferenceSession, tokenizer: Tokenizer, snippet: str
) -> str:
    prompt = get_chat_template(snippet)
    encoded = tokenizer.encode(prompt)

    input_ids = encoded.ids
    attention_mask = encoded.attention_mask
    prompt_length = len(input_ids)

    # 1. Příprava fixních parametrů modelu
    # SmolLM2-1.7B má 24 vrstev (0 až 23), num_heads=32, head_dim=64
    num_layers = 24
    num_heads = 32
    head_dim = 64
    batch_size = 1

    # Inicializace prázdné KV cache pro první krok (prefill)
    ort_inputs = {}
    for i in range(num_layers):
        ort_inputs[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_heads, 0, head_dim), dtype=np.float32
        )
        ort_inputs[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_heads, 0, head_dim), dtype=np.float32
        )

    # Hlavní generativní smyčka
    current_input_ids = list(input_ids)
    current_attention_mask = list(attention_mask)

    # Pomocné proměnné pro inkrementální feed
    next_input_ids = np.array([current_input_ids], dtype=np.int64)
    next_attention_mask = np.array([current_attention_mask], dtype=np.int64)

    while len(current_input_ids) < 512:
        # Výpočet korektních position_ids
        seq_len = next_input_ids.shape[1]
        if seq_len > 1:
            position_ids = np.arange(0, seq_len, dtype=np.int64).reshape(
                batch_size, seq_len
            )
        else:
            position_ids = np.array([[len(current_input_ids) - 1]], dtype=np.int64)

        # Sestavení kompletního input feedu
        ort_inputs["input_ids"] = next_input_ids
        ort_inputs["attention_mask"] = next_attention_mask
        ort_inputs["position_ids"] = position_ids

        outputs = session.run(None, ort_inputs)

        # Logits jsou vždy na prvním indexu výstupu
        logits = outputs[0]
        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])

        if next_token_id in (
            tokenizer.token_to_id("<|im_end|>"),
            tokenizer.token_to_id("<|endoftext|>"),
        ):
            break

        current_input_ids.append(next_token_id)
        current_attention_mask.append(1)

        # Příprava dat pro následující krok (předáváme již pouze 1 nový token)
        next_input_ids = np.array([[next_token_id]], dtype=np.int64)
        next_attention_mask = np.array([current_attention_mask], dtype=np.int64)

        # Mapování výstupní KV cache na vstup pro další iteraci
        # Indexace výstupů závisí na přesné struktuře ONNX exportu (logits je 0, následují keys/values)
        for i in range(num_layers):
            ort_inputs[f"past_key_values.{i}.key"] = outputs[1 + 2 * i]
            ort_inputs[f"past_key_values.{i}.value"] = outputs[1 + 2 * i + 1]

    # Dekódování pouze nově vygenerovaných tokenů
    generated_tokens = current_input_ids[prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text
