from argparse import ArgumentParser
import json
import os
import torch
from safetensors.torch import save_file, load_file

from pydoctor_model_pipeline.utils.config_models import DecoderConfig, MainConfig
from pydoctor_model_pipeline.model.decoder_arch import DecoderModel
from pydoctor_model_pipeline.utils.tokenizer import get_finetune_tokenizer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.base_model_path}/config.json", "r") as f:
        config = MainConfig.model_validate_json(f.read())

    tokenizer = get_finetune_tokenizer(config.tokenizer)

    config.decoder.vocab_size = len(tokenizer)

    print(len(tokenizer))

    model = DecoderModel(config.decoder, eos_token_id=0)

    state_dict = load_file(f"{args.base_model_path}/epoch_5/model.safetensors")
    if "token_embedding.weight" not in state_dict and "head.weight" in state_dict:
        state_dict["token_embedding.weight"] = state_dict["head.weight"].clone()
    elif "head.weight" not in state_dict and "token_embedding.weight" in state_dict:
        state_dict["head.weight"] = state_dict["token_embedding.weight"].clone()

    model.load_state_dict(strict=True, state_dict=state_dict)

    hf_state_dict = {}

    mapping = {
        "token_embedding.weight": "model.embed_tokens.weight",
        "final_norm.weight": "model.norm.weight",
        "head.weight": "lm_head.weight",
    }

    for src_key, dst_key in mapping.items():
        if src_key in state_dict:
            hf_state_dict[dst_key] = state_dict[src_key]

    for i in range(config.decoder.n_layers):
        hf_state_dict[f"model.layers.{i}.input_layernorm.weight"] = state_dict[
            f"layers.{i}.norm1.weight"
        ]
        hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = state_dict[
            f"layers.{i}.q_proj.weight"
        ]
        hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = state_dict[
            f"layers.{i}.k_proj.weight"
        ]
        hf_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = state_dict[
            f"layers.{i}.v_proj.weight"
        ]
        hf_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[
            f"layers.{i}.out_proj.weight"
        ]
        hf_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = state_dict[
            f"layers.{i}.norm2.weight"
        ]
        hf_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[
            f"layers.{i}.ffn.wo.weight"
        ]

        wi_weight = state_dict[f"layers.{i}.ffn.wi.weight"]

        up_proj_weight, gate_proj_weight = torch.chunk(wi_weight, 2, dim=0)

        hf_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight
        hf_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight

    model_path = os.path.join(args.output_dir, "model.safetensors")
    save_file(hf_state_dict, model_path)

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "eos_token_id": 0,
        "hidden_act": "silu",
        "hidden_size": config.decoder.d_model,
        "initializer_range": 0.02,
        "intermediate_size": config.decoder.d_ffn,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": config.decoder.n_head,
        "num_hidden_layers": config.decoder.n_layers,
        "num_key_value_heads": config.decoder.n_head,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "vocab_size": config.decoder.vocab_size,
    }

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    tokenizer.save_pretrained(args.output_dir)
