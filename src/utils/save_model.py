from accelerate import Accelerator
from pydantic_settings import BaseSettings
import json
import os
from torch import nn
from safetensors.torch import save_model


def save_decoder_model(
    accelerator: Accelerator,
    model: nn.Module,
    save_path: str,
    model_config: BaseSettings,
    train_config: BaseSettings,
    loss: float | None = None,
    eval_loss: float | None = None,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    resume_dir = os.path.join(save_path, "accelerator_state")
    accelerator.save_state(resume_dir, safe_serialization=True)

    # SAVE ONLY WHOLE MODEL
    unwrapped_model = accelerator.unwrap_model(model)
    save_model(unwrapped_model, f"{save_path}/model.safetensors")

    with open(f"{save_path}/config.json", "w") as f:
        f.write(model_config.model_dump_json(indent=4))
    with open(f"{save_path}/train_config.json", "w") as f:
        f.write(train_config.model_dump_json(indent=4))

    if loss is not None and eval_loss is not None:
        with open(f"{save_path}/train_state.json", "w") as f:
            json.dump({"train_loss": loss, "eval_loss": eval_loss}, f, indent=4)
