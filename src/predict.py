import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import set_seed

from src.data.builder import build_data_module
from src.trainer.trainer_builder import build_trainer
from src.vggt.arguments import parse_args
from src.vggt.builder import build_model, build_model_without_ckpt


import torch
from pathlib import Path
from safetensors.torch import load_file  # Add this import for safetensors support

def load_checkpoint(model, checkpoint_path: str, device: str = "cpu"):
    """
    Load a checkpoint into the model.
    Supports:
    - Trainer save_model saved weight directory
    - Individual .pt / .pth files (simple state_dict)
    - .safetensors file
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        # Check if the checkpoint directory contains a .safetensors file
        safetensors_file = checkpoint_path / "model.safetensors"
        if safetensors_file.exists():
            # Load model weights from the safetensors file
            state_dict = load_file(str(safetensors_file), device=device)
        else:
            # Otherwise, look for a .bin file (classic PyTorch saved model)
            model_file = checkpoint_path / "pytorch_model.bin"
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=device)
            else:
                raise FileNotFoundError(f"模型文件未找到：{model_file}")
    else:
        # If the checkpoint is a single file (e.g., .pt or .pth)
        if checkpoint_path.suffix == ".safetensors":
            state_dict = load_file(str(checkpoint_path), device=device)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

    # Check if we need to extract the model's state_dict from a nested dictionary
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    # Load the state_dict into the model
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[evaluate.py] loaded checkpoint from: {checkpoint_path}")
    print(f"[evaluate.py] load_state_dict message: {msg}")

    return model


def main():
    model_args, data_args, loss_args, training_args = parse_args()

    set_seed(training_args.seed)

    # 这里只构建验证集
    data_module = build_data_module(
        train_dataset_dir=None,
        val_dataset_dir=Path(data_args.val_dir),
        seed=training_args.seed,
        target_size=model_args.img_size,
        patch_size=model_args.patch_size,
        use_rays=model_args.use_rays,
        mean=data_args.mean,
        std=data_args.std,
        data_ratio=data_args.data_ratio,
        data_angle=data_args.data_angle,
    )

    val_dataset = data_module.val_dataset
    data_collator = data_module.data_collator

    if val_dataset is None:
        raise ValueError("val_dataset is None，无法进行评估。")

    print(f"[evaluate.py] val samples: {len(val_dataset)}")

    model = build_model_without_ckpt(
        data_args=data_args,
        model_args=model_args,
        loss_args=loss_args,
    )

    # 这里默认从 output_dir 加载最终模型
    # 你也可以改成单独参数，比如 model_args.checkpoint_path
    model = load_checkpoint(
        model=model,
        checkpoint_path=training_args.output_dir,
        device="cuda:0",
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=None,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        change_orientation_threshold=0.5,
    )

    predict_output = trainer.predict(test_dataset=val_dataset)

    predictions = predict_output.predictions

    change_orientation_logits = predictions[0]
    pose_pred = predictions[1]

    change_orientation_prob = torch.sigmoid(torch.tensor(change_orientation_logits)).numpy()

    change_orientation_pred = (change_orientation_prob > 0.5).astype(int)

    if trainer.is_world_process_zero():
        for i in range(len(val_dataset)):
            json_path = Path(data_args.val_dir) / val_dataset[i]["id"] / "generated_0.json"

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "target" not in data:
                data["target"] = {}

            data["target"]["need_adjustment"] = 1
            data["target"]["change_orientation"] = bool(
                int(np.asarray(change_orientation_pred[i]).squeeze()) == 1
            )
            data["target"]["pose"] = np.asarray(pose_pred[i]).tolist()

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()