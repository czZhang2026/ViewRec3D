from pathlib import Path

import torch

from src.vggt.arguments import ModelArguments, LossArguments, DataArguments
from src.vggt.models.vggt import VGGT

def build_model_without_ckpt(
    data_args: DataArguments,
    model_args: ModelArguments,
    loss_args: LossArguments,
):
    model = VGGT(
        img_size=model_args.img_size,
        patch_size=model_args.patch_size,
        embed_dim=model_args.embed_dim,
        change_orientation_loss_weight=loss_args.change_orientation_loss_weight,
        pose_loss_weight=loss_args.pose_loss_weight,
        use_rays=model_args.use_rays,
        rays_loss_weight=loss_args.rays_loss_weight,
    )
    return model

def build_model(
    data_args: DataArguments,
    model_args: ModelArguments,
    loss_args: LossArguments,
):
    model = VGGT(
        img_size=model_args.img_size,
        patch_size=model_args.patch_size,
        embed_dim=model_args.embed_dim,
        change_orientation_loss_weight=loss_args.change_orientation_loss_weight,
        pose_loss_weight=loss_args.pose_loss_weight,
        use_rays=model_args.use_rays,
        rays_loss_weight=loss_args.rays_loss_weight,
    )
    model = load_pretrained_weights(
        model=model,
        ckpt_path=model_args.ckpt_path,
    )

    return model

def load_pretrained_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
) -> torch.nn.Module:
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=map_location)

    model_dict = model.state_dict()

    filtered_ckpt = {
        k: v
        for k, v in checkpoint.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    skipped = [k for k in checkpoint.keys() if k not in filtered_ckpt]

    print(f"[builder] load_partial_weights from: {ckpt_path}")
    print(f"[builder] matched params: {len(filtered_ckpt)}")
    print(f"[builder] skipped params: {len(skipped)}")
    if skipped:
        print(f"[builder] examples of skipped keys: {skipped[:20]}")

    msg = model.load_state_dict(filtered_ckpt, strict=False)
    print(f"[builder] load_state_dict message: {msg}")

    return model