# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from src.vggt.heads.ray_head import RayHead
from src.vggt.heads.track_modules.orientation_head import OrientationHead
from src.vggt.models.aggregator import Aggregator
from src.vggt.heads.camera_head import CameraHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        change_orientation_loss_weight: float = 1.0,
        pose_loss_weight: float = 1.0,
        use_rays: bool = True,
        rays_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.change_orientation_head = OrientationHead(dim_in=2 * embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        if use_rays:
            self.ray_head = RayHead(dim_in=2 * embed_dim)
        else:
            self.ray_head = None

        self.change_orientation_loss_weight = change_orientation_loss_weight
        self.pose_loss_weight = pose_loss_weight
        self.rays_loss_weight = rays_loss_weight

    def compute_loss(
            self,
            change_orientation_pred: torch.Tensor,
            pose_pred: torch.Tensor,
            rays_pred: torch.Tensor | None,
            change_orientation: torch.Tensor,
            pose: torch.Tensor,
            rays: torch.Tensor | None,
            rays_mask: torch.Tensor | None
    ):
        loss_dict = {}

        loss_change_orientation = F.binary_cross_entropy_with_logits(
            change_orientation_pred,
            change_orientation,
        )
        loss_dict["loss_change_orientation"] = loss_change_orientation
        total_loss = self.change_orientation_loss_weight * loss_change_orientation

        loss_pose = F.smooth_l1_loss(
            pose_pred,
            pose,
        )
        loss_dict["loss_pose"] = loss_pose
        total_loss = total_loss + self.pose_loss_weight * loss_pose

        if rays_pred is not None and rays is not None and rays_mask is not None:
            loss_rays = self.compute_rays_loss(
                masks=rays_mask,
                rays_pred=rays_pred,
                rays=rays,
            )
            loss_dict["loss_rays"] = loss_rays
            total_loss = total_loss + self.rays_loss_weight * loss_rays
        else:
            loss_rays = loss_change_orientation.new_zeros(())
            loss_dict["loss_rays"] = loss_rays

        return total_loss, loss_dict

    def compute_rays_loss(self, masks, rays_pred, rays):
        eps = 1e-6

        t_pred, d_pred = rays_pred[..., :3], rays_pred[..., 3:]
        t_gt, d_gt = rays[..., :3], rays[..., 3:]

        d_pred = F.normalize(d_pred, dim=-1, eps=eps)
        d_gt = F.normalize(d_gt, dim=-1, eps=eps)

        loss_t = torch.abs(t_pred - t_gt).mean(dim=-1)
        loss_d = torch.abs(d_pred - d_gt).mean(dim=-1)

        ray_l1 = loss_t + loss_d

        valid_mask = masks.float()
        valid_count = valid_mask.sum().clamp_min(1.0)

        loss = (ray_l1 * valid_mask).sum() / valid_count
        return loss

    def forward(
        self,
        pixel_values: torch.Tensor,
        masks: torch.Tensor,
        change_orientation: torch.Tensor,
        pose: torch.Tensor,
        rays: torch.Tensor | None,
    ):
        images = pixel_values.unsqueeze(1)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        with torch.amp.autocast('cuda', enabled=False):
            change_orientation_pred = self.change_orientation_head(aggregated_tokens_list).squeeze(1)
            pose_pred = self.camera_head(aggregated_tokens_list)[-1].squeeze(1)

        rays_pred = None
        if self.ray_head is not None and rays is not None:
            rays_pred = self.ray_head(aggregated_tokens_list, patch_start_idx).squeeze(1)
            B, H, W, _ = rays.shape
            rays_pred = rays_pred.view(B, H, W, 6)

        loss, loss_dict = self.compute_loss(
            change_orientation_pred=change_orientation_pred,
            pose_pred=pose_pred,
            rays_pred=rays_pred,
            change_orientation=change_orientation,
            pose=pose,
            rays=rays,
            rays_mask=masks,
        )

        outputs = {
            "change_orientation": change_orientation_pred,
            "pose": pose_pred,
        }

        outputs.update(loss_dict)
        outputs["loss"] = loss

        return outputs

