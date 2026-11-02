import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class RayHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int = 6,
        conf_activation: str = "expp1",
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.conf_activation = conf_activation

        self.norm = nn.LayerNorm(dim_in)
        self.head = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),   # 6 ray
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        x = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        x = self.norm(x)
        rays = self.head(x)

        return rays

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        act = activation.lower()
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return F.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x