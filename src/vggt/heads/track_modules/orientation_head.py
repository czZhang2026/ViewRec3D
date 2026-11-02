import torch
import torch.nn as nn

class OrientationHead(nn.Module):
    """
    Predict change_orientation from aggregated tokens.

    Input:
        aggregated_tokens_list: list of token tensors
            use the last tensor, same as CameraHead
            expected shape of aggregated_tokens_list[-1]: [B, S, N, C]
            and token[:, :, 0] is the CLS / camera token

    Output:
        logits: [B, S, 1]
    """

    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim_in // 2

        self.token_norm = nn.LayerNorm(dim_in)

        self.head = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, aggregated_tokens_list: list[torch.Tensor]) -> torch.Tensor:
        # use tokens from the last block
        tokens = aggregated_tokens_list[-1]          # [B, S, N, C]

        # extract CLS / camera token
        cls_tokens = tokens[:, :, 0]                 # [B, S, C]
        cls_tokens = self.token_norm(cls_tokens)

        logits = self.head(cls_tokens)               # [B, S, 1]
        return logits