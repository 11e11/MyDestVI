"""
Simplified FCLayers:
- Does NOT know任何类别信息
- 输入维度固定为构造时的 n_in
- 不做 one-hot，不做拼接；上层自己准备好特征
"""
from __future__ import annotations
import torch
import torch.nn as nn

class FCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.05,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers = []
        if n_layers <= 0:
            layers.append(nn.Linear(n_in, n_out))
        else:
            for i in range(n_layers):
                in_dim = n_in if i == 0 else n_hidden
                out_dim = n_out if i == n_layers - 1 else n_hidden
                layers.append(nn.Linear(in_dim, out_dim))
                if i != n_layers - 1:
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(out_dim))
                    if use_layer_norm:
                        layers.append(nn.LayerNorm(out_dim))
                    layers.append(nn.ReLU(inplace=True))
                    if dropout_rate and dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [B, n_in], got {tuple(x.shape)}")
        return self.net(x)