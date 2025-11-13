"""
简单可用的 Encoder 实现：
- 使用 FCLayers 作为特征提取主干（支持类别条件 n_cat_list）
- 两个线性头：mu_layer, var_layer（输出 logvar）
- 当 return_dist=True 时，返回 torch.distributions.Normal(loc=mu, scale=std)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

# 注意：使用相对导入，避免 from src.nn 导致循环
from .layers import FCLayers


class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_cat_list=None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.05,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        return_dist: bool = False,
    ):
        super().__init__()
        if n_cat_list is None:
            n_cat_list = []

        # 主干 MLP（支持条件输入）
        self.net = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # 头部
        self.mu_layer = nn.Linear(n_hidden, n_latent)
        self.var_layer = nn.Linear(n_hidden, n_latent)  # 输出 logvar

        self.return_dist = return_dist

    def forward(self, x: torch.Tensor, cat_list=None, return_dist: bool | None = None):
        """
        x: [B, n_input]
        cat_list: None 或 [labels, (batch_index, ...)]，每个应为 [B] 的 LongTensor
        return_dist: 若为 None 则用 self.return_dist
        """
        # 统一 cat_list 形状为 1D
        if cat_list is not None:
            proc = []
            for t in cat_list:
                if t is None:
                    continue
                if t.dim() > 1:
                    t = t.squeeze(-1)
                t = t.long()
                proc.append(t)
            cat_list = proc if len(proc) > 0 else None

        # 前向
        h = self.net(x, cat_list) if cat_list is not None else self.net(x)

        mu = self.mu_layer(h)
        logvar = self.var_layer(h)
        std = torch.exp(0.5 * logvar)

        use_dist = self.return_dist if return_dist is None else return_dist
        if use_dist:
            return Normal(loc=mu, scale=std)  # 与 VAEC.inference 期望兼容
        else:
            return mu, logvar