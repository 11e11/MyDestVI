"""
Fixed conditional Encoder:
- Always treats provided categorical variables (labels, optional batch) as conditional inputs.
- One-hot them and concatenate to raw count input (after log1p 可选).
- No inject_covariates flag.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal
from .layers import FCLayers
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,        # gene count features
        n_latent: int,
        n_labels: int,       # number of cell types
        n_batch: int = 0,    # number of batches (0 -> no batch covariate)
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.05,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        return_dist: bool = True,
        log_variational: bool = True,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_labels = n_labels
        self.n_batch = n_batch
        self.return_dist = return_dist
        self.log_variational = log_variational

        # 计算扩展后的输入维度： genes + one-hot(labels) + one-hot(batch?)
        self.cat_dim = n_labels + (n_batch if n_batch > 0 else 0)
        effective_n_in = n_input + self.cat_dim

        self.backbone = FCLayers(
            n_in=effective_n_in,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.mu_layer = nn.Linear(n_hidden, n_latent)
        self.var_layer = nn.Linear(n_hidden, n_latent)

    def _one_hot_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).long()
        return F.one_hot(labels, num_classes=self.n_labels).float()

    def _one_hot_batch(self, batch_index: torch.Tensor | None) -> torch.Tensor | None:
        if self.n_batch <= 0 or batch_index is None:
            return None
        b = batch_index.view(-1).long()
        return F.one_hot(b, num_classes=self.n_batch).float()

    def forward(
        self,
        x: torch.Tensor,          # [B, n_input]
        labels: torch.Tensor,     # [B] ints
        batch_index: torch.Tensor | None = None,
    ):
        if self.log_variational:
            x_proc = torch.log1p(torch.clamp_min(x, 0.0))
        else:
            x_proc = x

        oh_labels = self._one_hot_labels(labels)
        oh_batch = self._one_hot_batch(batch_index)
        if oh_batch is not None:
            x_cat = torch.cat([x_proc, oh_labels, oh_batch], dim=1)
        else:
            x_cat = torch.cat([x_proc, oh_labels], dim=1)

        h = self.backbone(x_cat)
        mu = self.mu_layer(h)
        logvar = self.var_layer(h)
        std = torch.exp(0.5 * logvar)

        qz = Normal(mu, std)
        if self.return_dist:
            return qz, qz.rsample()
        else:
            return mu, logvar