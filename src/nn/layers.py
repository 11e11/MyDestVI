"""
src/nn/layers.py

Minimal, robust implementation of FCLayers used by your MRDeconv/VAEC code.

- Constructor signature compatible with previous usage in repo.
- forward(x, cat=None) accepts:
    - cat is None
    - cat is a torch.Tensor (1D or 2D)
    - cat is a list/tuple of tensors (will be concatenated or first elem used)
    - cat may be integer labels (tensor) — will be converted to long and unsqueezed if needed
- The class implements optional batch/layer norm and dropout, and returns a tensor.
"""
from __future__ import annotations
from typing import Sequence, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Optional[Sequence[int]] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.05,
        inject_covariates: bool = False,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        return_last_layer: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_cat_list = list(n_cat_list) if n_cat_list is not None else []
        self.inject_covariates = inject_covariates

        # compute covariate dimension if categorical lists provided
        self.cat_dim = int(sum(self.n_cat_list)) if len(self.n_cat_list) > 0 else 0

        layers = []
        in_dim = n_in + (self.cat_dim if inject_covariates else 0)

        hid = n_hidden
        if n_layers <= 0:
            # direct linear
            layers.append(nn.Linear(in_dim, n_out))
        else:
            for i in range(n_layers):
                out_dim = n_out if i == (n_layers - 1) else hid
                layers.append(nn.Linear(in_dim if i == 0 else hid, out_dim))
                if i != (n_layers - 1):
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(out_dim))
                    if use_layer_norm:
                        layers.append(nn.LayerNorm(out_dim))
                    layers.append(nn.ReLU(inplace=True))
                    if dropout_rate and dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
            # if last layer was created as linear above, ensure activation/softplus is applied outside if needed

        self.net = nn.Sequential(*layers)
        self.return_last_layer = return_last_layer

    def _process_cat(self, cat):
        """Normalize various cat inputs into a tensor or None."""
        if cat is None:
            return None

        # cat as list or tuple -> try to concatenate or take first
        if isinstance(cat, (list, tuple)):
            if len(cat) == 0:
                return None
            if len(cat) == 1:
                cat = cat[0]
            else:
                # try to convert each to tensor and concat on last dim
                try:
                    processed = []
                    for c in cat:
                        if c is None:
                            continue
                        if not torch.is_tensor(c):
                            c = torch.as_tensor(c)
                        if c.dim() == 1:
                            c = c.unsqueeze(-1)
                        processed.append(c)
                    if len(processed) == 0:
                        return None
                    return torch.cat(processed, dim=-1)
                except Exception:
                    # fallback: take first element
                    cat = cat[0]

        # if still not a tensor, coerce
        if not torch.is_tensor(cat):
            try:
                cat = torch.as_tensor(cat)
            except Exception:
                # last resort
                return None

        # if cat is 1D (labels), unsqueeze to 2D (N,1)
        if cat.dim() == 1:
            cat = cat.unsqueeze(-1)

        # ensure float for concatenation with x (unless it's intended to be index)
        if not torch.is_floating_point(cat):
            # keep integer labels as long if they represent categories
            try:
                cat = cat.long()
            except Exception:
                cat = cat.float()

        return cat

    def forward(self, x: torch.Tensor, cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, n_in) tensor
        cat: None, tensor (batch, k) or list/tuple of tensors
        """
        # Input checks
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        cat_processed = self._process_cat(cat)

        if self.inject_covariates and cat_processed is not None:
            # if cat_processed contains integer labels (long), we convert to one-hot vectors if needed
            if cat_processed.dtype == torch.long:
                # if categorical dims known (self.n_cat_list), expand accordingly
                if self.cat_dim > 0 and len(self.n_cat_list) > 0:
                    onehots = []
                    start = 0
                    for dim in self.n_cat_list:
                        col = cat_processed[:, start : start + 1]
                        # if col contains labels in [0, dim-1], use one-hot
                        try:
                            oh = F.one_hot(col.squeeze(-1).long(), num_classes=dim).to(dtype=torch.float32)
                        except Exception:
                            # fallback: convert to float
                            oh = col.to(dtype=torch.float32)
                        onehots.append(oh)
                        start += 1
                    cat_tensor = torch.cat(onehots, dim=-1)
                else:
                    # unknown categories -> cast to float
                    cat_tensor = cat_processed.to(dtype=torch.float32)
            else:
                cat_tensor = cat_processed.to(dtype=torch.float32)
            # concat on last dim
            inp = torch.cat([x, cat_tensor], dim=-1)
        else:
            inp = x

        out = self.net(inp)
        return out