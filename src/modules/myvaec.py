"""
VAEC 固定条件版：
- 始终使用细胞类型 (labels) + 可选 batch 的 one-hot 拼接。
- 去除 inject_covariates 相关逻辑。
- 与 scvi CondSCVI 行为类似：inference 返回 Normal 分布。
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from src.nn.encoder import Encoder
from src.nn.layers import FCLayers
from src.distributions.negative_binomial import NegativeBinomial
import torch.nn.functional as F

class VAEC(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 5,
        n_layers: int = 2,
        dropout_rate: float = 0.05,
        ct_weight=None,
        encode_covariates: bool = False,   # 保留参数名但逻辑统一
        log_variational: bool = True,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_labels = n_labels
        self.n_batch = n_batch if encode_covariates else 0
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.log_variational = log_variational

        # dispersion logits (per gene)
        self.px_r = nn.Parameter(torch.zeros(n_input))

        # Encoder (固定条件拼接)
        self.z_encoder = Encoder(
            n_input=n_input,
            n_latent=n_latent,
            n_labels=n_labels,
            n_batch=self.n_batch,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            return_dist=True,
            log_variational=log_variational,
        )

        # Decoder: 输入是 latent z + one-hot(labels) (+ one-hot(batch))
        cat_dim = n_labels + (self.n_batch if self.n_batch > 0 else 0)
        decoder_in = n_latent + cat_dim
        self.decoder_backbone = FCLayers(
            n_in=decoder_in,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        self.px_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_input),
            nn.Softplus(),
        )

        # 细胞类型权重
        if ct_weight is None:
            ct_weight = torch.ones(n_labels)
        else:
            ct_weight = torch.tensor(ct_weight, dtype=torch.float32)
        self.register_buffer("ct_weight", ct_weight)

    def _one_hot_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).long()
        return F.one_hot(labels, num_classes=self.n_labels).float()

    def _one_hot_batch(self, batch_index: torch.Tensor | None) -> torch.Tensor | None:
        if self.n_batch <= 0 or batch_index is None:
            return None
        b = batch_index.view(-1).long()
        return F.one_hot(b, num_classes=self.n_batch).float()

    def inference(self, x, labels, batch_index=None):
        # 使用 Encoder 得到 q(z|x)
        qz, z = self.z_encoder(x, labels, batch_index)
        library = x.sum(1, keepdim=True)
        return {
            "z": z,
            "qz": qz,
            "qz_m": qz.loc,
            "qz_v": qz.scale.pow(2),
            "library": library,
        }

    def generative(self, z, library, labels, batch_index=None):
        oh_labels = self._one_hot_labels(labels)
        oh_batch = self._one_hot_batch(batch_index)
        if oh_batch is not None:
            dec_in = torch.cat([z, oh_labels, oh_batch], dim=1)
        else:
            dec_in = torch.cat([z, oh_labels], dim=1)

        h = self.decoder_backbone(dec_in)
        px_scale = self.px_decoder(h)  # [B, n_input]
        px_rate = library * px_scale   # broadcast
        return {"px_rate": px_rate}

    def forward(self, item, kl_weight=1.0):
        x = item["X"]
        labels = item["labels"]
        batch_index = item.get("batch", None)

        inf = self.inference(x, labels, batch_index)
        gen = self.generative(inf["z"], inf["library"], labels, batch_index)

        px_rate = torch.clamp(gen["px_rate"], min=1e-8)
        reconst_loss = -NegativeBinomial(px_rate, logits=self.px_r).log_prob(x).sum(-1)

        # KL(q(z)|p(z))
        pz = Normal(torch.zeros_like(inf["qz_m"]), torch.ones_like(inf["qz_v"]))
        qz_d = Normal(inf["qz_m"], torch.sqrt(inf["qz_v"]))
        kl_local = kl_divergence(qz_d, pz).sum(-1)

        scaling = self.ct_weight[labels.view(-1).long()]
        loss = torch.mean(scaling * (reconst_loss + kl_weight * kl_local))

        return {
            "loss": loss,
            "reconstruction_loss": reconst_loss.mean(),
            "kl_local": kl_local.mean(),
        }