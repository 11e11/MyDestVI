"""
VAEC 固定条件版（修复 NB 参数化 + 恢复 library 推断 + 增加 library KL）：
- decoder 输出 px_scale ∈ (0,1)，px_rate = library * px_scale
- 基因级别离散度 px_r 仍作为 logits 传入 NegativeBinomial（内部 exp -> theta>0）
- inference 增加 q(log l | x) 的头并加入 KL(q(l)||p(l))
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
        # library 先验超参（Normal(mean, var) on log-library）
        library_prior_mean: float = 0.0,
        library_prior_logvar: float = 0.0,
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

        # dispersion raw logits (per gene), 供 NB logits 使用（内部 exp -> theta）
        self.px_r = nn.Parameter(torch.randn(n_input))

        # Encoder（固定条件拼接）：q(z|x,y,b)
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

        # 新增：log-library encoder 头 q(l|x)（轻量 MLP，输入仅用 x_ 即可）
        # 注：为尽量少侵入，不依赖自定义 Encoder；保持简单稳定
        self.l_encoder_backbone = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.l_mu = nn.Linear(n_hidden, 1)
        self.l_logvar = nn.Linear(n_hidden, 1)

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
        # 重要：输出概率域 [0,1]，用 Sigmoid（而不是 Softplus）
        self.px_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_input),
            nn.Sigmoid(),
        )

        # 细胞类型权重
        if ct_weight is None:
            ct_weight = torch.ones(n_labels)
        else:
            ct_weight = torch.tensor(ct_weight, dtype=torch.float32)
        self.register_buffer("ct_weight", ct_weight)

        # library 先验（标量，计算时会 broadcast）
        self.register_buffer("library_prior_mean", torch.tensor(library_prior_mean, dtype=torch.float32))
        self.register_buffer("library_prior_logvar", torch.tensor(library_prior_logvar, dtype=torch.float32))

    def _one_hot_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).long()
        return F.one_hot(labels, num_classes=self.n_labels).float()

    def _one_hot_batch(self, batch_index: torch.Tensor | None) -> torch.Tensor | None:
        if self.n_batch <= 0 or batch_index is None:
            return None
        b = batch_index.view(-1).long()
        return F.one_hot(b, num_classes=self.n_batch).float()

    def inference(self, x, labels, batch_index=None):
        # q(z|x,y,b)
        qz, z = self.z_encoder(x, labels, batch_index)

        # q(l|x)，在 log 变换域
        x_ = x
        if self.log_variational:
            x_ = torch.log1p(torch.clamp_min(x, 0.0))
        h_l = self.l_encoder_backbone(x_)
        l_mu = self.l_mu(h_l)
        l_logvar = self.l_logvar(h_l)
        l_std = torch.exp(0.5 * l_logvar)
        eps = torch.randn_like(l_std)
        log_library = l_mu + l_std * eps
        library = torch.exp(log_library)

        return {
            "z": z,
            "qz": qz,
            "qz_m": qz.loc,
            "qz_v": qz.scale.pow(2),
            "log_library": log_library,
            "l_mu": l_mu,
            "l_logvar": l_logvar,
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
        px_scale = self.px_decoder(h)   # [B, n_input], in (0,1)
        px_rate = library * px_scale    # broadcast
        return {"px_rate": px_rate, "px_scale": px_scale}

    def forward(self, item, kl_weight: float = 1.0, kl_library_weight: float = 1.0, n_samples: int = 1):
        x = item["X"]
        labels = item["labels"]
        batch_index = item.get("batch", None)

        inf = self.inference(x, labels, batch_index)
        gen = self.generative(inf["z"], inf["library"], labels, batch_index)

        # 重构项（不要再 clamp px_rate；概率域和 library 推断正确后无需）
        px_rate = gen["px_rate"]
        reconst_loss = -NegativeBinomial(px_rate, logits=self.px_r).log_prob(x).sum(-1)  # [B]

        # KL(q(z)|p(z))，标准 N(0,1)
        pz = Normal(torch.zeros_like(inf["qz_m"]), torch.ones_like(inf["qz_v"]))
        qz_d = Normal(inf["qz_m"], torch.sqrt(inf["qz_v"]))
        kl_z = kl_divergence(qz_d, pz).sum(-1)  # [B]

        # KL(q(l)|p(l))，在 log-library 域
        ql = Normal(inf["l_mu"], torch.exp(0.5 * inf["l_logvar"]))
        pl = Normal(
            torch.full_like(inf["l_mu"], self.library_prior_mean),
            torch.exp(0.5 * torch.full_like(inf["l_logvar"], self.library_prior_logvar)),
        )
        kl_library = kl_divergence(ql, pl).sum(-1)  # [B]，此处维度本为 1，sum 以保持一致

        # 按细胞类型权重缩放
        scaling = self.ct_weight[labels.view(-1).long()]
        loss = torch.mean(scaling * (reconst_loss + kl_weight * kl_z + kl_library_weight * kl_library))

        return {
            "loss": loss,
            "reconstruction_loss": reconst_loss.mean(),
            "kl_local": kl_z.mean(),
            "kl_library": kl_library.mean(),
        }