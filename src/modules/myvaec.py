"""
VAEC模块 - 简化版
主要改动：删除 _get_inference_input, _get_generative_input
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

# 建议直接从子模块导入，进一步减少耦合
from src.nn.encoder import Encoder
from src.nn.layers import FCLayers
from src.distributions.negative_binomial import NegativeBinomial


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
        encode_covariates: bool = False,
    ):
        super().__init__()

        self.n_latent = n_latent
        self.n_labels = n_labels
        self.n_batch = n_batch
        self.dropout_rate = dropout_rate
        self.encode_covariates = encode_covariates

        # NB 参数（dispersion 的 logits）
        self.px_r = nn.Parameter(torch.zeros(n_input))  # 更稳健的初始化

        # 编码器
        encoder_cat_list = [n_labels]
        if n_batch > 0 and encode_covariates:
            encoder_cat_list.append(n_batch)

        self.z_encoder = Encoder(
            n_input=n_input,
            n_latent=n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            return_dist=True,  # 直接返回 Normal 分布
        )

        # 解码器
        decoder_cat_list = [n_labels]
        if n_batch > 0:
            decoder_cat_list.append(n_batch)

        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=decoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
        )

        self.px_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_input),
            nn.Softplus()
        )

        # 细胞类型权重
        if ct_weight is None:
            ct_weight = torch.ones(n_labels)
        else:
            ct_weight = torch.tensor(ct_weight, dtype=torch.float32)
        self.register_buffer('ct_weight', ct_weight)

    @staticmethod
    def _to_1d_long(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        return t.long()

    def inference(self, x, labels, batch_index=None):
        library = x.sum(1, keepdim=True)
        x_ = torch.log1p(x)

        labels_1d = self._to_1d_long(labels)
        batch_1d = self._to_1d_long(batch_index) if self.encode_covariates else None

        cat_list = [labels_1d] if labels_1d is not None else []
        if batch_1d is not None:
            cat_list.append(batch_1d)

        if len(cat_list) == 0:
            enc_out = self.z_encoder(x_)
        else:
            enc_out = self.z_encoder(x_, cat_list)

        # 下方保持你当前的分支逻辑（Normal / (mu, logvar) 等）
        if isinstance(enc_out, Normal):
            qz = enc_out
            z = qz.rsample()
        elif isinstance(enc_out, tuple) and len(enc_out) == 2:
            if isinstance(enc_out[0], Normal):
                qz, z = enc_out
            else:
                mu, logvar_or_var = enc_out
                if (logvar_or_var < 0).any():
                    std = (logvar_or_var * 0.5).exp()
                else:
                    std = logvar_or_var.clamp_min(1e-8).sqrt()
                qz = Normal(mu, std)
                z = qz.rsample()
        else:
            raise TypeError("z_encoder 返回格式不对，期待 Normal 或 (mu,logvar) 或 (qz,z)")

        return {
            'z': z,
            'qz': qz,
            'qz_m': qz.loc,
            'qz_v': qz.scale.pow(2),
            'library': library,
        }

    def generative(self, z, library, labels, batch_index=None):
        labels_1d = self._to_1d_long(labels)
        batch_1d = self._to_1d_long(batch_index)

        cat_list = [labels_1d] if labels_1d is not None else []
        if batch_1d is not None:
            cat_list.append(batch_1d)

        h = self.decoder(z, cat_list if len(cat_list) > 0 else None)
        px_scale = self.px_decoder(h)
        px_rate = library * px_scale
        return {'px_rate': px_rate}

    def forward(self, item, kl_weight=1.0):
        x = item['X']
        labels = item['labels']
        batch_index = item.get('batch', None)

        inference_outputs = self.inference(x, labels, batch_index)
        generative_outputs = self.generative(
            inference_outputs['z'],
            inference_outputs['library'],
            labels,
            batch_index
        )

        px_rate = torch.clamp(generative_outputs['px_rate'], min=1e-8)
        qz_m = inference_outputs['qz_m']
        qz_v = inference_outputs['qz_v']

        reconst_loss = -NegativeBinomial(px_rate, logits=self.px_r).log_prob(x).sum(-1)
        pz = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        qz = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence_z = kl_divergence(qz, pz).sum(dim=1)

        # 注意：labels 可能是 [B]，直接用于索引即可
        labels_1d = self._to_1d_long(labels)
        scaling_factor = self.ct_weight[labels_1d]

        loss = torch.mean(scaling_factor * (reconst_loss + kl_weight * kl_divergence_z))
        return {
            'loss': loss,
            'reconstruction_loss': reconst_loss.mean(),
            'kl_local': kl_divergence_z.mean(),
        }