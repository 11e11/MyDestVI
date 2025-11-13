"""
负二项分布实现 - 替代 scvi.distributions.NegativeBinomial

兼容说明：
- 接受调用样式： NegativeBinomial(mu, logits=...) 或 NegativeBinomial(mu, theta=...)
- mu 必须提供（位置或关键字），logits 会被解释为 log(theta) 并转为 theta=exp(logits)
- 若 theta 为 1D (n_genes,) 且 mu 维度 >=2 且 mu.shape[-1] == n_genes，
  则自动把 theta reshape 为 (1, ..., n_genes) 以便正确广播。
- 提供 log_prob(value) 和 sample(...)。
"""
from __future__ import annotations

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from typing import Optional

class NegativeBinomial(Distribution):
    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(self, mu: Optional[torch.Tensor] = None, theta: Optional[torch.Tensor] = None, logits: Optional[torch.Tensor] = None, validate_args=None):
        if mu is None:
            raise ValueError("NegativeBinomial requires mu (px_rate) as first argument.")

        # 处理 logits -> theta 的转换（兼容 scvi 常见用法）
        if logits is not None:
            if not torch.is_tensor(logits):
                logits = torch.as_tensor(logits, dtype=torch.float32)
            theta_val = torch.exp(logits)
        elif theta is not None:
            theta_val = theta if torch.is_tensor(theta) else torch.as_tensor(theta, dtype=torch.float32)
        else:
            theta_val = torch.tensor(1.0, dtype=torch.float32)

        # 确保 mu tensor
        mu_val = mu if torch.is_tensor(mu) else torch.as_tensor(mu, dtype=torch.float32)

        # 如果 theta 是 1D 且看起来是 per-gene（theta.shape[-1] == mu.shape[-1]），
        # 就把 theta reshape 成 (1,..., n_genes) 以便能被 broadcast 到 mu 的 shape。
        try:
            if torch.is_tensor(theta_val) and theta_val.dim() == 1 and mu_val.dim() >= 2:
                if theta_val.shape[-1] == mu_val.shape[-1]:
                    # new_shape e.g. for mu.ndim==3 -> [1,1,n_genes]
                    new_shape = [1] * (mu_val.dim() - 1) + [theta_val.shape[-1]]
                    theta_val = theta_val.view(*new_shape)
        except Exception:
            # 若检查过程中出错，不中断，后续 broadcast_all 再尝试
            pass

        # 广播 mu 和 theta 到相同形状（若可能）
        try:
            self.mu, self.theta = broadcast_all(mu_val, theta_val)
        except Exception:
            # 若 broadcast 失败，保留原始并在计算时依赖广播
            self.mu = mu_val
            self.theta = theta_val

        super().__init__(self.mu.shape, validate_args=validate_args)
        self._eps = 1e-8

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        x = value.to(dtype=torch.float32)
        mu = self.mu
        theta = self.theta

        # 如果需要，扩展 theta
        try:
            theta = theta.expand_as(mu)
        except Exception:
            pass

        mu = torch.clamp(mu, min=self._eps)
        theta = torch.clamp(theta, min=self._eps)

        log_gamma_x_theta = torch.lgamma(x + theta)
        log_gamma_theta = torch.lgamma(theta)
        log_factorial_x = torch.lgamma(x + 1.0)

        log_theta_term = theta * (torch.log(theta) - torch.log(theta + mu))
        log_mu_term = x * (torch.log(mu) - torch.log(theta + mu))

        return log_gamma_x_theta - log_gamma_theta - log_factorial_x + log_theta_term + log_mu_term

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            mu = self.mu
            theta = self.theta

            try:
                theta = theta.expand_as(mu)
            except Exception:
                pass

            mu = mu.clamp(min=self._eps)
            theta = theta.clamp(min=self._eps)

            gamma_d = torch.distributions.Gamma(theta, theta / (mu + self._eps))
            lambda_ = gamma_d.sample(sample_shape)
            poisson_d = torch.distributions.Poisson(lambda_)
            return poisson_d.sample()