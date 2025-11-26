"""
MRDeconv 模块 - 固定条件版（手动拼接 one-hot 标签）+ 训练日程退火
- decoder 输入 = [latent, onehot(label)]
- 仅当传入 dict-like state_dict 才加载权重
- forward 支持 kl_weight 与 reg_warmup（L1/MMD 退火）
- VampPrior 混合先验分支做形状与数值健壮处理
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, Dirichlet
import torch.nn.functional as F
import numpy as np

from src.nn import FCLayers
from src.distributions.negative_binomial import NegativeBinomial


class MRDeconv(nn.Module):
    def __init__(
        self,
        n_spots: int,
        n_labels: int,
        n_genes: int,
        n_latent: int,
        n_hidden: int,
        n_layers: int,
        decoder_state_dict,
        px_decoder_state_dict,
        px_r,
        dropout_decoder: float = 0.05,
        dropout_amortization: float = 0.05,
        mean_vprior=None,
        var_vprior=None,
        mp_vprior=None,
        amortization: str = "both",
        l1_reg: float = 0.0,
        beta_reg: float = 5.0,
        eta_reg: float = 1e-4,
        dirichlet_alpha=None,
        dirichlet_mmd_reg: float = 0.0,
        use_gat: bool = False,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        use_dual_graph: bool = False,           # 双图
        dual_graph_fusion: str = "concat",      # 双图融合方式
        **kwargs,
    ):
        super().__init__()

        self.n_spots = n_spots
        self.n_labels = n_labels
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.amortization = amortization

        # 保存正则化的基值（用于退火）
        self.base_l1_reg = float(l1_reg)
        self.base_dirichlet_mmd_reg = float(dirichlet_mmd_reg)
        self.beta_reg = float(beta_reg)
        self.eta_reg = float(eta_reg)

        self.use_gat = use_gat
        self.use_dual_graph = use_dual_graph
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads

        # 冻结的解码器：输入维度 = n_latent + n_labels
        self.decoder = FCLayers(
            n_in=n_latent + n_labels,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_decoder,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        if isinstance(decoder_state_dict, dict):
            try:
                self.decoder.load_state_dict(decoder_state_dict)
            except Exception as e:
                print(f"[WARN] Loading decoder_state_dict failed: {e}")
        else:
            print(f"[WARN] decoder_state_dict is None (type={type(decoder_state_dict)}).")

        for p in self.decoder.parameters():
            p.requires_grad = False

        self.px_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_genes),
            nn.Softplus(),
        )
        if isinstance(px_decoder_state_dict, dict):
            try:
                self.px_decoder.load_state_dict(px_decoder_state_dict)
            except Exception as e:
                print(f"[WARN] Loading px_decoder_state_dict failed: {e}")
        else:
            print(f"[WARN] px_decoder_state_dict is None (type={type(px_decoder_state_dict)}).")

        for p in self.px_decoder.parameters():
            p.requires_grad = False

        # NB logits
        self.register_buffer("px_r", torch.tensor(px_r, dtype=torch.float32))

        # 可学习参数
        self.V = nn.Parameter(torch.randn(n_labels + 1, n_spots))
        self.gamma = nn.Parameter(torch.randn(n_latent, n_labels, n_spots))
        self.eta = nn.Parameter(torch.randn(n_genes))
        self.beta = nn.Parameter(0.01 * torch.randn(n_genes))

        # VampPrior
        if mean_vprior is not None and var_vprior is not None and mp_vprior is not None:
            self.register_buffer("mean_vprior", torch.tensor(mean_vprior, dtype=torch.float32))
            self.register_buffer("var_vprior", torch.tensor(var_vprior, dtype=torch.float32))
            self.register_buffer("mp_vprior", torch.tensor(mp_vprior, dtype=torch.float32))
        else:
            self.mean_vprior = None  # 触发标准正态先验

        # 摊销网络（非 GAT）
        if not use_gat:
            self.gamma_encoder = nn.Sequential(
                FCLayers(
                    n_in=n_genes,
                    n_out=n_hidden,
                    n_layers=2,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_amortization,
                    use_batch_norm=False,
                    use_layer_norm=True,
                ),
                nn.Linear(n_hidden, n_latent * n_labels),
            )
            self.V_encoder = nn.Sequential(
                FCLayers(
                    n_in=n_genes,
                    n_out=n_hidden,
                    n_layers=2,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_amortization,
                    use_batch_norm=False,
                    use_layer_norm=True,
                ),
                nn.Linear(n_hidden, n_labels + 1),
            )
        else:
            # === GAT/GNN 模式 ===
            from torch_geometric.nn import GINEConv
            
            if not use_dual_graph:
                # --- 单图模式（你可以后续实现，现在先报错）---
                raise NotImplementedError("use_gat=True 且 use_dual_graph=False 暂未实现，请设置 use_dual_graph=True")
            
            else:
                # === 双图模式 ===
                hidden = gat_hidden
                in_ch = n_genes
                
                def mlp_block(in_dim, out_dim):
                    return nn.Sequential(
                        nn.Linear(in_dim, hidden),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Dropout(p=0.1),
                        nn.Linear(hidden, out_dim),
                    )
                
                # gamma 分支的双图编码器
                self.gamma_spatial_gnn = GINEConv(mlp_block(in_ch, hidden), train_eps=True, edge_dim=4)
                self.gamma_expr_gnn = GINEConv(mlp_block(in_ch, hidden), train_eps=True, edge_dim=3)
                self.gamma_spatial_ln = nn.LayerNorm(hidden)
                self.gamma_expr_ln = nn.LayerNorm(hidden)
                
                # V 分支的双图编码器
                self.V_spatial_gnn = GINEConv(mlp_block(in_ch, hidden), train_eps=True, edge_dim=4)
                self.V_expr_gnn = GINEConv(mlp_block(in_ch, hidden), train_eps=True, edge_dim=3)
                self.V_spatial_ln = nn.LayerNorm(hidden)
                self.V_expr_ln = nn.LayerNorm(hidden)
                
                # 融合层（concat 模式）
                if dual_graph_fusion == "concat":
                    self.gamma_fusion = nn.Sequential(
                        nn.Linear(hidden * 2, hidden),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                    self.V_fusion = nn.Sequential(
                        nn.Linear(hidden * 2, hidden),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                else:
                    raise NotImplementedError(f"fusion={dual_graph_fusion} 未实现")
                
                # 输出层
                self.gamma_output = nn.Linear(hidden, n_latent * n_labels)
                self.V_output = nn.Linear(hidden, n_labels + 1)
                
                # 双图的 buffer（用于缓存图结构）
                self.register_buffer("_X_all", torch.empty(0), persistent=False)
                self.register_buffer("_spatial_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
                self.register_buffer("_spatial_edge_attr", torch.empty(0, dtype=torch.float32), persistent=False)
                self.register_buffer("_expr_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
                self.register_buffer("_expr_edge_attr", torch.empty(0, dtype=torch.float32), persistent=False)

        # Dirichlet 先验超参
        if dirichlet_alpha is None:
            self.dirichlet_alpha = torch.ones(n_labels)
        else:
            alpha = torch.tensor(dirichlet_alpha, dtype=torch.float32)
            self.dirichlet_alpha = alpha.repeat(n_labels) if alpha.numel() == 1 else alpha

    def _decode_ct(self, gamma_ind: torch.Tensor) -> torch.Tensor:
        """
        对每个 cell type 分别用冻结 decoder 生成 px_rate（未乘 library）
        gamma_ind: [m, n_labels, n_latent]
        return: [m, n_labels, n_genes]
        """
        m = gamma_ind.size(0)
        gamma_reshape = gamma_ind.reshape(-1, self.n_latent)  # [m*n_labels, n_latent]
        # one-hot label
        enum_label = torch.arange(0, self.n_labels, device=gamma_ind.device).repeat(m)
        label_oh = F.one_hot(enum_label.long(), num_classes=self.n_labels).float()  # [m*n_labels, n_labels]
        dec_in = torch.cat([gamma_reshape, label_oh], dim=1)  # [m*n_labels, n_latent+n_labels]
        h_dec = self.decoder(dec_in)
        px_rate = self.px_decoder(h_dec).reshape((m, self.n_labels, -1))
        return px_rate

    def _gamma_prior_nll(self, gamma: torch.Tensor) -> torch.Tensor:
        """
        计算 gamma 的先验负对数似然（按样本维度返回）。
        gamma: [m, n_labels, n_latent]
        return: [m]
        """
        if self.mean_vprior is None:
            mean = torch.zeros_like(gamma)
            scale = torch.ones_like(gamma)
            return -Normal(mean, scale).log_prob(gamma).sum(2).sum(1)

        # 形状健壮处理：期望 mean/var/mp 为 (n_labels, p, n_latent)/(n_labels, p)/(n_labels, p)
        mean_vprior = self.mean_vprior
        var_vprior = self.var_vprior
        mp_vprior = self.mp_vprior
        if mean_vprior.dim() == 2:
            mean_vprior = mean_vprior.unsqueeze(1)
        if var_vprior.dim() == 2:
            var_vprior = var_vprior.unsqueeze(1)
        if mp_vprior.dim() == 1:
            mp_vprior = mp_vprior.unsqueeze(1)

        # 转为 [1, p, n_labels, n_latent]
        mean_vprior = torch.transpose(mean_vprior, 0, 1).unsqueeze(0)
        var_vprior = torch.transpose(var_vprior, 0, 1).unsqueeze(0)
        mp_vprior = torch.transpose(mp_vprior, 0, 1)  # [p, n_labels]

        # 广播到 [1, p, m, n_labels, n_latent]
        gamma_exp = gamma.unsqueeze(0).unsqueeze(0)  # [1,1,m,n_labels,n_latent]
        mean_exp = mean_vprior.unsqueeze(2)
        var_exp = (var_vprior + 1e-4).unsqueeze(2)   # 数值稳定

        log_prob = Normal(mean_exp, torch.sqrt(var_exp)).log_prob(gamma_exp).sum(-1)  # [1,p,m,n_labels]
        pre_lse = log_prob + torch.log(mp_vprior.clamp_min(1e-12)).unsqueeze(0).unsqueeze(2)
        log_lik = torch.logsumexp(pre_lse, dim=1).sum(-1)  # [1,m]
        nll = -log_lik.squeeze(0)  # [m]
        return nll

    def generative(self, x: torch.Tensor, ind_x: torch.Tensor, x_encoder: torch.Tensor = None):
        """
        核心生成图（支持双图 + 双输入）
        x: raw counts for likelihood [m, n_genes]
        x_encoder: preprocessed features for encoder [m, n_genes] (可选)
        """
        m = x.shape[0]
        library = x.sum(1, keepdim=True)  # 🔥 用 raw counts 计算 library
        beta = torch.exp(self.beta)
        eps = F.softplus(self.eta)
        
        # 🔥 关键改动：如果提供了 x_encoder，用它；否则回退到 x
        # if x_encoder is None:
        #     x_encoder = x
        
        # x_ = torch.log1p(torch.clamp_min(x_encoder, 0.0))  # 🔥 用预处理后的特征

        if x_encoder is not None:
            x_ = x_encoder # ✅ 直接使用，不再 log
        else:
            x_ = torch.log1p(torch.clamp_min(x, 0.0))

        # === gamma / v 编码 ===
        if self.use_gat and self.use_dual_graph:
            # --- 双图模式 ---
            if not hasattr(self, "_X_all") or self._X_all.numel() == 0:
                raise RuntimeError("双图模式需要先调用 attach_full_X(adata)")
            if self._spatial_edge_index.numel() == 0 or self._expr_edge_index.numel() == 0:
                raise RuntimeError("双图模式需要先调用 attach_dual_graph(adata)")
            
            # 🔥 使用预处理后的全量特征（已在 attach_full_X 中缓存为 log-normalized）
            X_all_log = self._X_all.to(x.device)
            
            # 边权重
            spatial_edge_index = self._spatial_edge_index.to(x.device)
            spatial_attr = self._spatial_edge_attr.to(x.device)
            
            expr_edge_index = self._expr_edge_index.to(x.device)
            expr_attr = self._expr_edge_attr.to(x.device) if self._expr_edge_attr.numel() > 0 else None
            
            # === gamma 分支：双图编码 + 融合 ===
            h_s_gamma = self.gamma_spatial_gnn(X_all_log, spatial_edge_index, edge_attr=spatial_attr)
            h_s_gamma = self.gamma_spatial_ln(h_s_gamma)
            h_s_gamma = F.elu(h_s_gamma)

            h_e_gamma = self.gamma_expr_gnn(X_all_log, expr_edge_index, edge_attr=expr_attr)
            h_e_gamma = self.gamma_expr_ln(h_e_gamma)
            h_e_gamma = F.elu(h_e_gamma)

            h_gamma = torch.cat([h_s_gamma, h_e_gamma], dim=-1)
            h_gamma = self.gamma_fusion(h_gamma)
            gamma_all = self.gamma_output(h_gamma)

            gamma_mb = gamma_all[ind_x, :]
            gamma_ind = gamma_mb.view(m, self.n_labels, self.n_latent)
            gamma_ind = gamma_ind.permute(2, 1, 0)

            # === V 分支：双图编码 + 融合 ===
            h_s_v = self.V_spatial_gnn(X_all_log, spatial_edge_index, edge_attr=spatial_attr)
            h_s_v = self.V_spatial_ln(h_s_v)
            h_s_v = F.elu(h_s_v)

            h_e_v = self.V_expr_gnn(X_all_log, expr_edge_index, edge_attr=expr_attr)
            h_e_v = self.V_expr_ln(h_e_v)
            h_e_v = F.elu(h_e_v)

            h_v = torch.cat([h_s_v, h_e_v], dim=-1)
            h_v = self.V_fusion(h_v)
            v_all = self.V_output(h_v)

            v_ind = F.softplus(v_all[ind_x, :])
            # print("spat_attr.shape, dtype:", spatial_attr.shape, spatial_attr.dtype)
            # print("expr_attr.shape, dtype:", expr_attr.shape, expr_attr.dtype)
            # print("gamma_spatial_gnn.edge_dim:", self.gamma_spatial_gnn.edge_dim)
            # print("V_spatial_gnn.edge_dim:", self.V_spatial_gnn.edge_dim)

            
        else:
            # --- FC 模式（原逻辑）---
            if self.amortization in ["both", "latent"]:
                gamma_ind = torch.transpose(self.gamma_encoder(x_), 0, 1).reshape((self.n_latent, self.n_labels, -1))
            else:
                gamma_ind = self.gamma[:, :, ind_x]
            
            if self.amortization in ["both", "proportion"]:
                v_ind = F.softplus(self.V_encoder(x_))
            else:
                v_ind = F.softplus(self.V[:, ind_x].T)

        # 解码每个 celltype 的 rate（未乘 library）
        gamma_mlb = torch.transpose(gamma_ind, 2, 0)
        px_rate_ct = self._decode_ct(gamma_mlb)

        # 合成 dummy + 各类型
        eps_ct = eps.repeat((m, 1)).view(m, 1, -1)
        r_hat = torch.cat([beta.unsqueeze(0).unsqueeze(1) * px_rate_ct, eps_ct], dim=1)
        px_scale = torch.sum(v_ind.unsqueeze(2) * r_hat, dim=1)
        px_rate = library * px_scale  # 🔥 用 raw counts 的 library 重构

        return {
            "px_rate": px_rate,
            "px_scale": px_scale,
            "gamma": gamma_mlb,
            "v": v_ind,
        }

    def forward(self, item: dict, kl_weight: float = 1.0, n_obs: float = 1.0, reg_warmup: float = 1.0):
        """
        kl_weight: 对 gamma 先验项的退火权重（0→1）
        reg_warmup: 对 L1/MMD 的退火权重（0→1）
        """
        x = item["X"]
        x_encoder = item.get("X_encoder", None)  # 🔥 新增：预处理后的特征
        ind_x = item["ind_x"]

        outs = self.generative(x, ind_x, x_encoder=x_encoder)  # 🔥 传入 x_encoder
        px_rate = outs["px_rate"]
        gamma = outs["gamma"]
        v = outs["v"]

        # 重构
        reconst_loss = -NegativeBinomial(px_rate, logits=self.px_r).log_prob(x).sum(-1)  # [m]

        # 全局先验（eta/beta）
        mean_eta = torch.zeros_like(self.eta)
        scale_eta = torch.ones_like(self.eta)
        glo_neg_log_likelihood_prior = -self.eta_reg * Normal(mean_eta, scale_eta).log_prob(self.eta).sum()
        # 用 var(beta)
        glo_neg_log_likelihood_prior += self.beta_reg * torch.var(self.beta)

        # L1 稀疏（退火）
        effective_l1 = self.base_l1_reg * float(reg_warmup)
        v_sparsity = effective_l1 * torch.abs(v).mean(1)  # [m]

        # gamma 先验（退火）
        nll_gamma = self._gamma_prior_nll(gamma)          # [m]

        # MMD（退火）
        effective_mmd = self.base_dirichlet_mmd_reg * float(reg_warmup)
        mmd_term = torch.tensor(0.0, device=px_rate.device)
        if effective_mmd > 0.0:
            v_real = v[:, : self.n_labels]
            proportions = v_real / (v_real.sum(dim=1, keepdim=True) + 1e-8)
            alpha = self.dirichlet_alpha.to(proportions.device)
            dirich = Dirichlet(alpha)
            theta_prior = dirich.sample((proportions.size(0),)).to(proportions.device)
            mmd_term = self._mmd_rbf(proportions, theta_prior)

        # 汇总
        sample_term = reconst_loss + float(kl_weight) * (nll_gamma + v_sparsity)  # [m]
        loss = n_obs * (torch.mean(sample_term) + glo_neg_log_likelihood_prior + effective_mmd * mmd_term)

        return {
            "loss": loss,
            "reconstruction_loss": reconst_loss.mean(),
            "kl_local": nll_gamma.mean(),
            "kl_global": glo_neg_log_likelihood_prior,
            "mmd": mmd_term,
        }

    def _mmd_rbf(self, x, y, sigma=None):
        n = x.size(0)
        m = y.size(0)
        if n <= 1 or m <= 1:
            return torch.tensor(0.0, device=x.device)

        def pairwise_sq_dists(a, b):
            a_norm = (a ** 2).sum(dim=1).unsqueeze(1)
            b_norm = (b ** 2).sum(dim=1).unsqueeze(0)
            return a_norm + b_norm - 2.0 * a @ b.t()

        if sigma is None:
            comb = torch.cat([x, y], dim=0)
            d2 = pairwise_sq_dists(comb, comb)
            d2_flat = d2.view(-1)
            d2_pos = d2_flat[d2_flat > 0]
            sigma = (torch.sqrt(d2_pos.median()) + 1e-8).item() if d2_pos.numel() > 0 else 1.0

        Kxx = torch.exp(-pairwise_sq_dists(x, x) / (2.0 * sigma ** 2))
        Kyy = torch.exp(-pairwise_sq_dists(y, y) / (2.0 * sigma ** 2))
        Kxy = torch.exp(-pairwise_sq_dists(x, y) / (2.0 * sigma ** 2))

        sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
        sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
        sum_Kxy = Kxy.sum() / (n * m)

        return sum_Kxx + sum_Kyy - 2.0 * sum_Kxy


    @torch.no_grad()
    def get_proportions(self, x=None, x_encoder=None, keep_noise=False):
        was_training = self.training
        self.eval()

        if self.use_gat and self.use_dual_graph:
            # --- 双图模式 ---
            device = next(self.parameters()).device
            X_all_log = self._X_all.to(device)  # 已是预处理后的
            
            spatial_edge_index = self._spatial_edge_index.to(device)
            spatial_attr = self._spatial_edge_attr.to(device)
            expr_edge_index = self._expr_edge_index.to(device)
            expr_attr = self._expr_edge_attr.to(device) if self._expr_edge_attr.numel() > 0 else None
            
            h_s = self.V_spatial_gnn(X_all_log, spatial_edge_index, edge_attr=spatial_attr)
            h_s = self.V_spatial_ln(h_s)
            h_s = F.elu(h_s)
            
            h_e = self.V_expr_gnn(X_all_log, expr_edge_index, edge_attr=expr_attr)
            h_e = self.V_expr_ln(h_e)
            h_e = F.elu(h_e)
            
            h = torch.cat([h_s, h_e], dim=-1)
            h = self.V_fusion(h)
            res = F.softplus(self.V_output(h))
        
        elif self.amortization in ["both", "proportion"]:
            # --- FC 模式 ---
            if x_encoder is None:
                x_encoder = x
            if x_encoder is None:
                raise ValueError("FC 模式需要传入 x 或 x_encoder")
            x_ = torch.log1p(torch.clamp_min(x_encoder, 0.0))
            res = F.softplus(self.V_encoder(x_))
        else:
            res = F.softplus(self.V)
            if res.dim() == 2 and res.shape[0] == self.n_labels + 1:
                res = res.T

        if was_training:
            self.train()

        if not keep_noise:
            res = res[:, :-1]
        res = res / (res.sum(axis=1, keepdims=True) + 1e-8)
        return res

    @torch.no_grad()
    def get_gamma(self, x=None, x_encoder=None):
        if self.use_gat and self.use_dual_graph:
            # --- 双图模式 ---
            device = self.gamma_output.weight.device
            X_all_log = self._X_all.to(device)
            
            spatial_edge_index = self._spatial_edge_index.to(device)
            spatial_attr = self._spatial_edge_attr.to(device)
            expr_edge_index = self._expr_edge_index.to(device)
            expr_attr = self._expr_edge_attr.to(device) if self._expr_edge_attr.numel() > 0 else None

            h_s = self.gamma_spatial_gnn(X_all_log, spatial_edge_index, edge_attr=spatial_attr)
            h_s = self.gamma_spatial_ln(h_s)
            h_s = F.elu(h_s)

            h_e = self.gamma_expr_gnn(X_all_log, expr_edge_index, edge_attr=expr_attr)
            h_e = self.gamma_expr_ln(h_e)
            h_e = F.elu(h_e)

            h = torch.cat([h_s, h_e], dim=-1)
            h = self.gamma_fusion(h)
            gamma_all = self.gamma_output(h)
            N = gamma_all.size(0)
            gamma = gamma_all.view(N, self.n_labels, self.n_latent).permute(2, 1, 0)
            return gamma.cpu().numpy()
        
        elif self.amortization in ["latent", "both"]:
            # --- FC 模式 ---
            if x_encoder is None:
                x_encoder = x
            if x_encoder is None:
                raise ValueError("FC 模式需要传入 x 或 x_encoder")
            x_ = torch.log1p(torch.clamp_min(x_encoder, 0.0))
            gamma = self.gamma_encoder(x_)
            return torch.transpose(gamma, 0, 1).reshape((self.n_latent, self.n_labels, -1)).cpu().numpy()
        else:
            return self.gamma.cpu().numpy()
        

    def attach_full_X(self, adata, raw_layer=None, encoder_layer=None):
        """
        缓存全量表达矩阵（用于 GNN encoder）和原始 counts（用于 decoder）。
        
        参数：
        - raw_layer：原始 counts 所在的 layer（用于 NB likelihood）
        - encoder_layer：预处理好的特征（用于 encoder / GNN）
        """
        from scipy.sparse import issparse

        # -------------------------------------------------------------
        # 1. 获取原始 counts（始终保持用于 decoder，不做预处理）
        # -------------------------------------------------------------
        if raw_layer is not None and raw_layer in getattr(adata, "layers", {}):
            X_raw = adata.layers[raw_layer]
        else:
            X_raw = adata.X

        if issparse(X_raw):
            X_raw = X_raw.toarray()
        X_raw = np.asarray(X_raw, dtype=np.float32)

        # 缓存 raw counts（ST decoder reconstruction 用）
        self.register_buffer("_X_raw", torch.from_numpy(X_raw), persistent=False)


        # -------------------------------------------------------------
        # 2. 获取用于 encoder 的特征（如果没有，就对 raw 做 normalize+log1p）
        # -------------------------------------------------------------
        if encoder_layer is not None and encoder_layer in getattr(adata, "layers", {}):
            # 用户预处理好的特征
            X_enc = adata.layers[encoder_layer]

            if issparse(X_enc):
                X_enc = X_enc.toarray()
            X_enc = np.asarray(X_enc, dtype=np.float32)

            # ❗注意：用户自己处理好的 encoder_layer 不要再 normalize/log1p
            X_enc_processed = X_enc

            print(f"✅ 已加载 encoder_layer '{encoder_layer}'（不再做归一化/log）")

        else:
            # 默认：对 raw counts 做 normalize + log1p
            if issparse(X_raw):
                X_raw_np = X_raw.toarray()
            else:
                X_raw_np = X_raw

            X_raw_np = np.asarray(X_raw_np, dtype=np.float32)

            library = X_raw_np.sum(axis=1, keepdims=True)
            X_norm = X_raw_np / (library + 1e-6) * 10000
            X_enc_processed = np.log1p(X_norm)

            print(f"⚠ 未指定 encoder_layer，默认对 raw counts 做 normalize+log1p")


        # -------------------------------------------------------------
        # 3. 缓存 encoder 输入特征（给 GNN）
        # -------------------------------------------------------------
        self.register_buffer("_X_all", torch.from_numpy(X_enc_processed), persistent=False)
        print(f"✅ encoder 特征已缓存：{X_enc_processed.shape}")

        return self

    def attach_dual_graph(self, adata, k_spatial=6, k_expr=10, spatial_key='spatial'):
        """
        构建并注册双图结构（确保保存 edge_attr 而非 edge_weight）
        """
        if not getattr(self, 'use_dual_graph', False):
            raise RuntimeError("use_dual_graph=False，请在初始化时设置 use_dual_graph=True")
        
        from ..utils.dual_graph_builder import build_dual_graphs
        
        graphs = build_dual_graphs(adata, k_spatial=k_spatial, k_expr=k_expr, spatial_key=spatial_key)
        
        # graphs 应包含:
        # 'spatial_edge_index': torch.tensor([2, E_s])
        # 'spatial_edge_attr' : numpy/torch array shape (E_s, 4)
        # 'expr_edge_index'   : torch.tensor([2, E_e])
        # 'expr_edge_attr'    : numpy/torch array shape (E_e, 3)
        
        # convert and register consistently
        spat_ei = graphs['spatial_edge_index'].long()
        spat_eattr = graphs['spatial_edge_attr']
        expr_ei = graphs['expr_edge_index'].long()
        expr_eattr = graphs['expr_edge_attr']
        
        # if they are numpy arrays, convert
        if isinstance(spat_eattr, np.ndarray):
            spat_eattr = torch.from_numpy(spat_eattr.astype(np.float32))
        else:
            spat_eattr = spat_eattr.to(torch.float32)
        
        if isinstance(expr_eattr, np.ndarray):
            expr_eattr = torch.from_numpy(expr_eattr.astype(np.float32))
        else:
            expr_eattr = expr_eattr.to(torch.float32)
        
        # register buffers with consistent names
        self.register_buffer("_spatial_edge_index", spat_ei, persistent=False)
        self.register_buffer("_spatial_edge_attr", spat_eattr, persistent=False)
        self.register_buffer("_expr_edge_index", expr_ei, persistent=False)
        self.register_buffer("_expr_edge_attr", expr_eattr, persistent=False)
        
        # (Optional) print shapes
        print(f"✅ 双图已注册：spat edges={spat_ei.shape[1]}, spat_attr_dim={spat_eattr.shape[1]}; expr edges={expr_ei.shape[1]}, expr_attr_dim={expr_eattr.shape[1]}")
        return self
