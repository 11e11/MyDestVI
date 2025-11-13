"""
MRDeconv 模块 - 固定条件版（手动拼接 one-hot 标签）
- 去掉对 FCLayers 的 n_cat_list/inject_covariates 依赖
- decoder 的输入为 [latent, onehot(label)]
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, Dirichlet
import torch.nn.functional as F

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
        **kwargs,
    ):
        super().__init__()

        self.n_spots = n_spots
        self.n_labels = n_labels
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.amortization = amortization
        self.l1_reg = l1_reg
        self.beta_reg = beta_reg
        self.eta_reg = eta_reg
        self.dirichlet_mmd_reg = dirichlet_mmd_reg
        self.use_gat = use_gat
        self.gat_hidden = gat_hidden

        # 解码器（固定条件：输入维度 = n_latent + n_labels）
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
            print(f"[WARN] decoder_state_dict is None (type={type(decoder_state_dict)}). "
                  f"Ensure DestVI.from_rna_model extracted decoder_backbone.state_dict().")

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

        self.register_buffer("px_r", torch.tensor(px_r, dtype=torch.float32))

        # 可学习参数
        self.V = nn.Parameter(torch.randn(n_labels + 1, n_spots))
        self.gamma = nn.Parameter(torch.randn(n_latent, n_labels, n_spots))
        self.eta = nn.Parameter(torch.randn(n_genes))
        self.beta = nn.Parameter(0.01 * torch.randn(n_genes))

        # VampPrior
        if mean_vprior is not None:
            self.register_buffer("mean_vprior", torch.tensor(mean_vprior, dtype=torch.float32))
            self.register_buffer("var_vprior", torch.tensor(var_vprior, dtype=torch.float32))
            self.register_buffer("mp_vprior", torch.tensor(mp_vprior, dtype=torch.float32))
        else:
            self.mean_vprior = None

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
            from torch_geometric.nn import GINEConv

            def mlp_block(in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, gat_hidden),
                    nn.BatchNorm1d(gat_hidden),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(gat_hidden, out_dim),
                )

            self.gamma_gat_layers = nn.ModuleList([GINEConv(mlp_block(n_genes, gat_hidden), train_eps=True, edge_dim=1)])
            self.gamma_gat_linear = nn.Linear(gat_hidden, n_latent * n_labels)
            self.gamma_ln = nn.LayerNorm(gat_hidden)

            self.V_gat_layers = nn.ModuleList([GINEConv(mlp_block(n_genes, gat_hidden), train_eps=True, edge_dim=1)])
            self.V_gat_linear = nn.Linear(gat_hidden, n_labels + 1)
            self.V_ln = nn.LayerNorm(gat_hidden)

            self.register_buffer("_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
            self.register_buffer("_edge_weight", torch.empty(0, dtype=torch.float32), persistent=False)

        # Dirichlet
        if dirichlet_alpha is None:
            self.dirichlet_alpha = torch.ones(n_labels)
        else:
            alpha = torch.tensor(dirichlet_alpha, dtype=torch.float32)
            if alpha.numel() == 1:
                self.dirichlet_alpha = alpha.repeat(n_labels)
            else:
                self.dirichlet_alpha = alpha

    def generative(self, x, ind_x, batch_index=None):
        """
        构建生成图：固定条件版
        """
        m = x.shape[0]
        library = x.sum(1, keepdim=True)
        beta = torch.exp(self.beta)
        eps = F.softplus(self.eta)

        x_ = torch.log1p(torch.clamp(x, min=0.0))

        # gamma, v
        if self.use_gat:
            if not hasattr(self, "_X_all"):
                raise RuntimeError("use_gat=True 需要先 attach_full_X")
            if self._edge_index.numel() == 0:
                raise RuntimeError("use_gat=True 需要先 attach_graph")

            X_all = torch.clamp_min(self._X_all.to(x.device, dtype=torch.float32), 0.0)
            X_all_log = torch.log1p(X_all)
            edge_index = self._edge_index.to(x.device)
            edge_attr = getattr(self, "_edge_weight", None)
            if edge_attr is not None and edge_attr.numel() > 0:
                edge_attr = edge_attr.to(x.device).unsqueeze(-1)

            h = self.gamma_gat_layers[0](X_all_log, edge_index, edge_attr=edge_attr)
            h = self.gamma_ln(h)
            h = F.elu(h)
            gamma_raw_all = self.gamma_gat_linear(h)
            gamma_mb = gamma_raw_all[ind_x, :]
            gamma_ind = gamma_mb.view(m, self.n_labels, self.n_latent).permute(2, 1, 0)

            h2 = self.V_gat_layers[0](X_all_log, edge_index, edge_attr=edge_attr)
            h2 = self.V_ln(h2)
            h2 = F.elu(h2)
            v_raw_all = self.V_gat_linear(h2)
            v_ind = v_raw_all[ind_x, :]
        else:
            if self.amortization in ["both", "latent"]:
                gamma_ind = torch.transpose(self.gamma_encoder(x_), 0, 1).reshape((self.n_latent, self.n_labels, -1))
            else:
                gamma_ind = self.gamma[:, :, ind_x]

            if self.amortization in ["both", "proportion"]:
                v_ind = self.V_encoder(x_)
            else:
                v_ind = self.V[:, ind_x].T

        v_ind = F.softplus(v_ind)  # [m, n_labels+1]

        # 通过 CondSCVI 冻结的 decoder 生成每个 celltype 的基因尺度
        gamma_ind = torch.transpose(gamma_ind, 2, 0)  # [m, n_labels, n_latent]
        gamma_reshape = gamma_ind.reshape((-1, self.n_latent))  # [m*n_labels, n_latent]

        # 手动 one-hot label 并拼接
        enum_label = torch.arange(0, self.n_labels, device=x.device).repeat(m)  # [m*n_labels]
        label_oh = F.one_hot(enum_label.long(), num_classes=self.n_labels).float()  # [m*n_labels, n_labels]
        dec_in = torch.cat([gamma_reshape, label_oh], dim=1)  # [m*n_labels, n_latent+n_labels]

        h_dec = self.decoder(dec_in)
        px_rate = self.px_decoder(h_dec).reshape((m, self.n_labels, -1))  # [m, n_labels, n_genes]

        # 合成 dummy + celltype
        eps = eps.repeat((m, 1)).view(m, 1, -1)
        r_hat = torch.cat([beta.unsqueeze(0).unsqueeze(1) * px_rate, eps], dim=1)  # [m, n_labels+1, n_genes]
        px_scale = torch.sum(v_ind.unsqueeze(2) * r_hat, dim=1)  # [m, n_genes]
        px_rate_final = library * px_scale

        return {
            "px_rate": px_rate_final,
            "px_scale": px_scale,
            "gamma": gamma_ind,  # [m, n_labels, n_latent]
            "v": v_ind,
        }

    def forward(self, item, kl_weight=1.0, n_obs=1.0):
        x = item["X"]
        ind_x = item["ind_x"]
        batch_index = item.get("batch", None)

        outputs = self.generative(x, ind_x, batch_index)
        px_rate = outputs["px_rate"]
        gamma = outputs["gamma"]
        v = outputs["v"]

        reconst_loss = -NegativeBinomial(px_rate, logits=self.px_r).log_prob(x).sum(-1)

        mean = torch.zeros_like(self.eta)
        scale = torch.ones_like(self.eta)
        glo_neg_log_likelihood_prior = -self.eta_reg * Normal(mean, scale).log_prob(self.eta).sum()
        var_beta = torch.mean(self.beta ** 2) - torch.mean(self.beta) ** 2
        glo_neg_log_likelihood_prior += self.beta_reg * var_beta

        v_sparsity_loss = self.l1_reg * torch.abs(v).mean(1)

        # gamma 先验（VampPrior 混合）
        if self.mean_vprior is None:
            mean = torch.zeros_like(gamma)
            scale = torch.ones_like(gamma)
            neg_log_likelihood_prior = -Normal(mean, scale).log_prob(gamma).sum(2).sum(1)
        else:
            gamma_expanded = gamma.unsqueeze(1)  # [m,1,n_labels,n_latent]
            mean_vprior = self.mean_vprior
            var_vprior = self.var_vprior
            mp_vprior = self.mp_vprior
            if mean_vprior.dim() == 2:
                mean_vprior = mean_vprior.unsqueeze(1)
            if var_vprior.dim() == 2:
                var_vprior = var_vprior.unsqueeze(1)
            if mp_vprior.dim() == 1:
                mp_vprior = mp_vprior.unsqueeze(1)

            mean_vprior = torch.transpose(mean_vprior, 0, 1).unsqueeze(0)  # [1,p,n_labels,n_latent]
            var_vprior = torch.transpose(var_vprior, 0, 1).unsqueeze(0)
            mp_vprior = torch.transpose(mp_vprior, 0, 1)  # [p,n_labels]

            # 将 gamma 的 batch 维挪到前面，便于广播
            gamma_perm = gamma.permute(0, 1, 2)  # [m, n_labels, n_latent]
            gamma_perm_exp = gamma_perm.unsqueeze(0).unsqueeze(1)  # [1,1,m,n_labels,n_latent]
            mean_prior_exp = mean_vprior.unsqueeze(2)  # [1,p,1,n_labels,n_latent]
            var_prior_exp = (var_vprior + 1e-4).unsqueeze(2)

            log_prob = Normal(mean_prior_exp, torch.sqrt(var_prior_exp)).log_prob(gamma_perm_exp).sum(-1)  # [1,p,m,n_labels]
            pre_lse = log_prob + torch.log(mp_vprior.clamp_min(1e-12)).unsqueeze(0).unsqueeze(2)  # [1,p,m,n_labels]
            log_likelihood_prior = torch.logsumexp(pre_lse, dim=1).sum(-1)  # [1,m]
            neg_log_likelihood_prior = -log_likelihood_prior.squeeze(0)  # [m]

        # Dirichlet MMD（可选）
        mmd_term = torch.tensor(0.0, device=px_rate.device)
        if self.dirichlet_mmd_reg > 0.0:
            v_real = v[:, : self.n_labels]
            proportions = v_real / (v_real.sum(dim=1, keepdim=True) + 1e-8)
            alpha = self.dirichlet_alpha.to(proportions.device)
            dirich = Dirichlet(alpha)
            theta_prior = dirich.sample((proportions.size(0),)).to(proportions.device)
            mmd_term = self._mmd_rbf(proportions, theta_prior)

        sample_term = reconst_loss + kl_weight * (neg_log_likelihood_prior + v_sparsity_loss)
        loss = n_obs * (torch.mean(sample_term) + glo_neg_log_likelihood_prior + self.dirichlet_mmd_reg * mmd_term)

        return {
            "loss": loss,
            "reconstruction_loss": reconst_loss.mean(),
            "kl_local": neg_log_likelihood_prior.mean(),
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
    def get_proportions(self, x=None, keep_noise=False):
        if self.amortization in ["both", "proportion"]:
            if self.use_gat:
                if not hasattr(self, "_X_all"):
                    raise RuntimeError("需要attach_full_X")
                if self._edge_index.numel() == 0:
                    raise RuntimeError("需要attach_graph")

                device = self.px_r.device
                X_all = torch.clamp_min(self._X_all.to(device, dtype=torch.float32), 0.0)
                X_all_log = torch.log1p(X_all)
                edge_index = self._edge_index.to(device)
                edge_attr = getattr(self, "_edge_weight", None)
                if edge_attr is not None and edge_attr.numel() > 0:
                    edge_attr = edge_attr.to(device).unsqueeze(-1)

                h = self.V_gat_layers[0](X_all_log, edge_index, edge_attr=edge_attr)
                h = self.V_ln(h)
                h = F.elu(h)
                res = F.softplus(self.V_gat_linear(h))
            else:
                if x is None:
                    raise ValueError("FC模式需要传入x")
                x_ = torch.log1p(x)
                res = F.softplus(self.V_encoder(x_))
        else:
            res = F.softplus(self.V)
            if res.dim() == 2 and res.shape[0] == self.n_labels + 1:
                res = res.T

        if not keep_noise:
            res = res[:, :-1]
        res = res / (res.sum(axis=1, keepdims=True) + 1e-8)
        return res

    @torch.no_grad()
    def get_gamma(self, x=None):
        if self.amortization in ["latent", "both"]:
            if self.use_gat:
                if not hasattr(self, "_X_all"):
                    raise RuntimeError("需要attach_full_X")
                if self._edge_index.numel() == 0:
                    raise RuntimeError("需要attach_graph")

                device = self.px_r.device
                X_all = torch.clamp_min(self._X_all.to(device, dtype=torch.float32), 0.0)
                X_all_log = torch.log1p(X_all)
                edge_index = self._edge_index.to(device)
                edge_attr = getattr(self, "_edge_weight", None)
                if edge_attr is not None and edge_attr.numel() > 0:
                    edge_attr = edge_attr.to(device).unsqueeze(-1)

                h = self.gamma_gat_layers[0](X_all_log, edge_index, edge_attr=edge_attr)
                h = self.gamma_ln(h)
                h = F.elu(h)
                gamma_raw_all = self.gamma_gat_linear(h)
                N = gamma_raw_all.size(0)
                gamma = gamma_raw_all.view(N, self.n_labels, self.n_latent).permute(2, 1, 0)
                return gamma.cpu().numpy()
            else:
                if x is None:
                    raise ValueError("FC模式需要传入x")
                x_ = torch.log1p(x)
                gamma = self.gamma_encoder(x_)
                return torch.transpose(gamma, 0, 1).reshape((self.n_latent, self.n_labels, -1)).cpu().numpy()
        else:
            return self.gamma.cpu().numpy()

    @torch.no_grad()
    def get_ct_specific_expression(self, x, ind_x, y):
        beta = torch.exp(self.beta)
        y_torch = (y * torch.ones_like(ind_x)).ravel().long()

        if self.amortization in ["both", "latent"]:
            x_ = torch.log1p(x)
            gamma_ind = torch.transpose(self.gamma_encoder(x_), 0, 1).reshape((self.n_latent, self.n_labels, -1))
        else:
            gamma_ind = self.gamma[:, :, ind_x]

        gamma_select = gamma_ind[:, y_torch, torch.arange(ind_x.shape[0])].T  # [m, n_latent]
        label_oh = F.one_hot(y_torch, num_classes=self.n_labels).float()      # [m, n_labels]
        dec_in = torch.cat([gamma_select, label_oh], dim=1)                   # [m, n_latent+n_labels]

        h = self.decoder(dec_in)
        px_scale = self.px_decoder(h)
        px_ct = torch.exp(self.px_r).unsqueeze(0) * beta.unsqueeze(0) * px_scale
        return px_ct

    def attach_graph(self, adata=None, edge_index=None, edge_weight=None, k=6, spatial_key="spatial"):
        if edge_index is not None:
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.as_tensor(edge_index, dtype=torch.long)
            self.register_buffer("_edge_index", edge_index, persistent=False)

            if edge_weight is not None:
                if not isinstance(edge_weight, torch.Tensor):
                    edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32)
                self.register_buffer("_edge_weight", edge_weight, persistent=False)
            else:
                self.register_buffer("_edge_weight", torch.ones(edge_index.shape[1], dtype=torch.float32), persistent=False)
            return self

        if adata is None:
            raise ValueError("需要adata或edge_index")

        coords = None
        if spatial_key in adata.obsm:
            coords = adata.obsm[spatial_key]
        elif "spatial" in adata.obsm:
            coords = adata.obsm["spatial"]
        else:
            raise ValueError("未找到空间坐标")

        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]), algorithm="auto").fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        src = []
        dst = []
        for i in range(indices.shape[0]):
            neigh = indices[i, 1 : min(k + 1, indices.shape[1])]
            for j in neigh:
                src.append(i)
                dst.append(int(j))

        edge_index = torch.as_tensor([src, dst], dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

        self.register_buffer("_edge_index", edge_index, persistent=False)
        self.register_buffer("_edge_weight", edge_weight, persistent=False)
        return self

    def attach_full_X(self, adata, layer=None):
        import numpy as np
        from scipy.sparse import issparse

        if layer is not None and layer in getattr(adata, "layers", {}):
            X = adata.layers[layer]
        else:
            X = adata.X

        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        self.register_buffer("_X_all", torch.from_numpy(X), persistent=False)
        return self