"""
MRDeconv模块 - 完全修正版
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
        mean_vprior = None,
        var_vprior = None,
        mp_vprior = None,
        amortization: str = "both",
        l1_reg: float = 0.0,
        beta_reg: float = 5.0,
        eta_reg: float = 1e-4,
        dirichlet_alpha = None,
        dirichlet_mmd_reg: float = 0.0,
        use_gat: bool = False,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        **kwargs
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
        
        # 从CondSCVI继承的解码器（冻结）
        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=[n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_decoder,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        self.decoder.load_state_dict(decoder_state_dict)
        for p in self.decoder.parameters():
            p.requires_grad = False
        
        self.px_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_genes),
            nn.Softplus()
        )
        self.px_decoder.load_state_dict(px_decoder_state_dict)
        for p in self.px_decoder.parameters():
            p.requires_grad = False
        
        self.register_buffer('px_r', torch.tensor(px_r, dtype=torch.float32))
        
        # 可学习参数
        self.V = nn.Parameter(torch.randn(n_labels + 1, n_spots))
        self.gamma = nn.Parameter(torch.randn(n_latent, n_labels, n_spots))
        self.eta = nn.Parameter(torch.randn(n_genes))
        self.beta = nn.Parameter(0.01 * torch.randn(n_genes))
        
        # gamma prior likelihood
        if self.mean_vprior is None:
            mean = torch.zeros_like(gamma)
            scale = torch.ones_like(gamma)
            neg_log_likelihood_prior = -Normal(mean, scale).log_prob(gamma).sum(2).sum(1)
        else:
            gamma_expanded = gamma.unsqueeze(1)  # [m, 1, n_labels, n_latent]

            mean_vprior = self.mean_vprior
            var_vprior = self.var_vprior
            mp_vprior = self.mp_vprior

            # 兼容退化输入: (n_labels, D) -> (n_labels, 1, D); (n_labels,) -> (n_labels, 1)
            if mean_vprior.dim() == 2:
                mean_vprior = mean_vprior.unsqueeze(1)
            if var_vprior.dim() == 2:
                var_vprior = var_vprior.unsqueeze(1)
            if mp_vprior.dim() == 1:
                mp_vprior = mp_vprior.unsqueeze(1)

            # 与 SURF 一致的转置: [n_labels, p, D] -> [p, n_labels, D]
            mean_vprior = torch.transpose(mean_vprior, 0, 1).unsqueeze(0)  # [1, p, n_labels, D]
            var_vprior = torch.transpose(var_vprior, 0, 1).unsqueeze(0)    # [1, p, n_labels, D]
            mp_vprior = torch.transpose(mp_vprior, 0, 1)                   # [p, n_labels]

            pre_lse = (
                Normal(mean_vprior, torch.sqrt(var_vprior) + 1e-4)
                .log_prob(gamma_expanded)
                .sum(3)  # sum over D
            ) + torch.log(mp_vprior.clamp_min(1e-12))  # 数值稳定

            log_likelihood_prior = torch.logsumexp(pre_lse, dim=1)  # mix over p
            neg_log_likelihood_prior = -log_likelihood_prior.sum(1)  # sum over labels
        
        # 摊销网络
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
            # GAT相关网络
            from torch_geometric.nn import GINEConv
            
            def mlp_block(in_dim, out_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, gat_hidden),
                    nn.BatchNorm1d(gat_hidden),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(gat_hidden, out_dim),
                )
            
            self.gamma_gat_layers = nn.ModuleList([
                GINEConv(mlp_block(n_genes, gat_hidden), train_eps=True, edge_dim=1)
            ])
            self.gamma_gat_linear = nn.Linear(gat_hidden, n_latent * n_labels)
            self.gamma_ln = nn.LayerNorm(gat_hidden)
            
            self.V_gat_layers = nn.ModuleList([
                GINEConv(mlp_block(n_genes, gat_hidden), train_eps=True, edge_dim=1)
            ])
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
        """生成过程 - 关键修复：decoder调用方式"""
        m = x.shape[0]
        library = x.sum(1, keepdim=True)
        beta = torch.exp(self.beta)
        eps = F.softplus(self.eta)

        x_ = torch.log1p(torch.clamp(x, min=0.0))

        # 获取gamma和v
        if self.use_gat:
            if not hasattr(self, '_X_all'):
                raise RuntimeError("use_gat=True需要先调用attach_full_X")
            if self._edge_index.numel() == 0:
                raise RuntimeError("use_gat=True需要先调用attach_graph")

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
                gamma_ind = torch.transpose(self.gamma_encoder(x_), 0, 1).reshape(
                    (self.n_latent, self.n_labels, -1)
                )
            else:
                gamma_ind = self.gamma[:, :, ind_x]

            if self.amortization in ["both", "proportion"]:
                v_ind = self.V_encoder(x_)
            else:
                v_ind = self.V[:, ind_x].T

        v_ind = F.softplus(v_ind)

        # 解码 - 关键修复：使用正确的decoder调用方式
        gamma_ind = torch.transpose(gamma_ind, 2, 0)  # [m, n_labels, n_latent]
        gamma_reshape = gamma_ind.reshape((-1, self.n_latent))

        # ✅ 修复：构造类别列表的方式需要与DestVI_SURF一致
        enum_label = torch.arange(0, self.n_labels, device=x.device).repeat(m).view((-1, 1))
        
        # ✅ 修复：decoder需要接收两个参数 - 连续输入和类别列表
        h = self.decoder(gamma_reshape, enum_label)
        px_rate = self.px_decoder(h).reshape((m, self.n_labels, -1))

        # 混合
        eps = eps.repeat((m, 1)).view(m, 1, -1)
        r_hat = torch.cat([beta.unsqueeze(0).unsqueeze(1) * px_rate, eps], dim=1)
        px_scale = torch.sum(v_ind.unsqueeze(2) * r_hat, dim=1)
        px_rate_final = library * px_scale

        return {
            "px_rate": px_rate_final,
            "px_scale": px_scale,
            "gamma": gamma_ind,
            "v": v_ind,
        }
    
    def forward(self, item, kl_weight=1.0, n_obs=1.0):
        """完整前向传播"""
        x = item['X']
        ind_x = item['ind_x']
        batch_index = item.get('batch', None)
        
        # 生成
        outputs = self.generative(x, ind_x, batch_index)
        
        px_rate = outputs['px_rate']
        gamma = outputs['gamma']
        v = outputs['v']
        
        # 重构损失
        reconst_loss = -NegativeBinomial(px_rate, logits=self.px_r).log_prob(x).sum(-1)
        
        # eta和beta先验
        mean = torch.zeros_like(self.eta)
        scale = torch.ones_like(self.eta)
        glo_neg_log_likelihood_prior = -self.eta_reg * Normal(mean, scale).log_prob(self.eta).sum()
        
        # ✅ 修复：beta正则化需要使用torch.var而不是手动计算
        glo_neg_log_likelihood_prior += self.beta_reg * torch.var(self.beta)

        # L1稀疏性
        v_sparsity_loss = self.l1_reg * torch.abs(v).mean(1)
        
        # gamma先验
        if self.mean_vprior is None:
            mean = torch.zeros_like(gamma)
            scale = torch.ones_like(gamma)
            neg_log_likelihood_prior = -Normal(mean, scale).log_prob(gamma).sum(2).sum(1)
        else:
            gamma_expanded = gamma.unsqueeze(1)
            mean_vprior = torch.transpose(self.mean_vprior, 0, 1).unsqueeze(0)
            var_vprior = torch.transpose(self.var_vprior, 0, 1).unsqueeze(0)
            mp_vprior = torch.transpose(self.mp_vprior, 0, 1)
            pre_lse = (Normal(mean_vprior, torch.sqrt(var_vprior) + 1e-4).log_prob(gamma_expanded).sum(3)) + torch.log(mp_vprior)
            log_likelihood_prior = torch.logsumexp(pre_lse, 1)
            neg_log_likelihood_prior = -log_likelihood_prior.sum(1)
        
        # Dirichlet MMD
        mmd_term = torch.tensor(0.0, device=px_rate.device)
        if self.dirichlet_mmd_reg > 0.0:
            v_real = v[:, :self.n_labels]
            proportions = v_real / (v_real.sum(dim=1, keepdim=True) + 1e-8)
            alpha = self.dirichlet_alpha.to(proportions.device)
            dirich = Dirichlet(alpha)
            theta_prior = dirich.sample((proportions.size(0),)).to(proportions.device)
            mmd_term = self._mmd_rbf(proportions, theta_prior)
        
        sample_term = reconst_loss + kl_weight * (neg_log_likelihood_prior + v_sparsity_loss)
        loss = n_obs * (torch.mean(sample_term) + glo_neg_log_likelihood_prior + self.dirichlet_mmd_reg * mmd_term)
        
        return {
            'loss': loss,
            'reconstruction_loss': reconst_loss.mean(),
            'kl_local': neg_log_likelihood_prior.mean(),
            'kl_global': glo_neg_log_likelihood_prior,
            'mmd': mmd_term,
        }
    
    def _mmd_rbf(self, x, y, sigma=None):
        """MMD with RBF kernel"""
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
        """获取细胞类型比例"""
        if self.amortization in ["both", "proportion"]:
            if self.use_gat:
                if not hasattr(self, '_X_all'):
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
        """获取gamma参数"""
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
        """获取特定细胞类型的表达"""
        beta = torch.exp(self.beta)
        y_torch = (y * torch.ones_like(ind_x)).ravel()
        
        if self.amortization in ["both", "latent"]:
            x_ = torch.log1p(x)
            gamma_ind = torch.transpose(self.gamma_encoder(x_), 0, 1).reshape(
                (self.n_latent, self.n_labels, -1)
            )
        else:
            gamma_ind = self.gamma[:, :, ind_x]
        
        gamma_select = gamma_ind[:, y_torch, torch.arange(ind_x.shape[0])].T
        
        # ✅ 修复：使用正确的decoder调用方式
        h = self.decoder(gamma_select, y_torch.unsqueeze(1))
        px_scale = self.px_decoder(h)
        
        px_ct = torch.exp(self.px_r).unsqueeze(0) * beta.unsqueeze(0) * px_scale
        return px_ct
    
    def attach_graph(self, adata=None, edge_index=None, edge_weight=None, k=6, spatial_key="spatial"):
        """附加图结构"""
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
            neigh = indices[i, 1:min(k + 1, indices.shape[1])]
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
        """附加全量表达矩阵"""
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
    
    @torch.no_grad()
    def get_graph_embeddings(self, branch="V", projected=False, return_numpy=True):
        """获取图编码器的嵌入"""
        if not self.use_gat:
            raise RuntimeError("use_gat=False时无图编码器")
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
        
        if branch.lower() == "v":
            h = self.V_gat_layers[0](X_all_log, edge_index, edge_attr=edge_attr)
            h = self.V_ln(h)
            h = F.elu(h)
            z = self.V_gat_linear(h) if projected else h
        elif branch.lower() == "gamma":
            h = self.gamma_gat_layers[0](X_all_log, edge_index, edge_attr=edge_attr)
            h = self.gamma_ln(h)
            h = F.elu(h)
            z = self.gamma_gat_linear(h) if projected else h
        else:
            raise ValueError("branch必须是'V'或'gamma'")
        
        return z.detach().cpu().numpy() if return_numpy else z