from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers
from torch.distributions import Normal, Dirichlet
import torch.nn.functional as F
from torch_geometric.nn import GATConv 
import numpy as np

if TYPE_CHECKING:
    from collections import OrderedDict
    from typing import Literal

    import numpy as np


def identity(x):
    """Identity function."""
    return x


class MRDeconv(BaseModuleClass):
    """Model for multi-resolution deconvolution of spatial transriptomics.

    Parameters
    ----------
    n_spots
        Number of input spots
    n_labels
        Number of cell types
    n_hidden
        Number of neurons in the hidden layers
    n_layers
        Number of layers used in the encoder networks
    n_latent
        Number of dimensions used in the latent variables
    n_genes
        Number of genes used in the decoder
    dropout_decoder
        Dropout rate for the decoder neural network (same dropout as in CondSCVI decoder)
    dropout_amortization
        Dropout rate for the amortization neural network
    decoder_state_dict
        state_dict from the decoder of the CondSCVI model
    px_decoder_state_dict
        state_dict from the px_decoder of the CondSCVI model
    px_r
        parameters for the px_r tensor in the CondSCVI model
    mean_vprior
        Mean parameter for each component in the empirical prior over the latent space
    var_vprior
        Diagonal variance parameter for each component in the empirical prior over the latent space
    mp_vprior
        Mixture proportion in cell type sub-clustering of each component in the empirical prior
    amortization
        which of the latent variables to amortize inference over (gamma, proportions, both or none)
    l1_reg
        Scalar parameter indicating the strength of L1 regularization on cell type proportions.
        A value of 50 leads to sparser results.
    beta_reg
        Scalar parameter indicating the strength of the variance penalty for
        the multiplicative offset in gene expression values (beta parameter). Default is 5
        (setting to 0.5 might help if single cell reference and spatial assay are different
        e.g. UMI vs non-UMI.)
    eta_reg
        Scalar parameter indicating the strength of the prior for
        the noise term (eta parameter). Default is 1e-4.
        (changing value is discouraged.)
    extra_encoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.FCLayers`.
    extra_decoder_kwargs
        Extra keyword arguments passed into :class:`~scvi.nn.FCLayers`.
    """

    
    def __init__(
        self,
        n_spots: int,
        n_labels: int,
        n_hidden: int,
        n_layers: int,
        n_latent: int,
        n_genes: int,
        decoder_state_dict: OrderedDict,
        px_decoder_state_dict: OrderedDict,
        px_r: np.ndarray,
        dropout_decoder: float,
        dropout_amortization: float = 0.05,
        mean_vprior: np.ndarray = None,
        var_vprior: np.ndarray = None,
        mp_vprior: np.ndarray = None,
        amortization: Literal["none", "latent", "proportion", "both"] = "both",
        l1_reg: float = 0.0,
        beta_reg: float = 5.0,
        eta_reg: float = 1e-4,

        dirichlet_alpha: float | list | None = None,
        dirichlet_mmd_reg: float = 0.0,

        contrastive_reg: float = 1.0,
        contrastive_margin: float = 0.5,

        # ---- GAT 相关新参数 ----
        use_gat: bool = False,
        gat_hidden: int = 64,
        gat_heads: int = 4,

        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
    ):
        super().__init__()
        self.n_spots = n_spots
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_decoder = dropout_decoder
        self.dropout_amortization = dropout_amortization
        self.n_genes = n_genes
        self.amortization = amortization
        self.l1_reg = l1_reg
        self.beta_reg = beta_reg
        self.eta_reg = eta_reg

        self.dirichlet_mmd_reg = dirichlet_mmd_reg 
        # self.dirichlet_alpha = dirichlet_alpha

        self.contrastive_reg = contrastive_reg
        self.contrastive_margin = contrastive_margin

        self._dbg_print = 3  # 仅前3个batch打印调试信息

        # 记录是否使用 GAT
        self.use_gat = bool(use_gat)
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads

        # unpack and copy parameters
        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=[n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_decoder,
            use_layer_norm=True,
            use_batch_norm=False,
            **_extra_decoder_kwargs,
        )
        self.px_decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_genes), torch.nn.Softplus()
        )
        # don't compute gradient for those parameters
        self.decoder.load_state_dict(decoder_state_dict)
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.px_decoder.load_state_dict(px_decoder_state_dict)
        for param in self.px_decoder.parameters():
            param.requires_grad = False
        self.register_buffer("px_o", torch.tensor(px_r, dtype=torch.float32))

        # cell_type specific factor loadings
        self.V = torch.nn.Parameter(torch.randn(self.n_labels + 1, self.n_spots))

        # within cell_type factor loadings
        self.gamma = torch.nn.Parameter(torch.randn(n_latent, self.n_labels, self.n_spots))
        if mean_vprior is not None:
            self.p = mean_vprior.shape[1]
            self.register_buffer("mean_vprior", torch.tensor(mean_vprior, dtype=torch.float32))
            self.register_buffer("var_vprior", torch.tensor(var_vprior, dtype=torch.float32))
            self.register_buffer("mp_vprior", torch.tensor(mp_vprior, dtype=torch.float32))
        else:
            self.mean_vprior = None
            self.var_vprior = None
        # noise from data
        self.eta = torch.nn.Parameter(torch.randn(self.n_genes))
        # additive gene bias
        self.beta = torch.nn.Parameter(0.01 * torch.randn(self.n_genes))

        # create additional neural nets for amortization
        # within cell_type factor loadings
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        if self.use_gat:
            # gamma_encoder: 若使用 GAT -> 多头 GAT 层 + 线性投影
            # 输入 feature: n_genes -> gat_hidden, 最终输出 n_latent * n_labels
            self.gamma_gat_layers = torch.nn.ModuleList()
            in_ch = self.n_genes
            heads = self.gat_heads
            hidden = self.gat_hidden
            # 可堆叠两层 GAT（第一层多头，第二层合并 heads）
            self.gamma_gat_layers.append(GATConv(in_ch, hidden // heads, heads=heads, concat=True))
            self.gamma_gat_layers.append(GATConv(hidden, hidden // 1, heads=1, concat=False))
            self.gamma_gat_linear = torch.nn.Linear(hidden, n_latent * n_labels)

            # V encoder similar：输出 n_labels + 1
            self.V_gat_layers = torch.nn.ModuleList()
            self.V_gat_layers.append(GATConv(in_ch, hidden // heads, heads=heads, concat=True))
            self.V_gat_layers.append(GATConv(hidden, hidden // 1, heads=1, concat=False))
            self.V_gat_linear = torch.nn.Linear(hidden, n_labels + 1)

            # placeholder for edge_index（通过 attach_graph 注册）
            self.register_buffer("_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        else:
            # 原有实现（FC amortization）
            self.gamma_encoder = torch.nn.Sequential(
                FCLayers(
                    n_in=self.n_genes,
                    n_out=n_hidden,
                    n_cat_list=None,
                    n_layers=2,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_amortization,
                    use_layer_norm=True,
                    use_batch_norm=False,
                    **_extra_encoder_kwargs,
                ),
                torch.nn.Linear(n_hidden, n_latent * n_labels),
            )
            self.V_encoder = torch.nn.Sequential(
                FCLayers(
                    n_in=self.n_genes,
                    n_out=n_hidden,
                    n_layers=2,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_amortization,
                    use_layer_norm=True,
                    use_batch_norm=False,
                    **_extra_encoder_kwargs,
                ),
                torch.nn.Linear(n_hidden, n_labels + 1),
            )

        if dirichlet_alpha is None:
            self.dirichlet_alpha = torch.ones(self.n_labels)
        else:
            alpha = torch.tensor(dirichlet_alpha, dtype=torch.float32)
            if alpha.numel() == 1:
                self.dirichlet_alpha = alpha.repeat(self.n_labels)
            elif alpha.numel() != self.n_labels:
                raise ValueError("dirichlet_alpha must be scalar or length n_labels (exclude dummy)")
            else:
                self.dirichlet_alpha = alpha

        self.dirichlet_mmd_reg = float(dirichlet_mmd_reg)

    # 新增：把 edge_index 注册到 module 中（支持 AnnData 或直接 edge_index）
    def attach_graph(self, adata=None, edge_index: torch.Tensor | None = None, k: int = 6, spatial_key: str = "spatial"):
        """
        将空间邻接关系注册为 buffer，供 GAT 编码器使用。
        使用方式：
          - 直接传入 edge_index: tensor shape [2, E] (long)
          - 或传入 adata（含空间坐标在 adata.obsm[spatial_key]），并用 k-NN 构图
        """
        if edge_index is not None:
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.as_tensor(edge_index, dtype=torch.long)
            self.register_buffer("_edge_index", edge_index, persistent=False)
            return self

        if adata is None:
            raise ValueError("attach_graph 需要 adata 或 edge_index 之一")

        # 从空间坐标构造 kNN 边
        coords = None
        if spatial_key in adata.obsm:
            coords = adata.obsm[spatial_key]
        elif "spatial" in adata.obsm:
            coords = adata.obsm["spatial"]
        else:
            raise ValueError("未在 adata.obsm 中找到空间坐标，提供 spatial_key 参数或自行传入 edge_index")

        import numpy as _np
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]), algorithm="auto").fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        # indices 包含 self，去掉第0列（自己）
        src = []
        dst = []
        for i in range(indices.shape[0]):
            neigh = indices[i, 1: min(k + 1, indices.shape[1])]
            for j in neigh:
                src.append(i)
                dst.append(int(j))
        edge_index = torch.as_tensor([src, dst], dtype=torch.long)
        # 无向图 -> 对称
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        self.register_buffer("_edge_index", edge_index, persistent=False)
        return self

    def _pairwise_sq_dists(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_norm = (a ** 2).sum(dim=1).unsqueeze(1)
        b_norm = (b ** 2).sum(dim=1).unsqueeze(0)
        return a_norm + b_norm - 2.0 * a @ b.t()

    def _mmd_rbf(self, x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
        n = x.size(0)
        m = y.size(0)
        if n <= 1 or m <= 1:
            return torch.tensor(0.0, device=x.device)
        if sigma is None:
            comb = torch.cat([x, y], dim=0)
            d2 = self._pairwise_sq_dists(comb, comb)
            d2_flat = d2.view(-1)
            d2_pos = d2_flat[d2_flat > 0]
            sigma = (torch.sqrt(d2_pos.median()) + 1e-8).item() if d2_pos.numel() > 0 else 1.0
        Kxx = torch.exp(-self._pairwise_sq_dists(x, x) / (2.0 * sigma * sigma))
        Kyy = torch.exp(-self._pairwise_sq_dists(y, y) / (2.0 * sigma * sigma))
        Kxy = torch.exp(-self._pairwise_sq_dists(x, y) / (2.0 * sigma * sigma))
        sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
        sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
        sum_Kxy = Kxy.sum() / (n * m)
        return sum_Kxx + sum_Kyy - 2.0 * sum_Kxy

    def _get_inference_input(self, tensors):
        # we perform MAP here, so we just need to subsample the variables
        return {}
    
    def attach_full_X(self, adata, layer: str | None = None):
        """
        将全量表达矩阵缓存为 buffer: self._X_all [N, G] (float32, device 跟随模块)
        layer 为 None 则使用 adata.X；否则使用 adata.layers[layer]
        """
        import numpy as np
        from scipy.sparse import issparse

        if layer is not None and layer in getattr(adata, "layers", {}):
            X = adata.layers[layer]
        else:
            X = adata.X

        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # 用 buffer，确保 .to(device) 时一起迁移
        self.register_buffer("_X_all", torch.from_numpy(X), persistent=False)
        # 记录列数，供空返回时使用
        self.n_genes = X.shape[1]
        return self

    def _build_samples_from_indices(self, indices_mat: torch.Tensor, counts: torch.Tensor, device: torch.device):
        """
        根据 [B, K] 的索引矩阵或 object-数组与每个 anchor 的有效数量 counts 构造样本矩阵。
        兼容 torch.Tensor / np.ndarray(object or int) / list[list[int]] 等。
        返回 [sum(counts), n_genes]（若无有效样本返回形状 [0, n_genes]）。
        """
        if not hasattr(self, "_X_all"):
            return torch.empty((0, self.n_genes), dtype=torch.float32, device=device)

        # 统一为 numpy 容器，便于逐行处理
        if isinstance(indices_mat, torch.Tensor):
            idx_np = indices_mat.detach().cpu().numpy()
        else:
            idx_np = np.asarray(
                indices_mat,
                dtype=object if isinstance(indices_mat, np.ndarray) and indices_mat.dtype == object else None,
            )

        cnt_np = counts.detach().cpu().numpy().astype(int)

        rows = []
        for i, c in enumerate(cnt_np):
            if c <= 0:
                continue

            row = idx_np[i]
            # 将 row 统一为 1D numpy int 数组
            if isinstance(row, torch.Tensor):
                row = row.detach().cpu().numpy()
            elif isinstance(row, (list, tuple)):
                row = np.asarray(row)
            elif np.isscalar(row):
                row = np.asarray([row])

            row = row.astype(np.int64, copy=False)
            vids = row[:c]
            if vids.size == 0:
                continue
            vids = vids[vids >= 0]
            if vids.size == 0:
                continue

            # 用 torch.long 索引 torch 张量，避免 numpy 索引报错
            vids_t = torch.as_tensor(vids, dtype=torch.long, device=self._X_all.device)
            rows.append(self._X_all.index_select(0, vids_t))

        if len(rows) == 0:
            return torch.empty((0, self.n_genes), dtype=torch.float32, device=device)

        Xcat = torch.cat(rows, dim=0).to(device, dtype=torch.float32)
        return Xcat


    def _build_samples_from_padded(
        self,
        padded_indices: torch.Tensor,
        counts: torch.Tensor,
        device: torch.device,
    ):
        """
        padded_indices: [B, K] 长整型（-1 为 padding）
        counts:         [B]    每个 anchor 的候选数量（可能包含 padding）
        返回:
          - samples:     [sum(eff_counts), n_genes]
          - eff_counts:  [B] 每个 anchor 的有效样本数（已去除 -1）
        """
        if not hasattr(self, "_X_all"):
            eff_counts = torch.zeros_like(counts, dtype=torch.long, device=device)
            return torch.empty((0, self.n_genes), dtype=torch.float32, device=device), eff_counts

        if not isinstance(padded_indices, torch.Tensor):
            padded_indices = torch.as_tensor(padded_indices)
        if not isinstance(counts, torch.Tensor):
            counts = torch.as_tensor(counts)

        padded_indices = padded_indices.to(dtype=torch.long, device=self._X_all.device)
        counts = counts.to(dtype=torch.long, device=self._X_all.device)

        B, K = padded_indices.shape
        rows = []
        eff = []
        for i in range(B):
            c = int(counts[i].item())
            if c <= 0:
                eff.append(0)
                continue
            c = min(c, K)
            idx_i = padded_indices[i, :c]
            idx_i = idx_i[idx_i >= 0]  # 去掉 padding
            if idx_i.numel() == 0:
                eff.append(0)
                continue
            rows.append(self._X_all.index_select(0, idx_i))
            eff.append(int(idx_i.numel()))

        eff_counts = torch.as_tensor(eff, dtype=torch.long, device=self._X_all.device)
        if len(rows) == 0:
            return torch.empty((0, self.n_genes), dtype=torch.float32, device=device), eff_counts
        samples = torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)
        return samples, eff_counts

    def _get_generative_input(self, tensors, inference_outputs):
        x = tensors[REGISTRY_KEYS.X_KEY]
        ind_x = tensors[REGISTRY_KEYS.INDICES_KEY].long().ravel()
        batch_index = None

        pos_counts_obs = tensors.get("pos_indices_count", None)
        neg_counts_obs = tensors.get("neg_indices_count", None)
        pos_pad = tensors.get("pos_indices_padded", None)
        neg_pad = tensors.get("neg_indices_padded", None)

        device = x.device
        pos_samples = None
        neg_samples = None
        pos_counts_eff = None
        neg_counts_eff = None

        if pos_pad is not None and pos_counts_obs is not None:
            pos_samples, pos_counts_eff = self._build_samples_from_padded(pos_pad, pos_counts_obs, device)
        if neg_pad is not None and neg_counts_obs is not None:
            neg_samples, neg_counts_eff = self._build_samples_from_padded(neg_pad, neg_counts_obs, device)

        return {
            "x": x,
            "ind_x": ind_x,
            "batch_index": batch_index,
            "pos_samples": pos_samples,
            "neg_samples": neg_samples,
            # 用“有效计数”，若为空则回退到 obs 的 counts
            "pos_counts": pos_counts_eff if pos_counts_eff is not None else pos_counts_obs,
            "neg_counts": neg_counts_eff if neg_counts_eff is not None else neg_counts_obs,
        }

    @auto_move_data
    def inference(self):
        """Run the inference model."""
        return {}

    # 在 generative 方法中，完全替换 GAT 分支和对比学习部分：
    @auto_move_data
    def generative(self, x, ind_x, batch_index=None, transform_batch: torch.Tensor | None = None,
               pos_samples=None, neg_samples=None, pos_counts=None, neg_counts=None):
        """Build the deconvolution model for every cell in the minibatch."""
        m = x.shape[0]
        library = torch.sum(x, dim=1, keepdim=True)
        beta = torch.exp(self.beta)
        eps = torch.nn.functional.softplus(self.eta)
        x_ = torch.log(1 + x)

        # ---- 调用 encoder 的分支 ----
        if self.use_gat:
            edge_index = self._edge_index
            if edge_index.numel() == 0:
                raise RuntimeError("use_gat=True 需要先调用 attach_graph(...) 注册 edge_index")
            if not hasattr(self, "_X_all"):
                raise RuntimeError("use_gat=True 需要先调用 attach_full_X(adata) 缓存全量 X")

            # 1) 用全量 X 跑 GAT，再按 ind_x 选择 minibatch
            X_all = self._X_all.to(x.device, dtype=torch.float32)
            X_all_log = torch.log1p(X_all)

            # gamma GAT
            h = X_all_log
            for layer in self.gamma_gat_layers:
                h = F.elu(layer(h, edge_index))
            gamma_raw_all = self.gamma_gat_linear(h)  # [N, n_latent*n_labels]
            gamma_mb = gamma_raw_all[ind_x, :]        # [m, n_latent*n_labels]
            gamma_ind = gamma_mb.view(m, self.n_labels, self.n_latent).permute(2, 1, 0)  # (Z,L,m)

            # V GAT
            h2 = X_all_log
            for layer in self.V_gat_layers:
                h2 = F.elu(layer(h2, edge_index))
            v_raw_all = self.V_gat_linear(h2)         # [N, n_labels+1]
            v_ind = v_raw_all[ind_x, :]               # [m, n_labels+1]
        else:
            # 原来的 FC 分支
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

        v_ind = torch.nn.functional.softplus(v_ind)

        # reshape and get gene expression value for all minibatch
        gamma_ind = torch.transpose(gamma_ind, 2, 0)  # minibatch_size, n_labels, n_latent
        gamma_reshape = gamma_ind.reshape((-1, self.n_latent))
        enum_label = torch.arange(0, self.n_labels).repeat(m).view((-1, 1))
        h = self.decoder(gamma_reshape, enum_label.to(x.device))
        px_rate = self.px_decoder(h).reshape((m, self.n_labels, -1))

        # add the dummy cell type
        eps = eps.repeat((m, 1)).view(m, 1, -1)
        r_hat = torch.cat([beta.unsqueeze(0).unsqueeze(1) * px_rate, eps], dim=1)
        px_scale = torch.sum(v_ind.unsqueeze(2) * r_hat, dim=1)
        px_rate = library * px_scale

        # 简化返回（去掉对比学习相关）
        return {
            "px_o": self.px_o,
            "px_rate": px_rate,
            "px_scale": px_scale,
            "gamma": gamma_ind,
            "v": v_ind,
            "batch_index": batch_index,
        }


    # 添加新的对比学习损失方法，替换现有的compute_contrastive_loss
    def compute_contrastive_loss_direct(
        self,
        spot_representation,
        pos_representations,
        neg_representations,
        pos_counts,
        neg_counts,
    ):
        device = spot_representation.device

        if pos_counts is None or neg_counts is None:
            return torch.tensor(0.0, device=device)
        if (pos_representations is None or pos_representations.numel() == 0) and \
           (neg_representations is None or neg_representations.numel() == 0):
            return torch.tensor(0.0, device=device)

        anchor = F.normalize(spot_representation, p=2, dim=1, eps=1e-8)
        pos = None if pos_representations is None else F.normalize(pos_representations, p=2, dim=1, eps=1e-8)
        neg = None if neg_representations is None else F.normalize(neg_representations, p=2, dim=1, eps=1e-8)

        pos_counts = pos_counts.to(device).long().view(-1)
        neg_counts = neg_counts.to(device).long().view(-1)

        bs = anchor.size(0)
        pos_off = 0
        neg_off = 0
        losses = []
        for i in range(bs):
            a = anchor[i:i+1]

            pc = int(pos_counts[i].item()) if pos is not None else 0
            if pc > 0:
                pos_i = pos[pos_off:pos_off + pc]
                pos_sim = torch.tensor(0.0, device=device) if pos_i.numel() == 0 else \
                    F.cosine_similarity(a, pos_i, dim=1, eps=1e-8).mean()
            else:
                pos_sim = torch.tensor(0.0, device=device)

            nc = int(neg_counts[i].item()) if neg is not None else 0
            if nc > 0:
                neg_i = neg[neg_off:neg_off + nc]
                neg_sim = torch.tensor(0.0, device=device) if neg_i.numel() == 0 else \
                    F.cosine_similarity(a, neg_i, dim=1, eps=1e-8).mean()
            else:
                neg_sim = torch.tensor(0.0, device=device)

            pos_off += pc
            neg_off += nc

            losses.append(F.relu(self.contrastive_margin + neg_sim - pos_sim))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        loss = torch.stack(losses).mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    # 简化 loss 方法（去掉对比学习）
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0, n_obs: int = 1.0):
        """Compute the loss."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        px_o = generative_outputs["px_o"]
        gamma = generative_outputs["gamma"]
        v = generative_outputs["v"]

        reconst_loss = -NegativeBinomial(px_rate, logits=px_o).log_prob(x).sum(-1)

        # eta prior likelihood
        mean = torch.zeros_like(self.eta)
        scale = torch.ones_like(self.eta)
        glo_neg_log_likelihood_prior = -self.eta_reg * Normal(mean, scale).log_prob(self.eta).sum()
        glo_neg_log_likelihood_prior += self.beta_reg * torch.var(self.beta)

        v_sparsity_loss = self.l1_reg * torch.abs(v).mean(1)

        # gamma prior likelihood
        if self.mean_vprior is None:
            mean = torch.zeros_like(gamma)
            scale = torch.ones_like(gamma)
            neg_log_likelihood_prior = -Normal(mean, scale).log_prob(gamma).sum(2).sum(1)
        else:
            gamma = gamma.unsqueeze(1)
            mean_vprior = torch.transpose(self.mean_vprior, 0, 1).unsqueeze(0)
            var_vprior = torch.transpose(self.var_vprior, 0, 1).unsqueeze(0)
            mp_vprior = torch.transpose(self.mp_vprior, 0, 1)
            pre_lse = (Normal(mean_vprior, torch.sqrt(var_vprior) + 1e-4).log_prob(gamma).sum(3)) + torch.log(mp_vprior)
            log_likelihood_prior = torch.logsumexp(pre_lse, 1)
            neg_log_likelihood_prior = -log_likelihood_prior.sum(1)

        # Dirichlet MMD
        mmd_term = torch.tensor(0.0, device=px_rate.device)
        if getattr(self, "dirichlet_mmd_reg", 0.0) > 0.0:
            v_real = v[:, : self.n_labels]
            proportions = v_real / (v_real.sum(dim=1, keepdim=True) + 1e-8)
            alpha = self.dirichlet_alpha.to(proportions.device)
            dirich = Dirichlet(alpha)
            theta_prior = dirich.sample((proportions.size(0),)).to(proportions.device)
            mmd_term = self._mmd_rbf(proportions, theta_prior)

        sample_term = reconst_loss + kl_weight * (neg_log_likelihood_prior + v_sparsity_loss)
        loss = n_obs * (torch.mean(sample_term) + glo_neg_log_likelihood_prior + self.dirichlet_mmd_reg * mmd_term)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=neg_log_likelihood_prior,
            kl_global=glo_neg_log_likelihood_prior,
            extra_metrics={"mmd": mmd_term}
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ):
        """Sample from the posterior."""
        raise NotImplementedError("No sampling method for DestVI")

    @torch.inference_mode()
    @auto_move_data
    def get_proportions(self, x=None, keep_noise=False) -> np.ndarray:
        """Returns the loadings."""
        if self.amortization in ["both", "proportion"]:
            if self.use_gat:
                if not hasattr(self, "_X_all"):
                    raise RuntimeError("use_gat=True 需要先调用 attach_full_X(...)")
                if self._edge_index.numel() == 0:
                    raise RuntimeError("use_gat=True 需要先调用 attach_graph(...)")

                X_all = self._X_all.to(self.px_o.device, dtype=torch.float32)
                X_all_log = torch.log1p(X_all)
                h = X_all_log
                for layer in self.V_gat_layers:
                    h = F.elu(layer(h, self._edge_index))
                res = torch.nn.functional.softplus(self.V_gat_linear(h))  # 移除 .cpu().numpy()
            else:
                x_ = torch.log(1 + x)
                res = torch.nn.functional.softplus(self.V_encoder(x_))  # 也移除这里的 .cpu().numpy()
        else:
            # 非 amortization 模式
            res = torch.nn.functional.softplus(self.V)  
            if res.dim() == 2 and res.shape[0] == self.n_labels + 1:
                res = res.T  # 转置为 (n_spots, n_labels+1)

        if not keep_noise:
            res = res[:, :-1]
        res = res / res.sum(axis=1, keepdims=True)
        return res  # 返回 tensor，让调用方决定是否转换为 numpy

    # 修复 get_gamma 方法：
    @torch.inference_mode()
    @auto_move_data
    def get_gamma(self, x: torch.Tensor = None) -> torch.Tensor:
        """Returns the loadings."""
        if self.amortization in ["latent", "both"]:
            if self.use_gat:
                if not hasattr(self, "_X_all"):
                    raise RuntimeError("use_gat=True 需要先调用 attach_full_X(...)")
                if self._edge_index.numel() == 0:
                    raise RuntimeError("use_gat=True 需要先调用 attach_graph(...)")

                X_all = self._X_all.to(self.px_o.device, dtype=torch.float32)
                X_all_log = torch.log1p(X_all)
                h = X_all_log
                for layer in self.gamma_gat_layers:
                    h = F.elu(layer(h, self._edge_index))
                gamma_raw_all = self.gamma_gat_linear(h)
                N = gamma_raw_all.size(0)
                gamma = gamma_raw_all.view(N, self.n_labels, self.n_latent).permute(2, 1, 0)
                return gamma.cpu().numpy()
            else:
                x_ = torch.log(1 + x)
                gamma = self.gamma_encoder(x_)
                return torch.transpose(gamma, 0, 1).reshape((self.n_latent, self.n_labels, -1)).cpu().numpy()
        else:
            return self.gamma.cpu().numpy()

    @torch.inference_mode()
    @auto_move_data
    def get_ct_specific_expression(
        self, x: torch.Tensor = None, ind_x: torch.Tensor = None, y: int = None
    ):
        """Returns cell type specific gene expression at the queried spots.

        Parameters
        ----------
        x
            tensor of data
        ind_x
            tensor of indices
        y
            integer for cell types
        """
        # cell-type specific gene expression, shape (minibatch, celltype, gene).
        beta = torch.exp(self.beta)  # n_genes
        y_torch = (y * torch.ones_like(ind_x)).ravel()
        # obtain the relevant gammas
        if self.amortization in ["both", "latent"]:
            x_ = torch.log(1 + x)
            gamma_ind = torch.transpose(self.gamma_encoder(x_), 0, 1).reshape(
                (self.n_latent, self.n_labels, -1)
            )
        else:
            gamma_ind = self.gamma[:, :, ind_x]  # n_latent, n_labels, minibatch_size

        # calculate cell type specific expression
        gamma_select = gamma_ind[
            :, y_torch, torch.arange(ind_x.shape[0])
        ].T  # minibatch_size, n_latent
        h = self.decoder(gamma_select, y_torch.unsqueeze(1))
        px_scale = self.px_decoder(h)  # (minibatch, n_genes)
        px_ct = torch.exp(self.px_o).unsqueeze(0) * beta.unsqueeze(0) * px_scale
        return px_ct  # shape (minibatch, genes)