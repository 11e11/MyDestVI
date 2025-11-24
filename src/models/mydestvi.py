"""
DestVI模型 - 简化封装（修复：从 CondSCVI/VAEC 提取 decoder_backbone 权重）
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.modules.mymrdeconv import MRDeconv


class DestVI(nn.Module):
    """DestVI model - 用户接口层"""

    def __init__(
        self,
        n_spots: int,
        n_labels: int,
        n_genes: int,
        cell_type_mapping: np.ndarray,
        decoder_state_dict,
        px_decoder_state_dict,
        px_r: np.ndarray,
        n_hidden: int,
        n_latent: int,
        n_layers: int,
        dropout_decoder: float,
        l1_reg: float = 0.0,
        dirichlet_alpha: float | list = None,
        dirichlet_mmd_reg: float = 0.0,
        use_gat: bool = False,
        **module_kwargs
    ):
        super().__init__()

        self.module = MRDeconv(
            n_spots=n_spots,
            n_labels=n_labels,
            n_genes=n_genes,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            decoder_state_dict=decoder_state_dict,
            px_decoder_state_dict=px_decoder_state_dict,
            px_r=px_r,
            dropout_decoder=dropout_decoder,
            l1_reg=l1_reg,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_mmd_reg=dirichlet_mmd_reg,
            use_gat=use_gat,
            **module_kwargs
        )

        self.cell_type_mapping = cell_type_mapping
        self.n_labels = n_labels
        self.n_spots = n_spots

    @classmethod
    def from_rna_model(
        cls,
        n_spots: int,
        n_genes: int,
        sc_model,
        cell_type_mapping: np.ndarray,
        vamp_prior_params: tuple = None,
        l1_reg: float = 0.0,
        **module_kwargs
    ):
        """
        从预训练的 CondSCVI 模型创建 DestVI

        优先通过 CondSCVI 的导出方法/decoder_backbone 获取解码器/px_decoder/px_r
        """
        # 1) 解码器 backbone（优先导出方法）
        decoder_state_dict = None
        if hasattr(sc_model, "export_decoder_state"):
            decoder_state_dict = sc_model.export_decoder_state()

        if decoder_state_dict is None:
            m = getattr(sc_model, "module", None)
            dec = None
            if m is not None:
                # 新命名 decoder_backbone 优先
                dec = getattr(m, "decoder_backbone", None)
                if dec is None:
                    # 兼容旧命名 decoder
                    dec = getattr(m, "decoder", None)
            decoder_state_dict = dec.state_dict() if dec is not None else None
            if decoder_state_dict is None:
                print("[WARN] CondSCVI.module attributes:", dir(m) if m is not None else "None")

        # 2) px_decoder
        px_decoder_state_dict = None
        if hasattr(sc_model, "export_px_decoder_state"):
            px_decoder_state_dict = sc_model.export_px_decoder_state()
        if px_decoder_state_dict is None:
            m = getattr(sc_model, "module", None)
            px = getattr(m, "px_decoder", None) if m is not None else None
            px_decoder_state_dict = px.state_dict() if px is not None else None

        # 3) px_r / dropout
        if hasattr(sc_model, "export_px_r"):
            px_r = sc_model.export_px_r()
        else:
            px_r = sc_model.module.px_r.detach().cpu().numpy()
        if hasattr(sc_model, "export_dropout_decoder"):
            dropout_decoder = sc_model.export_dropout_decoder()
        else:
            dropout_decoder = getattr(sc_model.module, "dropout_rate", 0.05)

        # 4) VampPrior（如果提供）
        if vamp_prior_params is not None:
            mean_vprior, var_vprior, mp_vprior = vamp_prior_params
            module_kwargs["mean_vprior"] = mean_vprior
            module_kwargs["var_vprior"] = var_vprior
            module_kwargs["mp_vprior"] = mp_vprior

        return cls(
            n_spots=n_spots,
            n_labels=sc_model.n_labels,
            n_genes=n_genes,
            cell_type_mapping=cell_type_mapping,
            decoder_state_dict=decoder_state_dict,
            px_decoder_state_dict=px_decoder_state_dict,
            px_r=px_r,
            n_hidden=sc_model.n_hidden,
            n_latent=sc_model.n_latent,
            n_layers=sc_model.n_layers,
            dropout_decoder=dropout_decoder,
            l1_reg=l1_reg,
            **module_kwargs
        )

    def forward(self, item, kl_weight=1.0, n_obs=1.0):
        return self.module.forward(item, kl_weight, n_obs)

    def attach_full_X(self, adata, layer=None):
        return self.module.attach_full_X(adata, layer)

    def attach_dual_graph(self, adata, k_spatial=6, k_expr=10, spatial_key='spatial'):  # 🔥 新增
        return self.module.attach_dual_graph(adata, k_spatial, k_expr, spatial_key)

    def attach_graph(self, adata=None, edge_index=None, edge_weight=None, k=6, spatial_key='spatial'):
        return self.module.attach_graph(adata, edge_index, edge_weight, k, spatial_key)

    @torch.no_grad()
    def get_proportions(self, adata=None, keep_noise=False):
        self.eval()
        if adata is not None:
            from scipy.sparse import issparse
            X = adata.X
            X = torch.FloatTensor(X.toarray() if issparse(X) else X)
            X = X.to(next(self.parameters()).device)
            props = self.module.get_proportions(X, keep_noise)
        else:
            props = self.module.get_proportions(None, keep_noise)

        if isinstance(props, torch.Tensor):
            props = props.cpu().numpy()

        column_names = self.cell_type_mapping.tolist()
        if keep_noise:
            column_names.append('noise_term')

        index_names = adata.obs.index if adata is not None else np.arange(self.n_spots)
        return pd.DataFrame(props, columns=column_names, index=index_names)

    @torch.no_grad()
    def get_gamma(self, adata=None):
        self.eval()
        if adata is not None:
            from scipy.sparse import issparse
            X = adata.X
            X = torch.FloatTensor(X.toarray() if issparse(X) else X)
            X = X.to(next(self.parameters()).device)
            gamma = self.module.get_gamma(X)
        else:
            gamma = self.module.get_gamma(None)

        result = {}
        for i, ct in enumerate(self.cell_type_mapping):
            result[ct] = pd.DataFrame(
                gamma[:, i, :].T,
                columns=[f'latent_{j}' for j in range(gamma.shape[0])]
            )
        return result