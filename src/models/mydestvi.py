"""
DestVI模型 - 简化包装
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
        """
        Args:
            n_spots: spot数量
            n_labels: 细胞类型数量
            n_genes: 基因数量
            cell_type_mapping: 细胞类型名称映射
            decoder_state_dict: CondSCVI的decoder参数
            px_decoder_state_dict: CondSCVI的px_decoder参数
            px_r: NB分布参数
            n_hidden: 隐藏层神经元数
            n_latent: 潜在空间维度
            n_layers: 网络层数
            dropout_decoder: Decoder的dropout率
            l1_reg: L1正则化强度
            dirichlet_alpha: Dirichlet先验参数
            dirichlet_mmd_reg: MMD正则化强度
            use_gat: 是否使用GAT
            **module_kwargs: 传递给MRDeconv的其他参数
        """
        super().__init__()
        
        # 创建核心MRDeconv模块
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
        从预训练的CondSCVI模型创建DestVI
        
        Args:
            n_spots: spot数量
            n_genes: 基因数量
            sc_model: 训练好的CondSCVI模型
            cell_type_mapping: 细胞类型名称数组
            vamp_prior_params: (mean_vprior, var_vprior, mp_vprior) 元组
            l1_reg: L1正则化强度
            **module_kwargs: 其他参数
        
        Returns:
            DestVI模型实例
        """
        # 从CondSCVI提取参数
        decoder_state_dict = sc_model.module.decoder.state_dict()
        px_decoder_state_dict = sc_model.module.px_decoder.state_dict()
        px_r = sc_model.module.px_r.detach().cpu().numpy()
        dropout_decoder = sc_model.module.dropout_rate
        
        # VampPrior参数
        if vamp_prior_params is not None:
            mean_vprior, var_vprior, mp_vprior = vamp_prior_params
            module_kwargs['mean_vprior'] = mean_vprior
            module_kwargs['var_vprior'] = var_vprior
            module_kwargs['mp_vprior'] = mp_vprior
        
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
        """
        前向传播
        
        Args:
            item: Dataset返回的字典 {'X': ..., 'ind_x': ...}
            kl_weight: KL权重
            n_obs: 观测数量
        
        Returns:
            dict: {'loss', 'reconstruction_loss', 'kl_local', 'kl_global', 'mmd'}
        """
        return self.module.forward(item, kl_weight, n_obs)
    
    def attach_full_X(self, adata, layer=None):
        """附加全量表达矩阵（GAT模式需要）"""
        return self.module.attach_full_X(adata, layer)
    
    def attach_graph(self, adata=None, edge_index=None, edge_weight=None, k=6, spatial_key='spatial'):
        """附加图结构（GAT模式需要）"""
        return self.module.attach_graph(adata, edge_index, edge_weight, k, spatial_key)
    
    @torch.no_grad()
    def get_proportions(self, adata=None, keep_noise=False):
        """
        获取细胞类型比例
        
        Args:
            adata: AnnData对象（FC模式需要）
            keep_noise: 是否保留噪声项
        
        Returns:
            pd.DataFrame: [n_spots, n_labels]
        """
        self.eval()
        
        if adata is not None:
            # FC模式：需要表达矩阵
            from scipy.sparse import issparse
            X = adata.X
            X = torch.FloatTensor(X.toarray() if issparse(X) else X)
            X = X.to(next(self.parameters()).device)
            props = self.module.get_proportions(X, keep_noise)
        else:
            # GAT模式或非摊销模式
            props = self.module.get_proportions(None, keep_noise)
        
        if isinstance(props, torch.Tensor):
            props = props.cpu().numpy()
        
        # 转为DataFrame
        column_names = self.cell_type_mapping.tolist()
        if keep_noise:
            column_names.append('noise_term')
        
        index_names = adata.obs.index if adata is not None else np.arange(self.n_spots)
        
        return pd.DataFrame(props, columns=column_names, index=index_names)
    
    @torch.no_grad()
    def get_gamma(self, adata=None):
        """
        获取gamma参数
        
        Args:
            adata: AnnData对象（FC模式需要）
        
        Returns:
            dict: {cell_type: pd.DataFrame}
        """
        self.eval()
        
        if adata is not None:
            from scipy.sparse import issparse
            X = adata.X
            X = torch.FloatTensor(X.toarray() if issparse(X) else X)
            X = X.to(next(self.parameters()).device)
            gamma = self.module.get_gamma(X)
        else:
            gamma = self.module.get_gamma(None)
        
        # gamma shape: [n_latent, n_labels, n_spots]
        # 转为字典
        result = {}
        for i, ct in enumerate(self.cell_type_mapping):
            result[ct] = pd.DataFrame(
                gamma[:, i, :].T,  # [n_spots, n_latent]
                columns=[f'latent_{j}' for j in range(gamma.shape[0])]
            )
        
        return result