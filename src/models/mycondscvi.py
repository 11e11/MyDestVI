"""
CondSCVI模型 - 简化包装
"""
import torch
import torch.nn as nn
import numpy as np

from src.modules.myvaec import VAEC


class CondSCVI(nn.Module):
    """Conditional scVI model - 用户接口层"""
    
    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_batch: int = 1,
        n_hidden: int = 128,
        n_latent: int = 5,
        n_layers: int = 2,
        dropout_rate: float = 0.05,
        weight_obs: bool = False,
        ct_counts: np.ndarray = None,
        encode_covariates: bool = False,
        **module_kwargs
    ):
        """
        Args:
            n_input: 基因数量
            n_labels: 细胞类型数量
            n_batch: 批次数量
            n_hidden: 隐藏层神经元数
            n_latent: 潜在空间维度
            n_layers: 网络层数
            dropout_rate: Dropout率
            weight_obs: 是否对低频细胞类型重加权
            ct_counts: 细胞类型计数（用于weight_obs）
            encode_covariates: 是否在编码器中使用协变量
            **module_kwargs: 传递给VAEC的其他参数
        """
        super().__init__()
        
        self.n_input = n_input
        self.n_labels = n_labels
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # 处理细胞类型权重
        ct_weight = None
        if weight_obs and ct_counts is not None:
            ct_prop = ct_counts / np.sum(ct_counts)
            ct_prop[ct_prop < 0.05] = 0.05
            ct_prop = ct_prop / np.sum(ct_prop)
            ct_weight = 1.0 / ct_prop
        
        # 创建核心VAEC模块
        self.module = VAEC(
            n_input=n_input,
            n_labels=n_labels,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            ct_weight=ct_weight,
            encode_covariates=encode_covariates,
            **module_kwargs
        )
    
    def forward(self, item, kl_weight=1.0):
        """
        前向传播
        
        Args:
            item: Dataset返回的字典 {'X': ..., 'labels': ..., 'batch': ...}
            kl_weight: KL散度权重
        
        Returns:
            dict: {'loss', 'reconstruction_loss', 'kl_local'}
        """
        return self.module.forward(item, kl_weight)
    
    @torch.no_grad()
    def get_latent_representation(self, item, give_mean=True):
        """
        获取潜在表示
        
        Args:
            item: Dataset返回的字典
            give_mean: 是否返回均值（否则返回采样）
        
        Returns:
            [n_cells, n_latent]
        """
        self.eval()
        x = item['X']
        labels = item['labels']
        batch = item.get('batch', None)
        
        inference_outputs = self.module.inference(x, labels, batch)
        return inference_outputs['qz_m'] if give_mean else inference_outputs['z']
    
    def get_vamp_prior(self, dataloader, p=15, device='cuda'):
        """
        计算VampPrior参数（用于初始化DestVI）
        
        Args:
            dataloader: 单细胞数据的DataLoader
            p: 每个细胞类型的伪输入数量
            device: 计算设备
        
        Returns:
            (mean_vprior, var_vprior, mp_vprior)
            每个都是 [n_latent, n_labels] 的numpy数组
        """
        self.eval()
        self.to(device)
        
        # 收集每个细胞类型的潜在表示
        latent_by_label = {i: [] for i in range(self.n_labels)}
        
        with torch.no_grad():
            for item in dataloader:
                item = {k: v.to(device) for k, v in item.items()}
                z = self.get_latent_representation(item, give_mean=True)
                labels = item['labels']
                
                for label_idx in range(self.n_labels):
                    mask = labels == label_idx
                    if mask.sum() > 0:
                        latent_by_label[label_idx].append(z[mask])
        
        # 计算每个细胞类型的统计量
        mean_vprior = []
        var_vprior = []
        mp_vprior = []
        
        for label_idx in range(self.n_labels):
            if len(latent_by_label[label_idx]) > 0:
                z_label = torch.cat(latent_by_label[label_idx], dim=0)
                
                # 最多采样p个
                if len(z_label) > p:
                    indices = torch.randperm(len(z_label))[:p]
                    z_label = z_label[indices]
                
                mean_vprior.append(z_label.mean(0).cpu().numpy())
                var_vprior.append(z_label.var(0).cpu().numpy())
                mp_vprior.append([len(z_label) / p])
            else:
                mean_vprior.append(np.zeros(self.n_latent))
                var_vprior.append(np.ones(self.n_latent))
                mp_vprior.append([0.0])
        
        return (
            np.array(mean_vprior).T,  # [n_latent, n_labels]
            np.array(var_vprior).T,
            np.array(mp_vprior).T
        )