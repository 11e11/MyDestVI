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
    
    @torch.no_grad()
    def get_vamp_prior(self, dataloader, p=15, device="cuda"):
        """
        计算 VampPrior 参数（用于初始化 DestVI 的 gamma 先验）
        返回:
            mean_vprior: (n_labels, p, n_latent)
            var_vprior:  (n_labels, p, n_latent)
            mp_vprior:   (n_labels, p)
        说明:
            - 对每个细胞类型做 KMeans(p) 聚类（若该类型细胞数 < p，则每个细胞单独成簇）
            - 每个簇的方差 = 平均 posterior var + 该簇 z 均值的样本方差（与 DestVI_SURF 一致）
            - mp_vprior 为各簇的混合权（簇内样本数 / 该类型总样本数）
        """
        from sklearn.cluster import KMeans
        import numpy as np
        import torch

        self.eval()
        self.to(device)

        # 收集每个细胞类型的 q(z|x) 的均值与方差
        z_mean_by_label = {i: [] for i in range(self.n_labels)}
        z_var_by_label = {i: [] for i in range(self.n_labels)}

        for item in dataloader:
            item = {k: v.to(device) for k, v in item.items()}
            x = item["X"]
            labels = item["labels"]
            batch = item.get("batch", None)

            out = self.module.inference(x, labels, batch)
            qz_m = out["qz_m"]  # [B, D]
            qz_v = out["qz_v"]  # [B, D]

            for label_idx in range(self.n_labels):
                mask = (labels == label_idx).view(-1)
                if mask.any():
                    z_mean_by_label[label_idx].append(qz_m[mask])  # [b_i, D]
                    z_var_by_label[label_idx].append(qz_v[mask])   # [b_i, D]

        D = self.n_latent
        nL = self.n_labels
        # 预分配输出，未使用的簇槽位将保持默认（mean=0, var=1, mp=0）
        mean_vprior = np.zeros((nL, p, D), dtype=np.float32)
        var_vprior = np.ones((nL, p, D), dtype=np.float32)
        mp_vprior = np.zeros((nL, p), dtype=np.float32)

        for label_idx in range(nL):
            if len(z_mean_by_label[label_idx]) == 0:
                continue
            z_m = torch.cat(z_mean_by_label[label_idx], dim=0)  # [N_l, D]
            z_v = torch.cat(z_var_by_label[label_idx], dim=0)   # [N_l, D]
            N_l = z_m.size(0)
            if N_l == 0:
                continue

            if p > 0 and N_l > p:
                # 按 z 的均值做 KMeans 聚类（与 SURF 对齐：n_init=30）
                km = KMeans(n_clusters=p, n_init=30, random_state=0)
                labels_k = torch.as_tensor(km.fit_predict(z_m.detach().cpu().numpy()), device=z_m.device)
                keys, counts = torch.unique(labels_k, return_counts=True)
                n_clusters = keys.numel()
            else:
                # 每个细胞单独作为一个簇（最多填满到 p 个槽位）
                n_clusters = min(p if p > 0 else N_l, N_l)
                labels_k = torch.arange(n_clusters, device=z_m.device).repeat_interleave(1)
                if labels_k.numel() < N_l:
                    # 剩余样本直接映射到已有簇（简单循环）
                    extra = N_l - labels_k.numel()
                    extra_assign = torch.arange(extra, device=z_m.device) % n_clusters
                    labels_k = torch.cat([labels_k, extra_assign], dim=0)
                keys = torch.arange(n_clusters, device=z_m.device)
                counts = torch.tensor([torch.sum(labels_k == k).item() for k in keys], device=z_m.device)

            # 为该 label 的每个簇计算 mean/var 和混合权
            for local_idx, k in enumerate(keys.tolist()):
                idx = (labels_k == k)
                z_m_k = z_m[idx]  # [n_k, D]
                z_v_k = z_v[idx]  # [n_k, D]
                if z_m_k.size(0) == 0:
                    continue

                # 簇均值
                mean_cluster = z_m_k.mean(dim=0)  # [D]
                # 方差 = 平均 posterior var + 簇内均值的样本方差（与 SURF 一致）
                var_cluster = z_v_k.mean(dim=0) + z_m_k.var(dim=0, unbiased=False)  # [D]

                if local_idx < p:
                    mean_vprior[label_idx, local_idx, :] = mean_cluster.detach().cpu().numpy()
                    var_vprior[label_idx, local_idx, :] = var_cluster.detach().cpu().numpy()
                    mp_vprior[label_idx, local_idx] = counts[local_idx].item() / float(N_l)

        return mean_vprior, var_vprior, mp_vprior