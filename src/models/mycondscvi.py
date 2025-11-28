# """
# CondSCVI模型 - 简化封装
# """
# import torch
# import torch.nn as nn
# import numpy as np

# from src.modules.myvaec import VAEC


# class CondSCVI(nn.Module):
#     """Conditional scVI model - 用户接口层"""

#     def __init__(
#         self,
#         n_input: int,
#         n_labels: int,
#         n_batch: int = 1,
#         n_hidden: int = 128,
#         n_latent: int = 5,
#         n_layers: int = 2,
#         dropout_rate: float = 0.05,
#         weight_obs: bool = False,
#         ct_counts: np.ndarray = None,
#         encode_covariates: bool = False,
#         **module_kwargs
#     ):
#         """
#         Args:
#             n_input: 基因数量
#             n_labels: 细胞类型数量
#             n_batch: 批次数量
#             n_hidden: 隐藏层神经元数
#             n_latent: 潜在空间维度
#             n_layers: 网络层数
#             dropout_rate: Dropout率
#             weight_obs: 是否对低频细胞类型重加权
#             ct_counts: 细胞类型计数（用于weight_obs）
#             encode_covariates: 是否在编码器中使用协变量
#             **module_kwargs: 传递给VAEC的其它参数
#         """
#         super().__init__()

#         self.n_input = n_input
#         self.n_labels = n_labels
#         self.n_batch = n_batch
#         self.n_hidden = n_hidden
#         self.n_latent = n_latent
#         self.n_layers = n_layers
#         self.dropout_rate = dropout_rate

#         # 处理细胞类型权重
#         ct_weight = None
#         if weight_obs and ct_counts is not None:
#             ct_prop = ct_counts / np.sum(ct_counts)
#             ct_prop[ct_prop < 0.05] = 0.05
#             ct_prop = ct_prop / np.sum(ct_prop)
#             ct_weight = 1.0 / ct_prop

#         # 创建核心VAEC模块
#         self.module = VAEC(
#             n_input=n_input,
#             n_labels=n_labels,
#             n_batch=n_batch,
#             n_hidden=n_hidden,
#             n_latent=n_latent,
#             n_layers=n_layers,
#             dropout_rate=dropout_rate,
#             ct_weight=ct_weight,
#             encode_covariates=encode_covariates,
#             **module_kwargs
#         )

#     def forward(self, item, kl_weight=1.0, kl_library_weight: float = 1.0):
#         """
#         前向传播

#         Args:
#             item: Dataset返回的字典 {'X': ..., 'labels': ..., 'batch': ...}
#             kl_weight: z 的 KL 散度权重
#             kl_library_weight: library 的 KL 散度权重（默认 1.0）
#         Returns:
#             dict: {'loss', 'reconstruction_loss', 'kl_local', 'kl_library'}
#         """
#         return self.module.forward(item, kl_weight, kl_library_weight)

#     @torch.no_grad()
#     def get_latent_representation(self, item, give_mean=True):
#         """
#         获取潜在表示

#         Args:
#             item: Dataset返回的字典
#             give_mean: 是否返回均值（否则返回采样）
#         Returns:
#             [n_cells, n_latent]
#         """
#         self.eval()
#         x = item['X']
#         labels = item['labels']
#         batch = item.get('batch', None)

#         inference_outputs = self.module.inference(x, labels, batch)
#         return inference_outputs['qz_m'] if give_mean else inference_outputs['z']

#     @torch.no_grad()
#     def get_vamp_prior(self, dataloader, p=15, device='cuda'):
#         """
#         生成 DestVI 所需的 VampPrior 混合先验参数。
#         ...
#         """
#         from sklearn.cluster import KMeans
#         import numpy as np
#         import torch

#         self.eval()
#         self.to(device)

#         z_mean_by_label = {i: [] for i in range(self.n_labels)}
#         z_var_by_label = {i: [] for i in range(self.n_labels)}

#         # 收集所有细胞的 q(z|x) 参数
#         for item in dataloader:
#             item = {k: v.to(device) for k, v in item.items()}
#             out = self.module.inference(item['X'], item['labels'], item.get('batch', None))
#             qz_m = out['qz_m']
#             qz_v = out['qz_v']
#             labels = item['labels']

#             for label_idx in range(self.n_labels):
#                 mask = (labels == label_idx).view(-1)
#                 if mask.any():
#                     z_mean_by_label[label_idx].append(qz_m[mask])
#                     z_var_by_label[label_idx].append(qz_v[mask])

#         D = self.n_latent
#         L = self.n_labels
#         mean_vprior = np.zeros((L, p, D), dtype=np.float32)
#         var_vprior = np.ones((L, p, D), dtype=np.float32)
#         mp_vprior = np.zeros((L, p), dtype=np.float32)

#         for label_idx in range(L):
#             if len(z_mean_by_label[label_idx]) == 0:
#                 continue

#             z_m = torch.cat(z_mean_by_label[label_idx], dim=0)  # [N_l, D]
#             z_v = torch.cat(z_var_by_label[label_idx], dim=0)   # [N_l, D]
#             N_l = z_m.size(0)

#             if N_l > p and p > 0:
#                 km = KMeans(n_clusters=p, n_init=30, random_state=0)
#                 cluster_ids = torch.as_tensor(km.fit_predict(z_m.detach().cpu().numpy()), device=z_m.device)
#                 keys, counts = torch.unique(cluster_ids, return_counts=True)
#             else:
#                 # 将每个样本作为一个簇或循环映射到 p 个槽位
#                 n_clusters = min(p if p > 0 else N_l, N_l)
#                 base = torch.arange(n_clusters, device=z_m.device)
#                 if N_l > n_clusters:
#                     extra = (torch.arange(N_l - n_clusters, device=z_m.device) % n_clusters)
#                     cluster_ids = torch.cat([base, extra], dim=0)
#                 else:
#                     cluster_ids = base
#                 keys, counts = torch.unique(cluster_ids, return_counts=True)

#             for local_idx, k in enumerate(keys.tolist()):
#                 if local_idx >= p:
#                     break
#                 mask = (cluster_ids == k)
#                 z_m_k = z_m[mask]
#                 z_v_k = z_v[mask]
#                 n_k = z_m_k.size(0)
#                 if n_k == 0:
#                     continue
#                 # 簇均值
#                 mean_cluster = z_m_k.mean(dim=0)
#                 # 簇方差 = 平均 posterior 方差 + 均值向量样本方差
#                 var_cluster = z_v_k.mean(dim=0) + z_m_k.var(dim=0, unbiased=False)

#                 mean_vprior[label_idx, local_idx, :] = mean_cluster.detach().cpu().numpy()
#                 var_vprior[label_idx, local_idx, :] = var_cluster.detach().cpu().numpy()
#                 mp_vprior[label_idx, local_idx] = float(n_k) / float(N_l)

#         return mean_vprior, var_vprior, mp_vprior

#     def export_decoder_state(self):
#         """
#         返回 decoder_backbone（固定条件版解码器）的 state_dict；
#         若不存在则回退到 decoder（兼容旧命名）。
#         """
#         m = getattr(self, "module", None)
#         if m is None:
#             return None
#         dec = getattr(m, "decoder_backbone", None)
#         if dec is None:
#             dec = getattr(m, "decoder", None)
#         return dec.state_dict() if dec is not None else None

#     def export_px_decoder_state(self):
#         m = getattr(self, "module", None)
#         if m is None:
#             return None
#         px = getattr(m, "px_decoder", None)
#         return px.state_dict() if px is not None else None

#     def export_px_r(self):
#         m = getattr(self, "module", None)
#         if m is None:
#             return None
#         return m.px_r.detach().cpu().numpy()

#     def export_dropout_decoder(self):
#         m = getattr(self, "module", None)
#         if m is None:
#             return 0.05
#         return getattr(m, "dropout_rate", 0.05)

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
        dropout_rate: float = 0.1,
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
    def get_vamp_prior(self, dataloader, p=15, device='cuda'):
        """
        生成 DestVI 所需的 VampPrior 混合先验参数。

        返回:
            mean_vprior: (n_labels, p, n_latent)
            var_vprior:  (n_labels, p, n_latent)
            mp_vprior:   (n_labels, p)

        做法（与 DestVI_SURF 对齐）：
        - 对每个细胞类型的潜在均值 q(z|x).loc 做 KMeans(n_clusters=p, n_init=30)
          若该类型样本数 < p，则每个细胞各自成为一个簇（或复用到前面簇）
        - 每个簇的方差 = 该簇 posterior var 的均值 + 均值向量的样本方差
        - 混合权 mp = 簇样本数 / 该类型总样本数
        """
        from sklearn.cluster import KMeans
        import numpy as np
        import torch

        self.eval()
        self.to(device)

        z_mean_by_label = {i: [] for i in range(self.n_labels)}
        z_var_by_label = {i: [] for i in range(self.n_labels)}

        # 收集所有细胞的 q(z|x) 参数
        for item in dataloader:
            item = {k: v.to(device) for k, v in item.items()}
            out = self.module.inference(item['X'], item['labels'], item.get('batch', None))
            qz_m = out['qz_m']
            qz_v = out['qz_v']
            labels = item['labels']

            for label_idx in range(self.n_labels):
                mask = (labels == label_idx).view(-1)
                if mask.any():
                    z_mean_by_label[label_idx].append(qz_m[mask])
                    z_var_by_label[label_idx].append(qz_v[mask])

        D = self.n_latent
        L = self.n_labels
        mean_vprior = np.zeros((L, p, D), dtype=np.float32)
        var_vprior = np.ones((L, p, D), dtype=np.float32)
        mp_vprior = np.zeros((L, p), dtype=np.float32)

        for label_idx in range(L):
            if len(z_mean_by_label[label_idx]) == 0:
                continue

            z_m = torch.cat(z_mean_by_label[label_idx], dim=0)  # [N_l, D]
            z_v = torch.cat(z_var_by_label[label_idx], dim=0)   # [N_l, D]
            N_l = z_m.size(0)

            if N_l > p and p > 0:
                km = KMeans(n_clusters=p, n_init=30, random_state=0)
                cluster_ids = torch.as_tensor(km.fit_predict(z_m.detach().cpu().numpy()), device=z_m.device)
                keys, counts = torch.unique(cluster_ids, return_counts=True)
            else:
                # 将每个样本作为一个簇或循环映射到 p 个槽位
                n_clusters = min(p if p > 0 else N_l, N_l)
                base = torch.arange(n_clusters, device=z_m.device)
                if N_l > n_clusters:
                    extra = (torch.arange(N_l - n_clusters, device=z_m.device) % n_clusters)
                    cluster_ids = torch.cat([base, extra], dim=0)
                else:
                    cluster_ids = base
                keys, counts = torch.unique(cluster_ids, return_counts=True)

            for local_idx, k in enumerate(keys.tolist()):
                if local_idx >= p:
                    break
                mask = (cluster_ids == k)
                z_m_k = z_m[mask]
                z_v_k = z_v[mask]
                n_k = z_m_k.size(0)
                if n_k == 0:
                    continue
                # 簇均值
                mean_cluster = z_m_k.mean(dim=0)
                # 簇方差 = 平均 posterior 方差 + 均值向量样本方差
                var_cluster = z_v_k.mean(dim=0) + z_m_k.var(dim=0, unbiased=False)

                mean_vprior[label_idx, local_idx, :] = mean_cluster.detach().cpu().numpy()
                var_vprior[label_idx, local_idx, :] = var_cluster.detach().cpu().numpy()
                mp_vprior[label_idx, local_idx] = float(n_k) / float(N_l)

        return mean_vprior, var_vprior, mp_vprior
    
    def export_decoder_state(self):
        """
        返回 decoder_backbone（固定条件版解码器）的 state_dict；
        若不存在则回退到 decoder（兼容旧命名）。
        """
        m = getattr(self, "module", None)
        if m is None:
            return None
        dec = getattr(m, "decoder_backbone", None)
        if dec is None:
            dec = getattr(m, "decoder", None)
        return dec.state_dict() if dec is not None else None

    def export_px_decoder_state(self):
        m = getattr(self, "module", None)
        if m is None:
            return None
        px = getattr(m, "px_decoder", None)
        return px.state_dict() if px is not None else None

    def export_px_r(self):
        m = getattr(self, "module", None)
        if m is None:
            return None
        return m.px_r.detach().cpu().numpy()

    def export_dropout_decoder(self):
        m = getattr(self, "module", None)
        if m is None:
            return 0.05
        return getattr(m, "dropout_rate", 0.05)