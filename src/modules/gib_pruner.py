"""GIB (Graph Information Bottleneck) 剪枝模块 - 改进版"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GIBPruner(nn.Module):
    """
    基于表达特征的图剪枝器 - 改进版
    
    🔥 关键改进：
    1.去掉max_keep_rate上限，允许GIB做极端判断
    2.edge feature从拼接改为差异+相似度，显式感知"是否跨层"
    3.添加显式的稀疏性正则，鼓励剪边
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        min_keep_rate: float = 0.05,  # 🔥 降低到0.05，允许激进剪枝
        temperature: float = 0.3  # 🔥 提高温度，让决策更软
    ):
        super().__init__()
        
        self.min_keep_rate = min_keep_rate
        self.temperature = temperature
        
        # 🔥 边特征提取器 - 改进版
        # 输入：[L2距离, 余弦相似度, 拼接特征]
        edge_feat_dim = 2 + node_dim * 2  # distance(1) + cosine(1) + concat(2*D)
        
        self.edge_scorer = nn.Sequential(
            nn.Linear(node_dim * 2 + node_dim + 1, hidden_dim),  # 修正输入维度 (拼接 + 差异 + 余弦)
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        print(f"🔥 GIB Pruner (改进版):")
        print(f"  - Node dim: {node_dim}")
        print(f"  - Min keep rate: {min_keep_rate} (无上限)")
        print(f"  - Temperature: {temperature}")
        print(f"  - Edge feature:  差异+相似度+拼接")
    
    def compute_edge_features(self, x, edge_index):
        """
        🔥 计算边特征 - 新版本
        
        包含三种信息：
        1.L2距离：显式表达"两点距离"
        2.余弦相似度：显式表达"两点方向是否一致"
        3.拼接特征：保留原始信息
        
        Args:
            x:  [N, D] 节点特征
            edge_index: [2, E] 边索引
        
        Returns:
            edge_feats: [E, 2+2D] 边特征
        """
        src, dst = edge_index[0], edge_index[1]

        x_src, x_dst = x[src], x[dst]  # [E, D], [E, D]
        
        # 拼接特征
        edge_feats_concat = torch.cat([x_src, x_dst], dim=-1)  # [E, 2*D]
        
        # 差异特征
        edge_feats_diff = torch.abs(x_src - x_dst)  # [E, D]

        # 余弦相似性
        cos_sim = F.cosine_similarity(x_src, x_dst, dim=-1).unsqueeze(-1)  # [E, 1]

        # 拼接所有特征
        edge_feats = torch.cat([edge_feats_concat, edge_feats_diff, cos_sim], dim=-1)  # [E, 2D + D + 1]

        assert edge_feats.size(1) == x.size(1) * 3 + 1, f"Unexpected edge_feats size: {edge_feats.size(1)}"
        
        return edge_feats
    
    def compute_edge_scores(self, x, edge_index):
        """
        计算边的保留概率
        
        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
        
        Returns:
            keep_probs: [E,] 保留概率
            raw_logits: [E,] 原始logits（用于分析）
        """
        # 🔥 使用新的边特征
        edge_feats = self.compute_edge_features(x, edge_index)  # [E, 2+2D]
        
        # 打分
        logits = self.edge_scorer(edge_feats).squeeze(-1)  # [E,]
        
        # 🔥 去掉max_keep_rate上限
        # 归一化到[min_keep_rate, 1.0]
        keep_probs = torch.sigmoid(logits / self.temperature)
        keep_probs = self.min_keep_rate + (1.0 - self.min_keep_rate) * keep_probs
        
        return keep_probs, logits
    
    def compute_sparsity_loss(self, keep_probs):
        """
        🔥 稀疏性损失：鼓励GIB剪边
        
        目标：让keep_probs的均值接近目标稀疏度（例如0.5）
        """
        target_sparsity = 0.5  # 目标：保留50%的边
        current_sparsity = keep_probs.mean()
        
        # L1损失
        sparsity_loss = torch.abs(current_sparsity - target_sparsity)
        
        return sparsity_loss
    
    def forward(self, x, spatial_edge_index, spatial_edge_weight=None):
        """
        生成剪枝后的增强图
        
        Args:  
            x: [N, D] 节点特征
            spatial_edge_index: [2, E] 原始空间图边
            spatial_edge_weight: [E,] 边权重（可选）
        
        Returns:
            aug_edge_index: [2, E'] 增强图边
            aug_edge_weight: [E'] 增强图权重
            keep_probs: [E,] 每条边的保留概率（用于分析）
            sparsity_loss: 稀疏性损失
        """
        device = x.device
        E = spatial_edge_index.size(1)
        
        # 计算保留概率
        keep_probs, raw_logits = self.compute_edge_scores(x, spatial_edge_index)  # [E,]
        
        if self.training:
            # 训练时：Gumbel-Softmax采样（可微）
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(keep_probs) + 1e-8) + 1e-8)
            logits = torch.log(keep_probs + 1e-8) - torch.log(1 - keep_probs + 1e-8)
            keep_mask = torch.sigmoid((logits + gumbel_noise) / self.temperature) > 0.5
        else:
            # 测试时：确定性采样
            keep_mask = keep_probs > 0.5
        
        # 筛选保留的边
        aug_edge_index = spatial_edge_index[: , keep_mask]
        
        if spatial_edge_weight is not None:
            aug_edge_weight = spatial_edge_weight[keep_mask] * keep_probs[keep_mask]
        else:
            aug_edge_weight = keep_probs[keep_mask]
        
        # 🔥 计算稀疏性损失
        sparsity_loss = self.compute_sparsity_loss(keep_probs)
        
        return aug_edge_index, aug_edge_weight, keep_probs, sparsity_loss