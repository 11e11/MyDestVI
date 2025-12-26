"""GIB (Graph Information Bottleneck) 剪枝模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GIBPruner(nn.Module):
    """
    基于表达特征的图剪枝器
    
    核心逻辑：
    1.输入空间KNN图 + 节点表达特征
    2.对每条边打分（保留概率）
    3.伯努利采样生成增强图
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        min_keep_rate: float = 0.4,
        max_keep_rate: float = 0.8,
        temperature: float = 0.5
    ):
        super().__init__()
        
        self.min_keep_rate = min_keep_rate
        self.max_keep_rate = max_keep_rate
        self.temperature = temperature
        
        # 边特征提取器
        self.edge_scorer = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出logits
        )
        
        print(f"🔥 GIB Pruner:")
        print(f"  - Node dim: {node_dim}")
        print(f"  - Keep rate: [{min_keep_rate}, {max_keep_rate}]")
        print(f"  - Temperature: {temperature}")
    
    def compute_edge_scores(self, x, edge_index):
        """
        计算边的保留概率
        
        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
        
        Returns:
            keep_probs: [E,] 保留概率
        """
        src, dst = edge_index[0], edge_index[1]
        
        # 边特征 = concat(src_feat, dst_feat)
        edge_feats = torch.cat([x[src], x[dst]], dim=-1)  # [E, 2D]
        
        # 打分
        logits = self.edge_scorer(edge_feats).squeeze(-1)  # [E,]
        
        # 归一化到[min, max]范围
        keep_probs = torch.sigmoid(logits / self.temperature)
        keep_probs = self.min_keep_rate + (self.max_keep_rate - self.min_keep_rate) * keep_probs
        
        return keep_probs
    
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
        """
        device = x.device
        E = spatial_edge_index.size(1)
        
        # 计算保留概率
        keep_probs = self.compute_edge_scores(x, spatial_edge_index)  # [E,]
        
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
        
        return aug_edge_index, aug_edge_weight, keep_probs
    
