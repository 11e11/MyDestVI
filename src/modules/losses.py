"""损失函数模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """
    🔥 空间一致性损失
    
    目标：鼓励空间上相邻的节点具有相似的预测分布 Q
    这对薄层结构至关重要，能"连接"断裂的薄层
    """
    
    def __init__(self, loss_type='mse', reduction='mean'):
        """
        Args:
            loss_type: 'mse' (均方误差) 或 'kl' (KL 散度)
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, Q, spatial_edge_index):
        """
        Args:
            Q: [N, K] 节点到原型的分配概率（softmax 后）
            spatial_edge_index: [2, E] 空间邻接边（原始的，非 GIB 剪枝的）
        
        Returns:
            loss: 标量
        """
        src, dst = spatial_edge_index[0], spatial_edge_index[1]
        
        Q_src = Q[src]  # [E, K]
        Q_dst = Q[dst]  # [E, K]
        
        if self.loss_type == 'mse':
            # 均方误差：简单直接
            diff = (Q_src - Q_dst).pow(2).sum(dim=-1)  # [E]
        elif self.loss_type == 'kl':
            # KL 散度：更符合概率分布的度量
            eps = 1e-8
            kl_loss = (Q_src * (torch.log(Q_src + eps) - torch.log(Q_dst + eps))).sum(dim=-1)
            diff = kl_loss
        else:
            raise ValueError(f"不支持的损失类型:  {self.loss_type}")
        
        if self.reduction == 'mean': 
            loss = diff.mean()
        elif self.reduction == 'sum': 
            loss = diff.sum()
        else:
            loss = diff
        
        return loss


class GraphSmoothnessLoss(nn.Module):
    """
    图平滑损失
    
    鼓励相邻节点具有相似的表示
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, z, edge_index, edge_weight):
        """
        Args:
            z: [N, D] 节点表示
            edge_index: [2, E] 边索引
            edge_weight: [E] 边权重
        
        Returns:
            loss: 标量
        """
        src, dst = edge_index[0], edge_index[1]
        
        z_src = z[src]
        z_dst = z[dst]
        
        # L2 距离
        diff = (z_src - z_dst).pow(2).sum(dim=-1)
        
        # 加权
        weighted_diff = diff * edge_weight
        
        if self.reduction == 'mean':
            loss = weighted_diff.mean()
        elif self.reduction == 'sum':
            loss = weighted_diff.sum()
        else:
            loss = weighted_diff
        
        return loss


class GraphSparsityLoss(nn.Module):
    """
    图稀疏性损失
    
    防止所有边权重都变得很大
    """
    
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, edge_weight):
        """
        Args: 
            edge_weight: [E] 边权重
        
        Returns:
            loss: 标量
        """
        if self.loss_type == 'l1':
            loss = edge_weight.abs().mean()
        elif self.loss_type == 'l2':
            loss = (edge_weight ** 2).mean()
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
        
        return loss