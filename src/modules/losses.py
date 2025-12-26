"""损失函数模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSmoothnessLoss(nn.Module):
    """
    图平滑损失
    
    鼓励相邻节点具有相似的表示
    这有助于学习更平滑的图结构
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
        
        # 计算相邻节点表示的差异
        z_src = z[src]  # [E, D]
        z_dst = z[dst]  # [E, D]
        
        # L2距离
        diff = (z_src - z_dst).pow(2).sum(dim=-1)  # [E]
        
        # 加权（边权重越大，越要求相似）
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