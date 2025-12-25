"""自定义损失函数"""
import torch
import torch.nn as nn


class GraphSparsityLoss(nn.Module):
    """
    图稀疏性损失
    
    防止学到全连接图
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, adj: torch.Tensor):
        """
        Args:
            adj: [N, N] 邻接矩阵
        
        Returns: 
            loss: 标量
        """
        if self.loss_type == 'l1':
            loss = adj.abs().mean()
        elif self.loss_type == 'l2':
            loss = (adj ** 2).mean()
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
        
        return loss