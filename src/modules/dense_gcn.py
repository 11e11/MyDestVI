"""支持稠密邻接矩阵的GCN层"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGCNConv(nn.Module):
    """
    稠密GCN卷积层
    
    支持 [N, N] 的稠密邻接矩阵，梯度可以反向传播到A
    """
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x: [N, in_channels] 节点特征
            adj: [N, N] 邻接矩阵（稠密，可微）
        
        Returns:
            out: [N, out_channels]
        """
        # 度归一化
        deg = adj.sum(dim=1, keepdim=True) + 1e-12  # [N, 1]
        adj_norm = adj / deg  # 行归一化
        
        # GCN传播:  AXW
        support = torch.mm(x, self.weight)  # [N, out_channels]
        out = torch.mm(adj_norm, support)  # [N, out_channels]
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class DenseGCNEncoder(nn.Module):
    """
    稠密GCN编码器
    
    用于处理可微的学习图
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        print(f"🏗️ 稠密GCN编码器:")
        print(f"  - 层数: {num_layers}")
        print(f"  - 隐藏维度: {hidden_channels}")
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(DenseGCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(DenseGCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 最后一层
        self.convs.append(DenseGCNConv(hidden_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x:  [N, in_channels]
            adj: [N, N] 稠密邻接矩阵
        
        Returns: 
            out: [N, out_channels]
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, adj)
            x_new = bn(x_new)
            
            if i < self.num_layers - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                # 残差连接（如果维度匹配）
                if x.size(-1) == x_new.size(-1):
                    x = x_new + x
                else:
                    x = x_new
            else:
                x = x_new
        
        return x