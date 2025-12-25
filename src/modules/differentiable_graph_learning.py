"""可微图结构学习模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGraphLearner(nn.Module):
    """
    可微图学习器
    
    核心机制：
    1.输入特征 → MLP → 学习的嵌入空间
    2.计算相似度 → TopK + Gumbel-Softmax（可微）
    3.梯度通过任务损失反向传播到MLP
    """
    
    def __init__(
        self,
        n_input: int,
        hidden_dim: int = 128,
        k_neighbors: int = 15,
        temperature: float = 1.0,
        use_gumbel:  bool = True
    ):
        super().__init__()
        
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        
        print(f"🔥 可微图学习器:")
        print(f"  - 输入维度: {n_input}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - K邻居:  {k_neighbors}")
        print(f"  - 温度: {temperature}")
        print(f"  - 使用Gumbel:  {use_gumbel}")
        
        # 特征变换网络（学习最优的相似度空间）
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 边权重精炼器
        self.edge_refiner = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def compute_similarity(self, x: torch.Tensor):
        """
        计算节点相似度矩阵
        
        Args:
            x: [N, n_input]
        
        Returns:
            S: [N, N] 相似度矩阵
        """
        # 特征编码
        h = self.feature_encoder(x)  # [N, hidden_dim//2]
        
        # L2归一化
        h_norm = F.normalize(h, p=2, dim=-1)
        
        # 余弦相似度
        S = torch.mm(h_norm, h_norm.t())  # [N, N]
        
        return S
    
    def differentiable_topk(self, S: torch.Tensor, k: int):
        """
        可微的TopK操作（使用Straight-Through Estimator）
        
        Args:
            S:  [N, N] 相似度矩阵
            k: 保留的邻居数
        
        Returns:
            A: [N, N] 稀疏化后的邻接矩阵（可微）
        """
        N = S.size(0)
        device = S.device
        
        # 1.硬TopK选择
        topk_values, topk_indices = torch.topk(S, k=k, dim=1)
        
        # 2.创建mask矩阵
        mask = torch.zeros_like(S)
        mask.scatter_(1, topk_indices, 1.0)
        
        # 3.Straight-Through Estimator
        if self.training:
            # 前向：使用硬mask
            # 反向：梯度通过软相似度
            A = S * mask + (S - S.detach())
        else:
            A = S * mask
        
        # 4.对称化
        A = (A + A.t()) / 2
        
        return A
    
    def apply_spatial_prior(self, A_learned: torch.Tensor, 
                           spatial_edge_index: torch.Tensor,
                           spatial_edge_weight: torch.Tensor,
                           alpha: float = 0.3):
        """
        融合空间先验（防止学到的图偏离物理结构太远）
        
        Args: 
            A_learned: [N, N] 学到的邻接矩阵
            spatial_edge_index: [2, E] 空间图的边
            spatial_edge_weight: [E,] 空间图的权重
            alpha: 空间先验的权重
        
        Returns:
            A_fused: [N, N]
        """
        N = A_learned.size(0)
        device = A_learned.device
        
        # 将稀疏空间图转为稠密矩阵
        A_spatial = torch.zeros((N, N), device=device)
        A_spatial[spatial_edge_index[0], spatial_edge_index[1]] = spatial_edge_weight
        
        # 加权融合
        A_fused = (1 - alpha) * A_learned + alpha * A_spatial
        
        return A_fused
    
    def forward(self, x: torch.Tensor, 
                spatial_edge_index: torch.Tensor = None,
                spatial_edge_weight: torch.Tensor = None):
        """
        生成可微的图结构
        
        Args:
            x: [N, n_input] 节点特征
            spatial_edge_index: [2, E] 空间先验（可选）
            spatial_edge_weight: [E,] 空间先验权重
        
        Returns:
            A: [N, N] 学到的邻接矩阵（可微）
            stats: dict 统计信息
        """
        # 计算相似度
        S = self.compute_similarity(x)
        
        # 可微TopK稀疏化
        A = self.differentiable_topk(S, self.k_neighbors)
        
        # 融合空间先验
        if spatial_edge_index is not None: 
            A = self.apply_spatial_prior(A, spatial_edge_index, spatial_edge_weight, alpha=0.3)
        
        # 统计信息
        stats = {
            'graph_density': (A > 0.01).float().mean().item(),
            'avg_edge_weight': A.mean().item(),
            'max_edge_weight': A.max().item()
        }
        
        return A, stats