"""基于 Attention 的 GIB 剪枝模块（Soft Weighting 最终版）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add


class AttentionGIBPruner(nn.Module):
    """
    Attention-GIB 剪枝器（Soft Weighting 最终版）
    
    🔥 核心设计理念：
    1.Attention 打分（基于原始 PCA 特征）
    2.Local Competition（每个节点的邻居做 softmax）
    3.Soft Edge Weighting（不做 hard mask，完全可微）
    4.Entropy 正则（让分布变尖，但不强制稀疏度）
    
    🎯 剪枝 ≠ 删除边，而是让错误边的权重趋近于 0
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        entropy_weight: float = 0.1  # Entropy loss 的权重
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Query 和 Key 投影
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"🔥 Attention-GIB 剪枝器 (Soft Weighting 最终版):")
        print(f"  - 输入维度 (PCA): {input_dim}")
        print(f"  - 隐藏维度:  {hidden_dim}")
        print(f"  - 注意力头数: {num_heads}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Entropy Weight: {entropy_weight}")
        print(f"  - 🔥 Local Competition (per-node softmax)")
        print(f"  - 🔥 Soft Weighting (无 hard mask)")
        print(f"  - 🔥 完全可微分")
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.input_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
    
    def compute_attention_scores(self, x_pca, edge_index):
        """
        计算边的 Attention 原始分数
        
        Args:
            x_pca:  [N, input_dim] 原始 PCA 特征
            edge_index: [2, E] 边索引
        
        Returns:
            attention_logits: [E] 未归一化的 attention 分数
        """
        N = x_pca.size(0)
        
        # Step 1: 投影到 hidden_dim
        h = self.input_proj(x_pca)  # [N, hidden_dim]
        
        # Step 2: 计算 Query 和 Key
        Q = self.W_Q(h).view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        K = self.W_K(h).view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        src, dst = edge_index[0], edge_index[1]
        
        # Step 3: 计算边的 Attention 分数
        Q_src = Q[src]  # [E, num_heads, head_dim]
        K_dst = K[dst]  # [E, num_heads, head_dim]
        
        # Scaled Dot-Product Attention
        scores = (Q_src * K_dst).sum(dim=-1) / (self.head_dim ** 0.5)  # [E, num_heads]
        scores = scores.mean(dim=-1)  # [E] 多头平均
        
        # Step 4: 除以 temperature
        attention_logits = scores / self.temperature
        
        return attention_logits
    
    def local_competition(self, attention_logits, edge_index):
        """
        🔥 Local Competition:  每个节点的邻居做 softmax
        
        这是核心机制：
        - 对每个源节点 u，其所有邻边 (u→v) 竞争
        - 权重和 = 1 (per source node)
        - 重要的边获得高权重，不重要的边自然接近 0
        
        Args:
            attention_logits: [E] 未归一化的分数
            edge_index: [2, E] 边索引
        
        Returns:
            edge_weights: [E] 归一化后的边权重 (per-node sum = 1)
        """
        src = edge_index[0]  # [E]
        
        # 🔥 核心操作：对每个源节点的所有邻边做 softmax
        edge_weights = scatter_softmax(attention_logits, src, dim=0)
        
        return edge_weights
    
    def compute_entropy_loss(self, edge_weights, edge_index):
        """
        计算 Entropy Loss（鼓励分布变尖）
        
        目标：最小化每个节点的邻居权重分布的熵
        - 熵越小 → 分布越尖 → 权重越集中在少数边上
        - 自然产生稀疏性，不需要人为设定
        
        Args:
            edge_weights: [E] 归一化后的边权重
            edge_index: [2, E] 边索引
        
        Returns:
            entropy_loss: scalar
        """
        src = edge_index[0]
        
        # 计算每条边的 entropy 贡献
        eps = 1e-8
        log_weights = torch.log(edge_weights + eps)
        entropy_per_edge = -edge_weights * log_weights  # [E]
        
        # 对每个节点求和（每个节点的 entropy）
        entropy_per_node = scatter_add(entropy_per_edge, src, dim=0)  # [N]
        
        # 平均
        return entropy_per_node.mean()
    
    def forward(self, x_pca, spatial_edge_index, spatial_edge_weight=None):
        """
        前向传播
        
        Args: 
            x_pca: [N, input_dim] 原始 PCA 特征
            spatial_edge_index: [2, E] 原始空间图的边索引
            spatial_edge_weight: [E] 原始边权重（可选）
        
        Returns:
            pruned_edge_index: [2, E] 边索引（不变，不删边）
            pruned_edge_weight: [E] Soft 边权重
            edge_weights: [E] Attention 权重（用于统计）
            entropy_loss: scalar, Entropy 正则化损失
        """
        # 1.计算 Attention 原始分数
        attention_logits = self.compute_attention_scores(x_pca, spatial_edge_index)
        
        # 2.🔥 Local Competition:  每个节点的邻居做 softmax
        edge_weights = self.local_competition(attention_logits, spatial_edge_index)
        
        # 3.🔥 Soft Weighting: 不删边，只调整权重
        if spatial_edge_weight is not None: 
            # 原始权重 × Attention 权重
            pruned_edge_weight = spatial_edge_weight * edge_weights
        else:
            pruned_edge_weight = edge_weights
        
        # 4.🔥 不做 hard mask，所有边都保留
        pruned_edge_index = spatial_edge_index
        
        # 5.Entropy Loss（鼓励分布变尖）
        entropy_loss = self.compute_entropy_loss(edge_weights, spatial_edge_index)
        
        # 总损失（只有 entropy）
        total_loss = self.entropy_weight * entropy_loss
        
        return pruned_edge_index, pruned_edge_weight, edge_weights, total_loss
    
    def get_edge_importance(self, x_pca, spatial_edge_index):
        """获取边的重要性分数（用于可视化）"""
        with torch.no_grad():
            attention_logits = self.compute_attention_scores(x_pca, spatial_edge_index)
            edge_weights = self.local_competition(attention_logits, spatial_edge_index)
            return edge_weights
    
    def get_effective_sparsity(self, edge_weights, threshold=0.01):
        """
        计算"有效稀疏性"（权重 > threshold 的边的比例）
        
        这不是训练目标，只是用于监控
        """
        return (edge_weights > threshold).float().mean().item()
