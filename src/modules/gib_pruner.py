"""GIB (Graph Information Bottleneck) 剪枝模块 - 语义感知版（STAGUE 模式）"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GIBPruner(nn.Module):
    """
    语义感知的 GIB 剪枝器
    
    🔥 核心改进：
    - 输入改为编码后的语义特征（不再是原始 PCA）
    - 计算语义距离（不再是物理特征距离）
    - 更准确地识别跨层边界
    """
    
    def __init__(
        self,
        node_dim: int,  # 语义特征维度（hidden_dim）
        hidden_dim: int = 64,
        min_keep_rate: float = 0.05,
        max_keep_rate: float = 0.8,
        temperature: float = 0.3,
        use_soft_mask: bool = True
    ):
        super().__init__()
        
        self.min_keep_rate = min_keep_rate
        self.max_keep_rate = max_keep_rate
        self.temperature = temperature
        self.use_soft_mask = use_soft_mask
        
        # 🔥 边特征提取器（输入是语义特征）
        self.edge_scorer = nn.Sequential(
            nn.Linear(node_dim * 2 + node_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 🔥 初始化偏置为 +3.0（确保训练开始时保留大部分边）
        nn.init.constant_(self.edge_scorer[-1].bias, 3.0)
        
        print(f"🔥 GIB Pruner (语义感知版 - STAGUE):")
        print(f"  - Input:  Semantic Features (from Encoder)")
        print(f"  - Init Bias: +3.0 -> P(keep) ≈ 0.95")
        print(f"  - Max Keep Rate: {max_keep_rate} (软约束)")
        print(f"  - Use Soft Mask: {use_soft_mask}")

    def compute_edge_features(self, h_semantic, edge_index):
        """
        计算边特征（基于语义特征）
        
        Args:
            h_semantic: [N, hidden_dim] 语义特征（来自 Encoder）
            edge_index: [2, E] 边索引
        """
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = h_semantic[src], h_semantic[dst]
        
        # 1.拼接语义特征
        edge_feats_concat = torch.cat([h_src, h_dst], dim=-1)
        
        # 2.语义差异（L1）
        edge_feats_diff = torch.abs(h_src - h_dst)
        
        # 3.语义相似度（余弦）
        cos_sim = F.cosine_similarity(h_src, h_dst, dim=-1).unsqueeze(-1)
        
        return torch.cat([edge_feats_concat, edge_feats_diff, cos_sim], dim=-1)
    
    def compute_edge_scores(self, h_semantic, edge_index):
        """计算边的保留概率"""
        edge_feats = self.compute_edge_features(h_semantic, edge_index)
        logits = self.edge_scorer(edge_feats).squeeze(-1)
        
        # Sigmoid + 缩放到 [min_keep_rate, 1.0]
        keep_probs = torch.sigmoid(logits / self.temperature)
        keep_probs = self.min_keep_rate + (1.0 - self.min_keep_rate) * keep_probs
        
        return keep_probs, logits
    
    def forward(self, h_semantic, spatial_edge_index, spatial_edge_weight=None):
        """
        前向传播（语义感知版）
        
        Args:
            h_semantic: [N, hidden_dim] 语义特征（来自 Anchor 编码器）
            spatial_edge_index: [2, E] 原始空间图的边索引
            spatial_edge_weight: [E] 原始边权重
        
        Returns:
            aug_edge_index: 剪枝后的边索引
            aug_edge_weight:  剪枝后的边权重
            keep_probs: 保留概率
            sparsity_loss: 稀疏性损失
        """
        keep_probs, _ = self.compute_edge_scores(h_semantic, spatial_edge_index)
        num_edges = keep_probs.size(0)
        
        # 1.训练时：软权重（不采样）
        if self.training and self.use_soft_mask:
            keep_mask = keep_probs > self.min_keep_rate
        else:
            # 推理时：硬截断
            keep_mask = keep_probs > 0.5
        
        # 2.自适应 Top-K 截断（软约束）
        if self.max_keep_rate < 1.0:
            actual_keep_rate = keep_mask.float().mean().item()
            
            if actual_keep_rate > self.max_keep_rate:
                k = int(num_edges * self.max_keep_rate)
                threshold = torch.kthvalue(keep_probs, num_edges - k + 1).values
                keep_mask = keep_mask & (keep_probs >= threshold)
        
        # 3.应用 mask
        aug_edge_index = spatial_edge_index[: , keep_mask]
        
        # 4.边权重 = 原始权重 × 保留概率
        if spatial_edge_weight is not None:
            aug_edge_weight = spatial_edge_weight[keep_mask] * keep_probs[keep_mask]
        else: 
            aug_edge_weight = keep_probs[keep_mask]
        
        # 5.稀疏性损失（L1 正则）
        sparsity_loss = keep_probs.mean()
        
        return aug_edge_index, aug_edge_weight, keep_probs, sparsity_loss