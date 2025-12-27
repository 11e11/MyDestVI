"""GIB (Graph Information Bottleneck) 剪枝模块 - 救援版"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GIBPruner(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        min_keep_rate: float = 0.05,
        max_keep_rate: float = 0.8, # 这个参数现在只用于 Top-K 截断
        temperature: float = 0.3
    ):
        super().__init__()
        
        self.min_keep_rate = min_keep_rate
        self.max_keep_rate = max_keep_rate
        self.temperature = temperature
        
        # 边特征提取器
        self.edge_scorer = nn.Sequential(
            nn.Linear(node_dim * 2 + node_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            # 最后一层：输出 Logits
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 🔥🔥🔥 关键救命修改：初始化偏置为 +3.0
        # Sigmoid(3.0) ≈ 0.95
        # 即使经过 Linear 初始化波动，也能保证初始概率 > 0.5
        # 确保训练开始时，图是连通的！
        nn.init.constant_(self.edge_scorer[-1].bias, 0.0)
        
        print(f"🔥 GIB Pruner (救援版):")
        print(f"  - Init Bias: +3.0 (Start Fully Connected)")
        print(f"  - Hard Pruning limit: {max_keep_rate}")
        print(f"  - Loss: Mean L1")

    def compute_edge_features(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        x_src, x_dst = x[src], x[dst]
        edge_feats_concat = torch.cat([x_src, x_dst], dim=-1)
        edge_feats_diff = torch.abs(x_src - x_dst)
        cos_sim = F.cosine_similarity(x_src, x_dst, dim=-1).unsqueeze(-1)
        return torch.cat([edge_feats_concat, edge_feats_diff, cos_sim], dim=-1)
    
    def compute_edge_scores(self, x, edge_index):
        edge_feats = self.compute_edge_features(x, edge_index)
        logits = self.edge_scorer(edge_feats).squeeze(-1)
        
        # 🔥 修改：让 keep_probs 能够达到 1.0 (表示连接强度)
        # 不要在这里用 max_keep_rate 限制它，否则会误杀
        keep_probs = torch.sigmoid(logits / self.temperature)
        keep_probs = self.min_keep_rate + (1.0 - self.min_keep_rate) * keep_probs
        
        return keep_probs, logits
    
    def forward(self, x, spatial_edge_index, spatial_edge_weight=None):
        keep_probs, _ = self.compute_edge_scores(x, spatial_edge_index)
        
        # 1. 生成 Mask
        if self.training:
            # 训练时：Bernoulli 采样 (随机性)
            keep_mask = torch.bernoulli(keep_probs).bool()
        else:
            # 推理时：硬截断
            keep_mask = keep_probs > 0.5
            
        # 2. 🔥 强制 Top-K 截断 (Budget Control)
        # 只有在这里使用 max_keep_rate 来控制数量，而不是限制强度
        if self.max_keep_rate < 1.0:
            current_keep_rate = keep_probs.mean() # 检查平均保留率
            if current_keep_rate > self.max_keep_rate:
                # 找出阈值
                num_edges = keep_probs.size(0)
                k = int(num_edges * self.max_keep_rate)
                # 使用负值求 topk 来找第 k 大
                threshold = torch.kthvalue(keep_probs, num_edges - k + 1).values
                # 只有概率足够大的才保留
                keep_mask = keep_mask & (keep_probs >= threshold)

        aug_edge_index = spatial_edge_index[:, keep_mask]
        
        if spatial_edge_weight is not None:
            aug_edge_weight = spatial_edge_weight[keep_mask] * keep_probs[keep_mask]
        else:
            aug_edge_weight = keep_probs[keep_mask]
        
        # L1 Loss
        sparsity_loss = keep_probs.mean()
        
        return aug_edge_index, aug_edge_weight, keep_probs, sparsity_loss