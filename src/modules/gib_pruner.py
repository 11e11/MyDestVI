"""基于 Attention 的 GIB 剪枝模块（修复版：断开数值联系 + Entropy Loss）"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGIBPruner(nn.Module):
    """
    Attention-GIB 剪枝器（修复版）
    
    🔥 核心改进：
    1.断开 pruned_edge_weight 和 attention_probs 的数值联系
       - Prob 只决定边的存亡（0或1），不影响边的权重强度
    2.引入 Entropy Loss 促进二元决策
       - 避免模型停留在 0.5 左右"混日子"
    """
    
    def __init__(
        self,
        node_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        gumbel_tau: float = 0.5,
        entropy_weight: float = 0.01  # 🔥 NEW: Entropy Loss 权重
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.entropy_weight = entropy_weight
        
        assert node_dim % num_heads == 0, "node_dim 必须能被 num_heads 整除"
        
        # Query 和 Key 投影
        self.W_Q = nn.Linear(node_dim, node_dim, bias=False)
        self.W_K = nn.Linear(node_dim, node_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"🔥 Attention-GIB 剪枝器 (修复版):")
        print(f"  - 输入维度: {node_dim}")
        print(f"  - 注意力头数: {num_heads}")
        print(f"  - Temperature: {temperature}")
        print(f"  - 使用 Gumbel:  {use_gumbel}")
        print(f"  - Entropy Weight: {entropy_weight}")
        print(f"  - 🔥 边权重与 Prob 解耦")
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
    
    def compute_attention_scores(self, h_base, edge_index):
        """
        计算边的 Attention 分数
        
        Args: 
            h_base: [N, node_dim] 基础特征
            edge_index: [2, E] 边索引
        
        Returns:
            attention_probs: [E] 每条边的保留概率
        """
        N = h_base.size(0)
        
        # Query 和 Key
        Q = self.W_Q(h_base).view(N, self.num_heads, self.head_dim)
        K = self.W_K(h_base).view(N, self.num_heads, self.head_dim)
        
        src, dst = edge_index[0], edge_index[1]
        
        # 计算边的 Attention 分数
        Q_src = Q[src]
        K_dst = K[dst]
        
        # Scaled Dot-Product Attention
        scores = (Q_src * K_dst).sum(dim=-1) / (self.head_dim ** 0.5) + 2.0
        scores = scores.mean(dim=-1)  # 多头平均
        
        # 缩放到 [0, 1]
        attention_probs = torch.sigmoid(scores / self.temperature)
        
        return attention_probs
    
    def straight_through_estimator(self, probs, hard:  bool = True):
        """
        STE (Straight-Through Estimator) 离散化
        
        Args: 
            probs: [E] 概率值
            hard: 是否使用硬阈值
        
        Returns: 
            mask: [E] 离散化的 mask (0 或 1)
        """
        if not self.training or not hard:
            # 推理时：直接硬截断
            return (probs > 0.5).float()
        
        if self.use_gumbel:
            # 训练时：Gumbel-Softmax 采样
            logits = torch.stack([
                torch.log(probs + 1e-8),
                torch.log(1 - probs + 1e-8)
            ], dim=-1)
            
            # Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits_with_noise = (logits + gumbel_noise) / self.gumbel_tau
            
            soft_mask = F.softmax(logits_with_noise, dim=-1)[:, 0]
            
            # Straight-Through
            hard_mask = (soft_mask > 0.5).float()
            mask = hard_mask - soft_mask.detach() + soft_mask
        else:
            # 简单的硬阈值 + STE
            hard_mask = (probs > 0.5).float()
            mask = hard_mask - probs.detach() + probs
        
        return mask
    
    def compute_entropy_loss(self, probs):
        """
        🔥 计算 Entropy Loss
        
        目标：鼓励 probs 趋向 0 或 1，避免停留在 0.5
        
        Args:
            probs: [E] 概率值
        
        Returns: 
            entropy_loss: scalar
        """
        # Binary Entropy:  H(p) = -p*log(p) - (1-p)*log(1-p)
        # 当 p=0 或 p=1 时，H(p)=0（确定性最大）
        # 当 p=0.5 时，H(p)=1（不确定性最大）
        eps = 1e-8
        entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        
        # 我们希望最小化 entropy（即鼓励确定性决策）
        return entropy.mean()
    
    def forward(self, h_base, spatial_edge_index, spatial_edge_weight=None):
        """
        前向传播
        
        Args:
            h_base: [N, node_dim] 基础特征
            spatial_edge_index: [2, E] 原始空间图的边索引
            spatial_edge_weight: [E] 原始边权重
        
        Returns: 
            pruned_edge_index: 剪枝后的边索引
            pruned_edge_weight: 剪枝后的边权重
            attention_probs: [E] Attention 概率
            total_loss: 总的正则化损失（sparsity + entropy）
        """
        # 1.计算 Attention 分数
        attention_probs = self.compute_attention_scores(h_base, spatial_edge_index)
        
        # 2.STE 离散化
        mask = self.straight_through_estimator(attention_probs, hard=True)
        
        # 3.应用 mask
        keep_indices = mask > 0.5
        pruned_edge_index = spatial_edge_index[: , keep_indices]
        
        # 🔥 关键修改：边权重不再乘以 attention_probs
        # Prob 只决定边的存亡，不影响权重强度
        if spatial_edge_weight is not None:
            pruned_edge_weight = spatial_edge_weight[keep_indices]  # ✅ 保持原始权重
        else:
            pruned_edge_weight = torch.ones(keep_indices.sum(), device=h_base.device)
        
        # 4.稀疏性损失（L1 正则，鼓励剪枝）
        sparsity_loss = attention_probs.mean()
        
        # 5.🔥 Entropy Loss（鼓励二元决策）
        entropy_loss = self.compute_entropy_loss(attention_probs)
        
        # 总损失
        total_loss = sparsity_loss + self.entropy_weight * entropy_loss
        
        return pruned_edge_index, pruned_edge_weight, attention_probs, total_loss
    
    def get_edge_importance(self, h_base, spatial_edge_index):
        """获取边的重要性分数（用于可视化）"""
        with torch.no_grad():
            return self.compute_attention_scores(h_base, spatial_edge_index)
