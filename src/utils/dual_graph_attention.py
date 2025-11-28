"""
双图注意力融合模块
"""
import torch
import torch.nn as nn
import torch. nn.functional as F


class DualGraphAttentionFusion(nn.Module):
    """
    使用多头注意力融合空间图和表达图的特征
    
    核心思想：
    - 对于每个节点，自适应学习空间特征和表达特征的权重
    - 使用 cross-attention 让两种特征相互增强
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        
        # ===== 方案1：简单加权注意力（推荐开始用这个）=====
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 输出2个权重（空间图 vs 表达图）
        )
        
        # ===== 方案2：多头交叉注意力（更强但复杂）=====
        # self.W_q = nn.Linear(hidden_dim, hidden_dim)
        # self.W_k = nn.Linear(hidden_dim, hidden_dim)
        # self.W_v = nn.Linear(hidden_dim, hidden_dim)
        # self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h_spatial: torch.Tensor, h_expr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_spatial: [N, hidden_dim] 空间图特征
            h_expr: [N, hidden_dim] 表达图特征
        Returns:
            h_fused: [N, hidden_dim] 融合后的特征
        """
        N = h_spatial.size(0)
        
        # ===== 简单加权注意力（推荐） =====
        # 1. 拼接两种特征
        h_concat = torch.cat([h_spatial, h_expr], dim=-1)  # [N, 2*hidden_dim]
        
        # 2. 学习注意力权重
        attn_logits = self.attention_net(h_concat)  # [N, 2]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [N, 2]，归一化到 [0,1]
        
        # 3. 加权求和
        h_fused = attn_weights[:, 0:1] * h_spatial + attn_weights[:, 1:2] * h_expr  # [N, hidden_dim]
        
        # 4.  残差连接 + LayerNorm
        h_fused = self.layer_norm(h_fused + h_spatial + h_expr)  # 三者融合
        h_fused = self.dropout(h_fused)
        
        return h_fused
    
    
    # ===== 可选：多头交叉注意力版本（更强但训练慢）=====
    def forward_cross_attention(self, h_spatial: torch.Tensor, h_expr: torch.Tensor) -> torch.Tensor:
        """
        使用 cross-attention 让空间特征和表达特征相互增强
        """
        N = h_spatial.size(0)
        
        # Query: 空间特征，Key/Value: 表达特征
        Q = self.W_q(h_spatial). view(N, self.num_heads, self.head_dim)  # [N, heads, head_dim]
        K = self.W_k(h_expr).view(N, self.num_heads, self.head_dim)
        V = self.W_v(h_expr).view(N, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        scores = torch.einsum('nhd,nhd->nh', Q, K) / (self.head_dim ** 0.5)  # [N, heads]
        attn = F.softmax(scores, dim=-1)  # [N, heads]
        
        # 加权 Value
        out = torch.einsum('nh,nhd->nhd', attn, V)  # [N, heads, head_dim]
        out = out.reshape(N, self.hidden_dim)  # [N, hidden_dim]
        
        # 输出投影
        out = self.W_o(out)
        
        # 残差连接
        h_fused = self.layer_norm(out + h_spatial)
        h_fused = self.dropout(h_fused)
        
        return h_fused