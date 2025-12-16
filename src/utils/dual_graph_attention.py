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
        # self.attention_net = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, 2),  # 输出2个权重（空间图 vs 表达图）
        # )
        
        # ===== 方案2：多头交叉注意力（更强但复杂）=====
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
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
    
class DualGraphCrossAttentionFusion(nn.Module):
    """
    双向交互注意力融合：让空间图和表达图特征相互增强
    
    核心思想：
    1. 空间特征作为Query，表达特征作为Key/Value → 空间增强表达信息
    2. 表达特征作为Query，空间特征作为Key/Value → 表达增强空间信息
    3. 融合两个方向的输出
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        
        # ===== 方向1：空间 → 表达 =====
        self.spatial_to_expr_Q = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_to_expr_K = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_to_expr_V = nn.Linear(hidden_dim, hidden_dim)
        
        # ===== 方向2：表达 → 空间 =====
        self. expr_to_spatial_Q = nn.Linear(hidden_dim, hidden_dim)
        self. expr_to_spatial_K = nn.Linear(hidden_dim, hidden_dim)
        self. expr_to_spatial_V = nn.Linear(hidden_dim, hidden_dim)
        
        # ===== 输出投影 =====
        self. W_o = nn.Linear(hidden_dim * 2, hidden_dim)  # 融合两个方向
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn. LayerNorm(hidden_dim)
        
    def _multihead_attention(self, Q, K, V, mask=None):
        """
        标准多头注意力计算
        Q, K, V: [N, hidden_dim]
        返回: [N, hidden_dim]
        """
        N = Q. size(0)
        
        # 分头：[N, hidden_dim] → [N, num_heads, head_dim]
        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)
        
        # 计算注意力分数：[N, num_heads, head_dim] × [N, num_heads, head_dim] → [N, num_heads]
        scores = torch.einsum('nhd,nhd->nh', Q, K) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)  # [N, num_heads]
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和：[N, num_heads] × [N, num_heads, head_dim] → [N, num_heads, head_dim]
        out = torch.einsum('nh,nhd->nhd', attn_weights, V)
        
        # 合并多头：[N, num_heads, head_dim] → [N, hidden_dim]
        out = out.reshape(N, self.hidden_dim)
        
        return out, attn_weights
        
    def forward(self, h_spatial: torch.Tensor, h_expr: torch.Tensor) -> torch.Tensor:
        """
        双向交互注意力融合
        
        Args:
            h_spatial: [N, hidden_dim] 空间图特征
            h_expr: [N, hidden_dim] 表达图特征
        Returns:
            h_fused: [N, hidden_dim] 融合后的特征
        """
        N = h_spatial.size(0)
        
        # ===== 方向1：空间特征查询表达特征 =====
        # Query from spatial, Key/Value from expression
        Q1 = self.spatial_to_expr_Q(h_spatial)  # [N, hidden_dim]
        K1 = self.spatial_to_expr_K(h_expr)
        V1 = self.spatial_to_expr_V(h_expr)
        
        spatial_enhanced, attn_s2e = self._multihead_attention(Q1, K1, V1)  # [N, hidden_dim]
        
        # ===== 方向2：表达特征查询空间特征 =====
        # Query from expression, Key/Value from spatial
        Q2 = self.expr_to_spatial_Q(h_expr)
        K2 = self.expr_to_spatial_K(h_spatial)
        V2 = self.expr_to_spatial_V(h_spatial)
        
        expr_enhanced, attn_e2s = self._multihead_attention(Q2, K2, V2)  # [N, hidden_dim]
        
        # ===== 融合两个方向 =====
        h_concat = torch.cat([spatial_enhanced, expr_enhanced], dim=-1)  # [N, hidden_dim*2]
        h_fused = self.W_o(h_concat)  # [N, hidden_dim]
        
        # ===== 残差连接 + LayerNorm =====
        h_fused = self.layer_norm(h_fused + h_spatial + h_expr)
        h_fused = self.dropout(h_fused)
        
        return h_fused


class DualGraphBidirectionalCrossAttention(nn.Module):
    """
    更强版本：使用完整的双向交叉注意力（类似Transformer Encoder-Decoder）
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self. num_heads = num_heads
        self.num_layers = num_layers
        
        # 堆叠多层交互注意力
        self. layers = nn.ModuleList([
            DualGraphCrossAttentionFusion(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn. GELU(),
            nn. Dropout(dropout),
            nn. Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
        
    def forward(self, h_spatial: torch.Tensor, h_expr: torch.Tensor) -> torch.Tensor:
        """
        多层双向交互注意力
        """
        h = h_spatial  # 初始化
        
        # 逐层交互
        for layer in self.layers:
            h = layer(h_spatial, h_expr)
            h_spatial = h  # 更新空间特征（用于下一层）
        
        # FFN
        h_ffn = self.ffn(h)
        h = self.layer_norm_ffn(h + h_ffn)
        
        return h