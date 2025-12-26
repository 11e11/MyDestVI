"""自适应图精炼模块 - 性能优化版"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphRefiner(nn.Module):
    """自适应图精炼器 - 性能优化版"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.alpha = alpha
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        print(f"🔧 自适应图精炼器:")
        print(f"  - 隐藏维度:  {hidden_dim}")
        print(f"  - 注意力头数: {num_heads}")
        
        # 多头注意力
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, h, edge_index, edge_weight):
        """前向传播"""
        N = h.size(0)
        E = edge_index.size(1)
        
        # 投影
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim)
        
        src, dst = edge_index[0], edge_index[1]
        
        # 注意力计算
        Q_src = Q[src]
        K_dst = K[dst]
        QK_cat = torch.cat([Q_src, K_dst], dim=-1)
        
        attention_logits = (QK_cat * self.att).sum(dim=-1)
        attention_logits = self.leaky_relu(attention_logits)
        
        # 🔥 优化版edge_softmax
        attention_weights = self._edge_softmax_fast(attention_logits, src, N)
        attention_weights = self.dropout(attention_weights)
        
        attention_weights_mean = attention_weights.mean(dim=-1)
        
        # 门控融合
        h_src = h[src]
        h_dst = h[dst]
        h_edge = torch.cat([h_src, h_dst], dim=-1)
        
        gate_values = self.gate_net(h_edge).squeeze(-1)
        
        edge_weight_normalized = edge_weight / (edge_weight.max() + 1e-8)
        
        refined_edge_weight = gate_values * attention_weights_mean + \
                             (1 - gate_values) * edge_weight_normalized
        
        return refined_edge_weight
    
    def _edge_softmax_fast(self, logits, src_index, num_nodes):
        """
        🔥 快速版edge_softmax（使用稀疏张量技巧）
        
        Args:
            logits: [E, H] 边的logits
            src_index: [E] 源节点索引
            num_nodes: 节点总数
        
        Returns: 
            softmax_values: [E, H]
        """
        E, H = logits.shape
        device = logits.device
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 方法：使用scatter_add (PyTorch内置)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Step 1: 找到每个节点的最大logit
        max_logits = torch.full((num_nodes, H), float('-inf'), device=device)
        max_logits.scatter_reduce_(
            0, 
            src_index.unsqueeze(1).expand(-1, H), 
            logits, 
            reduce='amax',
            include_self=False
        )
        
        # Step 2: 数值稳定的exp
        logits_stable = logits - max_logits[src_index]
        exp_logits = torch.exp(logits_stable)
        
        # Step 3: 计算sum(exp)
        sum_exp = torch.zeros((num_nodes, H), device=device)
        sum_exp.scatter_add_(
            0,
            src_index.unsqueeze(1).expand(-1, H),
            exp_logits
        )
        
        # Step 4: 归一化
        softmax_values = exp_logits / (sum_exp[src_index] + 1e-8)
        
        return softmax_values


class DualGraphRefiner(nn.Module):
    """双图精炼器"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        refine_spatial: bool = True,
        refine_expr: bool = False
    ):
        super().__init__()
        
        self.refine_spatial = refine_spatial
        self.refine_expr = refine_expr
        
        print(f"🔧 双图精炼器:")
        print(f"  - 精炼空间图: {refine_spatial}")
        print(f"  - 精炼表达图: {refine_expr}")
        
        if refine_spatial: 
            self.spatial_refiner = AdaptiveGraphRefiner(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        if refine_expr:
            self.expr_refiner = AdaptiveGraphRefiner(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
    
    def forward(self, h, spatial_edge_index, spatial_edge_weight, 
                expr_edge_index, expr_edge_weight):
        """精炼双图"""
        if self.refine_spatial:
            refined_spatial_weight = self.spatial_refiner(
                h, spatial_edge_index, spatial_edge_weight
            )
        else:
            refined_spatial_weight = spatial_edge_weight
        
        if self.refine_expr:
            refined_expr_weight = self.expr_refiner(
                h, expr_edge_index, expr_edge_weight
            )
        else:
            refined_expr_weight = expr_edge_weight
        
        return refined_spatial_weight, refined_expr_weight