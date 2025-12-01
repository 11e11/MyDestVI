"""
自适应残差门控模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResidualGate(nn.Module):
    """
    学习每个节点应该保留多少自身信息 vs 邻居信息
    
    核心思想：
    - 自身信号强 → 高 gate 值 → 保留自身特征
    - 自身信号弱 → 低 gate 值 → 依赖邻居特征
    """
    def __init__(self, hidden_dim: int, use_confidence: bool = True):
        super().__init__()
        self.use_confidence = use_confidence
        
        # 方案1：简单门控（基于特征本身）
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),  # 输入：[h_self, h_gnn]
            nn.ReLU(),
            nn. Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的门控值
        )
        
        # 方案2：置信度增强（考虑节点自身的"信号强度"）
        if use_confidence:
            self.confidence_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn. Sigmoid()
            )
    
    def compute_confidence(self, h: torch.Tensor) -> torch. Tensor:
        """
        计算节点的"置信度"（信号强度）
        高置信度 → 自身信息可靠 → 不需要邻居
        
        Args:
            h: [N, hidden_dim] 节点特征
        Returns:
            confidence: [N, 1] 置信度分数
        """
        if self. use_confidence:
            # 基于特征学习置信度
            confidence = self.confidence_net(h)
            return confidence
        else:
            # 简化版：使用特征范数作为置信度
            return torch.norm(h, p=2, dim=-1, keepdim=True) / (h.size(-1) ** 0.5)
    
    def forward(
        self, 
        h_self: torch.Tensor,   # 自身特征（GNN前）
        h_gnn: torch.Tensor,    # GNN聚合后的特征
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Args:
            h_self: [N, hidden_dim] 原始节点特征（线性变换后但未经GNN）
            h_gnn: [N, hidden_dim] GNN聚合后的特征
            return_gate: 是否返回门控值（用于可视化）
        Returns:
            h_out: [N, hidden_dim] 融合后的特征
            (可选) gate: [N, 1] 门控值
        """
        N = h_self.size(0)
        
        # 🔥 方法1：基于特征差异的门控（简单有效）
        gate_input = torch.cat([h_self, h_gnn], dim=-1)  # [N, 2*hidden_dim]
        gate = self.gate_net(gate_input)  # [N, 1]，范围 [0, 1]
        
        # 🔥 方法2：置信度加权（可选）
        if self.use_confidence:
            confidence = self. compute_confidence(h_self)  # [N, 1]
            # 高置信度 → 增加 gate（更依赖自身）
            gate = gate * 0.7 + confidence * 0.3  # 加权组合
            gate = torch.clamp(gate, 0.1, 0.9)  # 防止极端值
        
        # 残差连接
        h_out = gate * h_self + (1 - gate) * h_gnn
        
        if return_gate:
            return h_out, gate
        return h_out


class AdaptiveResidualGateV2(nn.Module):
    """
    增强版：考虑邻居的一致性
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # 门控网络：输入包含自身特征、GNN特征、邻居一致性
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 for consistency score
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_neighbor_consistency(
        self, 
        h_self: torch.Tensor, 
        h_neighbors: torch.Tensor
    ) -> torch.Tensor:
        """
        计算邻居特征的一致性
        高一致性 → 邻居可靠 → 可以依赖GNN
        
        Args:
            h_self: [N, hidden_dim]
            h_neighbors: [N, K, hidden_dim] K个邻居的特征
        Returns:
            consistency: [N, 1]
        """
        # 计算邻居间的标准差（低标准差 = 高一致性）
        neighbor_std = torch.std(h_neighbors, dim=1). mean(dim=-1, keepdim=True)  # [N, 1]
        consistency = torch.exp(-neighbor_std)  # 转换为 [0, 1]
        return consistency
    
    def forward(
        self, 
        h_self: torch.Tensor, 
        h_gnn: torch.Tensor,
        neighbor_consistency: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            h_self: [N, hidden_dim]
            h_gnn: [N, hidden_dim]
            neighbor_consistency: [N, 1] 邻居一致性分数（可选）
        """
        if neighbor_consistency is None:
            # 简化：用特征差异估计一致性
            diff = torch.norm(h_self - h_gnn, p=2, dim=-1, keepdim=True)
            neighbor_consistency = torch.exp(-diff)
        
        gate_input = torch.cat([h_self, h_gnn, neighbor_consistency], dim=-1)
        gate = self.gate_net(gate_input)
        
        h_out = gate * h_self + (1 - gate) * h_gnn
        return h_out