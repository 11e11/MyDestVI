"""原型对比学习模块"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePrototypes(nn.Module):
    """可学习的域原型"""
    
    def __init__(
        self,
        n_prototypes:  int,
        prototype_dim:  int,
        temperature: float = 0.15,  # 🔥 使用较低温度获得清晰边界
        momentum: float = 0.99
    ):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.prototype_dim = prototype_dim
        self.temperature = temperature
        self.momentum = momentum
        
        print(f"🎯 可学习原型:")
        print(f"  - 原型数量: {n_prototypes}")
        print(f"  - 原型维度: {prototype_dim}")
        print(f"  - 温度: {temperature}")
        print(f"  - EMA 动量: {momentum}")
        
        # 原型参数（可学习）
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prototype_dim))
        nn.init.orthogonal_(self.prototypes)
        
        # 原型的 EMA 缓存
        self.register_buffer(
            'prototypes_ema',
            torch.zeros(n_prototypes, prototype_dim)
        )
        
        # 标记是否已初始化
        self.register_buffer('initialized', torch.tensor(False))
    
    @torch.no_grad()
    def initialize_with_kmeans(self, z:  torch.Tensor):
        """
        使用 K-Means 初始化原型
        
        Args:
            z: [N, prototype_dim] 节点表示
        """
        from sklearn.cluster import KMeans
        
        print(f"🎯 使用 K-Means 初始化原型...")
        
        z_np = z.detach().cpu().numpy()
        
        kmeans = KMeans(
            n_clusters=self.n_prototypes,
            n_init=20,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(z_np)
        
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        centers = F.normalize(centers, p=2, dim=-1)
        
        self.prototypes.data = centers.to(self.prototypes.device)
        self.prototypes_ema.data = centers.to(self.prototypes.device)
        
        self.initialized.data = torch.tensor(True)
        
        print(f"  ✅ 原型初始化完成")
        print(f"  - 聚类惯性: {kmeans.inertia_:.4f}")
        print(f"  - 各簇样本数: {np.bincount(kmeans.labels_)}")
    
    @torch.no_grad()
    def normalize_prototypes(self):
        """L2 归一化原型"""
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=-1)
    
    def compute_assignment(self, z: torch.Tensor):
        """计算 spot 到原型的分配"""
        z_norm = F.normalize(z, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        logits = torch.mm(z_norm, prototypes_norm.t()) / self.temperature
        
        Q = F.softmax(logits, dim=-1)
        labels = logits.argmax(dim=-1)
        
        return Q, labels
    
    @torch.no_grad()
    def update_prototypes(self, z: torch.Tensor, Q: torch.Tensor):
        """使用 EMA 更新原型"""
        z_norm = F.normalize(z, p=2, dim=-1)
        
        new_prototypes = torch.mm(Q.t(), z_norm)
        weights = Q.sum(dim=0, keepdim=True).t() + 1e-8
        new_prototypes = new_prototypes / weights
        
        if self.prototypes_ema.sum().abs() < 1e-6:
            self.prototypes_ema.data = new_prototypes
        else:
            self.prototypes_ema.data = \
                self.momentum * self.prototypes_ema.data + \
                (1 - self.momentum) * new_prototypes
        
        self.prototypes.data = F.normalize(self.prototypes_ema.data, p=2, dim=-1)
    
    def forward(self, z: torch.Tensor, update: bool = True):
        """前向传播"""
        Q, labels = self.compute_assignment(z)
        
        if update and self.training and self.initialized:
            self.update_prototypes(z, Q)
        
        return Q, labels


class PrototypeLoss(nn.Module):
    """
    原型对比损失
    
    🔥 关键变化：
    1.移除域内紧致性损失
    2.改进原型分散性损失（只防止完全塌缩）
    3.主要依靠 spot-to-prototype 对比
    """
    
    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature
    
    def spot_to_prototype_loss(self, z: torch.Tensor, prototypes: torch.Tensor, Q: torch.Tensor):
        """
        Spot-to-Prototype 对比损失
        """
        z_norm = F.normalize(z, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        
        logits = torch.mm(z_norm, prototypes_norm.t()) / self.temperature
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(Q * log_probs).sum(dim=-1).mean()
        
        return loss
    
    def prototype_dispersion(self, prototypes: torch.Tensor):
        """
        原型分散性损失
        
        只防止完全塌缩，不要求均匀分散
        """
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        
        centroid = prototypes_norm.mean(dim=0, keepdim=True)
        variance = ((prototypes_norm - centroid) ** 2).sum(dim=-1).mean()
        
        target_variance = 0.5
        loss = F.relu(target_variance - variance)
        
        return loss
    
    def forward(self, z: torch.Tensor, prototypes: torch.Tensor, 
                Q: torch.Tensor, labels: torch.Tensor):
        """
        计算总原型损失
        
        Returns:
            total_loss: 标量
            loss_dict: dict 各项损失详情
        """
        loss_s2p = self.spot_to_prototype_loss(z, prototypes, Q)
        loss_compact = torch.tensor(0.0, device=z.device)
        loss_disperse = self.prototype_dispersion(prototypes)
        
        total_loss = loss_s2p + 0.5 * loss_disperse
        
        loss_dict = {
            'proto_s2p': loss_s2p.item(),
            'proto_compact': 0.0,
            'proto_disperse': loss_disperse.item()
        }
        
        return total_loss, loss_dict