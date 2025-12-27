"""原型对比学习模块"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePrototypes(nn.Module):
    """可学习的域原型"""
    
    def __init__(
        self,
        n_prototypes: int,
        prototype_dim: int,
        temperature: float = 0.3,  # 🔥 提高温度：0.1 -> 0.3
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
        print(f"  - EMA动量: {momentum}")
        
        # 原型参数（可学习）
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prototype_dim))
        nn.init.orthogonal_(self.prototypes)
        
        # 原型的EMA缓存
        self.register_buffer(
            'prototypes_ema',
            torch.zeros(n_prototypes, prototype_dim)
        )
        
        # 🔥 新增：标记是否已初始化
        self.register_buffer('initialized', torch.tensor(False))
    
    @torch.no_grad()
    def initialize_with_kmeans(self, z:   torch.Tensor):
        """
        🔥 使用K-Means初始化原型
        
        Args:
            z: [N, prototype_dim] 节点表示（来自warmup期的z_latent）
        """
        from sklearn.cluster import KMeans
        
        print(f"🎯 使用K-Means初始化原型...")
        
        # 转到CPU
        z_np = z.detach().cpu().numpy()
        
        # K-Means聚类
        kmeans = KMeans(
            n_clusters=self.n_prototypes,
            n_init=20,  # 多次初始化，选最好的
            max_iter=300,
            random_state=42
        )
        kmeans.fit(z_np)
        
        # 获取聚类中心
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        # 归一化
        centers = F.normalize(centers, p=2, dim=-1)
        
        # 赋值给prototypes
        self.prototypes.data = centers.to(self.prototypes.device)
        self.prototypes_ema.data = centers.to(self.prototypes.device)
        
        # 标记为已初始化
        self.initialized.data = torch.tensor(True)
        
        print(f"  ✅ 原型初始化完成")
        print(f"  - 聚类惯性: {kmeans.inertia_:.4f}")
        print(f"  - 各簇样本数: {np.bincount(kmeans.labels_)}")
    
    @torch.no_grad()
    def normalize_prototypes(self):
        """L2归一化原型"""
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=-1)
    
    def compute_assignment(self, z: torch.Tensor):
        """计算spot到原型的分配"""
        # 归一化
        z_norm = F.normalize(z, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # 计算相似度
        logits = torch.mm(z_norm, prototypes_norm.t()) / self.temperature  # [N, K]
        
        # 软分配
        Q = F.softmax(logits, dim=-1)
        
        # 硬分配
        labels = logits.argmax(dim=-1)
        
        return Q, labels
    
    @torch.no_grad()
    def update_prototypes(self, z: torch.Tensor, Q: torch.Tensor):
        """使用EMA更新原型"""
        # 归一化
        z_norm = F.normalize(z, p=2, dim=-1)
        
        # 加权平均
        new_prototypes = torch.mm(Q.t(), z_norm)  # [K, N] × [N, D] = [K, D]
        weights = Q.sum(dim=0, keepdim=True).t() + 1e-8  # [K, 1]
        new_prototypes = new_prototypes / weights
        
        # EMA更新
        if self.prototypes_ema.sum().abs() < 1e-6:
            # 第一次更新
            self.prototypes_ema.data = new_prototypes
        else:
            self.prototypes_ema.data = \
                self.momentum * self.prototypes_ema.data + \
                (1 - self.momentum) * new_prototypes
        
        # 更新原型参数
        self.prototypes.data = F.normalize(self.prototypes_ema.data, p=2, dim=-1)
    
    def forward(self, z:  torch.Tensor, update:   bool = True):
        """前向传播"""
        Q, labels = self.compute_assignment(z)
        
        # 🔥 只有在已初始化后才更新
        if update and self.training and self.initialized:
            self.update_prototypes(z, Q)
        
        return Q, labels


class PrototypeLoss(nn.Module):
    """
    原型对比损失 - 修改版
    
    🔥 关键变化：
    1.完全移除域内紧致性损失
    2.改进原型分散性损失（只防止完全塌缩）
    3.主要依靠 spot-to-prototype 对比
    """
    
    def __init__(self, temperature: float = 0.3):  # 🔥 提高温度
        super().__init__()
        self.temperature = temperature
    
    def spot_to_prototype_loss(self, z:  torch.Tensor, prototypes: torch.Tensor, Q: torch.Tensor):
        """
        Spot-to-Prototype对比损失
        
        目标：每个spot接近其分配的原型（但不要求紧致）
        """
        # 归一化
        z_norm = F.normalize(z, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # 计算logits
        logits = torch.mm(z_norm, prototypes_norm.t()) / self.temperature  # [N, K]
        
        # 交叉熵（软标签）
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(Q * log_probs).sum(dim=-1).mean()
        
        return loss
    
    def prototype_dispersion(self, prototypes: torch.Tensor):
        """
        原型分散性损失 - 改进版
        
        🔥 只防止完全塌缩，不要求均匀分散
        
        方法：使用方差作为度量
        - 如果所有原型都塌缩到一个点，方差接近0
        - 允许部分原型接近（例如Layer3和Layer4可以很接近）
        """
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # 计算原型的"扩散程度"（方差）
        # 如果所有原型塌缩，centroid周围的方差会很小
        centroid = prototypes_norm.mean(dim=0, keepdim=True)  # [1, D]
        variance = ((prototypes_norm - centroid) ** 2).sum(dim=-1).mean()
        
        # 惩罚低方差（即塌缩）
        # 目标：variance应该大于某个阈值（例如0.1）
        target_variance = 0.5  # 超参数：可以调整
        loss = F.relu(target_variance - variance)  # 只有当variance < target时才有损失
        
        return loss
    
    def forward(self, z: torch.Tensor, prototypes: torch.Tensor, 
                Q: torch.Tensor, labels: torch.Tensor):
        """
        计算总原型损失
        
        Returns:
            total_loss: 标量
            loss_dict: dict 各项损失详情
        """
        # 1.Spot-to-Prototype（主要损失）
        loss_s2p = self.spot_to_prototype_loss(z, prototypes, Q)
        
        # 2.🔥 移除域内紧致性损失
        loss_compact = torch.tensor(0.0, device=z.device)
        
        # 3.原型分散性（只防止塌缩）
        loss_disperse = self.prototype_dispersion(prototypes)
        
        # 总损失：大幅降低分散性权重
        total_loss = loss_s2p + 0.5 * loss_disperse  # 🔥 移除compact，降低disperse权重
        
        loss_dict = {
            'proto_s2p': loss_s2p.item(),
            'proto_compact': 0.0,  # 已移除
            'proto_disperse': loss_disperse.item()
        }
        
        return total_loss, loss_dict