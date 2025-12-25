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
        temperature: float = 0.1,
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
    def initialize_with_kmeans(self, z:  torch.Tensor):
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
    
    def forward(self, z: torch.Tensor, update:  bool = True):
        """前向传播"""
        Q, labels = self.compute_assignment(z)
        
        # 🔥 只有在已初始化后才更新
        if update and self.training and self.initialized:
            self.update_prototypes(z, Q)
        
        return Q, labels


class PrototypeLoss(nn.Module):
    """
    原型对比损失
    
    包含三个组件：
    1.Spot-to-Prototype对比
    2.域内紧致性
    3.原型分散性
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def spot_to_prototype_loss(self, z: torch.Tensor, prototypes: torch.Tensor, Q: torch.Tensor):
        """
        Spot-to-Prototype对比损失
        
        目标：每个spot接近其分配的原型
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
    
    def within_domain_compactness(self, z: torch.Tensor, labels: torch.Tensor):
        """
        域内紧致性损失
        
        目标：同域spot聚集
        """
        unique_labels = labels.unique()
        
        compactness = 0.0
        count = 0
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() < 2:
                continue
            
            z_domain = z[mask]
            center = z_domain.mean(dim=0, keepdim=True)
            
            # 域内方差
            var = ((z_domain - center) ** 2).sum(dim=-1).mean()
            compactness += var
            count += 1
        
        if count > 0:
            compactness = compactness / count
        
        return compactness
    
    def prototype_dispersion(self, prototypes: torch.Tensor):
        """
        原型分散性损失
        
        目标：原型之间分散（防止塌缩）
        """
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # 原型间相似度
        sim_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
        
        # 去除对角线
        mask = torch.eye(prototypes.size(0), device=prototypes.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        
        # 惩罚高相似度
        loss = sim_matrix.abs().mean()
        
        return loss
    
    def forward(self, z: torch.Tensor, prototypes: torch.Tensor, 
                Q: torch.Tensor, labels: torch.Tensor):
        """
        计算总原型损失
        
        Returns:
            total_loss: 标量
            loss_dict: dict 各项损失详情
        """
        # 1.Spot-to-Prototype
        loss_s2p = self.spot_to_prototype_loss(z, prototypes, Q)
        
        # 2.域内紧致性
        loss_compact = self.within_domain_compactness(z, labels)
        
        # 3.原型分散性
        loss_disperse = self.prototype_dispersion(prototypes)
        
        # 总损失
        total_loss = loss_s2p + 0.5 * loss_compact + 0.1 * loss_disperse
        
        loss_dict = {
            'proto_s2p': loss_s2p.item(),
            'proto_compact': loss_compact.item() if isinstance(loss_compact, torch.Tensor) else 0.0,
            'proto_disperse': loss_disperse.item()
        }
        
        return total_loss, loss_dict