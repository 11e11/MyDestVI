"""一致性图信息瓶颈 (Consistency-aware GIB) 模块 - 保守修复版本"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyMaskGenerator(nn.Module):
    """
    基于特征相似度生成共享图结构掩码
    
    🔥 v2.1:  保守修复版本
    - 冷启动（避免初始饱和）
    - 温和Temperature（先用1.0）
    - L2归一化
    - 支持L1和熵损失
    """
    
    def __init__(
        self,
        n_input:  int,
        hidden_dim:  int = 128,
        dropout: float = 0.1,
        init_strategy: str = 'cold',  # 🔥 默认冷启动
        temperature: float = 1.0,  # 🔥 默认温和temperature
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self. init_strategy = init_strategy
        self.temperature = temperature
        self.use_layer_norm = use_layer_norm
        
        print(f"🔥 初始化一致性GIB模块 (保守版本):")
        print(f"  - 输入维度: {n_input}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - 初始化策略: {init_strategy}")
        print(f"  - Temperature: {temperature}")
        print(f"  - 使用LayerNorm: {use_layer_norm}")
        
        # Metric Network:   将节点特征映射到度量空间
        layers = []
        layers.append(nn.Linear(n_input, hidden_dim))
        if use_layer_norm: 
            layers.append(nn. LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 最终投影层
        layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 4))
        
        self.metric_net = nn.Sequential(*layers)
        
        # 🔥 只有warm模式才做热启动初始化
        if init_strategy == 'warm':
            self._init_weights_warm()
        else:
            print(f"  使用默认初始化 (冷启动)")
    
    def _init_weights_warm(self):
        """
        🔥 修正：温和的热启动（避免过度初始化）
        """
        with torch.no_grad():
            last_linear = None
            for module in self.metric_net: 
                if isinstance(module, nn.Linear):
                    last_linear = module
            
            if last_linear is not None:
                # 🔥 减小初始化强度
                nn.init.xavier_uniform_(last_linear.weight, gain=1.0)  # 从2.0改为1.0
                if last_linear.bias is not None:
                    nn.init. constant_(last_linear.bias, 0.0)  # 从0.5改为0.0
                print(f"  应用温和热启动初始化")
    
    def compute_similarity_matrix(self, x:  torch.Tensor):
        """
        计算归一化的相似度矩阵
        
        Args:
            x: [N, n_input] 输入特征
        
        Returns:
            S: [N, N] 相似度矩阵 (余弦相似度, 范围[-1,1])
        """
        # 1. 特征映射到度量空间
        h = self.metric_net(x)  # [N, hidden_dim//4]
        
        # 🔥 2. L2归一化 (确保余弦相似度范围在[-1,1])
        h_norm = F.normalize(h, p=2, dim=-1)
        
        # 3. 计算余弦相似度
        S = torch.mm(h_norm, h_norm.t())  # [N, N]
        
        return S
    
    def forward(self, x: torch.Tensor, edge_index: torch. Tensor):
        """
        🔥 核心方法：生成边掩码
        
        Args:
            x: [N, n_input] 节点特征
            edge_index: [2, E] 边索引
        
        Returns:
            edge_mask: [E,] 边权重掩码 (0-1之间)
        """
        # 1. 计算归一化的相似度矩阵
        S = self. compute_similarity_matrix(x)  # [N, N], 范围[-1, 1]
        
        # 2. 提取边的相似度
        src, dst = edge_index[0], edge_index[1]
        edge_similarity = S[src, dst]  # [E,]
        
        # 🔥 3. Temperature缩放 + Sigmoid
        # 小temperature会放大相似度差异，大temperature会平滑差异
        scaled_similarity = edge_similarity / self.temperature
        edge_mask = torch.sigmoid(scaled_similarity)
        
        return edge_mask
    
    def compute_gib_loss(
        self, 
        edge_mask: torch.Tensor, 
        loss_type: str = 'l1',  # 🔥 默认L1损失
        sparsity_weight: float = 1.0,
        entropy_weight: float = 0.0
    ):
        """
        🔥 简化：计算GIB损失（主要使用L1）
        
        Args:
            edge_mask:  [E,] 边掩码
            loss_type:  'l1', 'entropy', 'l1+entropy'
            sparsity_weight: L1损失权重
            entropy_weight: 熵损失权重
        
        Returns:
            gib_loss: 标量
            details: dict (各项损失详情)
        """
        eps = 1e-6
        
        # 🔥 1. L1损失（鼓励稀疏，降低整体mask均值）
        loss_l1 = edge_mask.mean()
        
        # 🔥 2. 熵损失（鼓励两极分化，可选）
        if entropy_weight > 0:
            mask_clipped = torch.clamp(edge_mask, eps, 1 - eps)
            entropy = -(
                mask_clipped * torch. log(mask_clipped) + 
                (1 - mask_clipped) * torch.log(1 - mask_clipped)
            )
            loss_entropy = entropy. mean()
        else:
            loss_entropy = torch.tensor(0.0, device=edge_mask.device)
        
        # 🔥 3. 混合损失
        if loss_type == 'l1': 
            gib_loss = loss_l1
        elif loss_type == 'entropy':
            gib_loss = loss_entropy
        elif loss_type == 'l1+entropy':
            gib_loss = sparsity_weight * loss_l1 + entropy_weight * loss_entropy
        else:
            raise ValueError(f"不支持的GIB损失类型: {loss_type}")
        
        details = {
            'l1_loss': loss_l1. item(),
            'entropy_loss': loss_entropy.item() if isinstance(loss_entropy, torch. Tensor) else 0.0
        }
        
        return gib_loss, details


class AdaptiveGIBScheduler: 
    """
    自适应GIB权重调度器
    
    策略:  训练初期GIB权重小(保留大部分边),训练后期逐渐增大(稀疏化)
    """
    
    def __init__(
        self,
        initial_weight: float = 0.0,
        final_weight: float = 1.0,
        warmup_epochs: int = 50,
        total_epochs: int = 300,
        schedule_type: str = 'linear'  # 'linear', 'cosine', 'exp'
    ):
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        
        print(f"📅 GIB调度器:")
        print(f"  - 初始权重: {initial_weight}")
        print(f"  - 最终权重: {final_weight}")
        print(f"  - 预热轮数: {warmup_epochs}")
        print(f"  - 调度方式: {schedule_type}")
    
    def get_weight(self, epoch: int) -> float:
        """获取当前epoch的GIB权重"""
        if epoch < self.warmup_epochs:
            return self.initial_weight
        
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        if self.schedule_type == 'linear':
            weight = self.initial_weight + (self.final_weight - self. initial_weight) * progress
        
        elif self.schedule_type == 'cosine':
            import math
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * \
                     (1 - math.cos(progress * math.pi)) / 2
        
        elif self.schedule_type == 'exp':
            weight = self.initial_weight + (self.final_weight - self. initial_weight) * (progress ** 2)
        
        else:
            weight = self.final_weight
        
        return weight


# 🔥 可选：动态Temperature调度器
class DynamicTemperatureScheduler:
    """
    动态Temperature调度器
    
    策略: 训练初期用大temperature(平滑),后期用小temperature(高对比度)
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.2,
        warmup_epochs: int = 100,
        total_epochs: int = 300,
        schedule_type: str = 'linear'
    ):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        
        print(f"🌡️ Temperature调度器:")
        print(f"  - 初始Temperature: {initial_temperature}")
        print(f"  - 最终Temperature: {final_temperature}")
        print(f"  - 预热轮数: {warmup_epochs}")
        print(f"  - 调度方式: {schedule_type}")
    
    def get_temperature(self, epoch: int) -> float:
        """获取当前epoch的temperature"""
        if epoch < self. warmup_epochs:
            return self.initial_temperature
        
        progress = (epoch - self. warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        if self.schedule_type == 'linear': 
            temp = self.initial_temperature + (self.final_temperature - self. initial_temperature) * progress
        
        elif self.schedule_type == 'cosine':
            import math
            temp = self. initial_temperature + (self.final_temperature - self.initial_temperature) * \
                   (1 - math.cos(progress * math.pi)) / 2
        
        elif self.schedule_type == 'exp':
            temp = self.initial_temperature + (self.final_temperature - self.initial_temperature) * (progress ** 2)
        
        else:
            temp = self. final_temperature
        
        return temp
    
    def update_model_temperature(self, model, epoch: int):
        """更新模型中的temperature"""
        new_temp = self.get_temperature(epoch)
        
        # 更新encoder中的gib_module的temperature
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'gib_module'):
            model.encoder.gib_module.temperature = new_temp
        
        return new_temp