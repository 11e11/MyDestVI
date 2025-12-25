"""空间域识别模型 - 三图架构 + 双层对比学习"""
from scvi.module.base._base_module import BaseModuleClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv, GCNConv
from torch_geometric.utils import to_undirected, coalesce

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch.distributions import Poisson, Normal
from scvi import REGISTRY_KEYS

from .graph_augmentation import GraphAugmentor, GraphContrastiveLoss


class TripleGraphEncoder(nn.Module):
    """
    三图编码器: 
    - 空间图（原始）
    - 空间图（增强后）→ 用于图对比学习
    - 表达图（独立）→ 用于局部-全局对比学习
    """
    
    def __init__(
        self,
        n_input:  int,
        hidden_dim:  int = 256,
        latent_dim:  int = 64,
        gnn_type:  str = 'gine',
        n_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = 'independent',  # 空间+表达的融合方式
        # 🔥 图增强参数
        use_graph_augmentation: bool = True,
        k_augment:  int = 20,
        augment_mode: str = 'topk',
        augment_method: str = 'add',  # 'add' or 'replace'
        augment_weight: float = 0.5,  # 增强图的权重（加法时）
        gnn_sharing_strategy:  str = 'partial',  # 空间图和增强图的GNN共享策略
    ):
        super().__init__()
        
        self.use_graph_augmentation = use_graph_augmentation
        self.fusion_type = fusion_type
        self.augment_method = augment_method
        self.augment_weight = augment_weight
        self.gnn_type = gnn_type
        self.gnn_sharing_strategy = gnn_sharing_strategy
        
        print(f"🏗️ 三图编码器配置:")
        print(f"  - 🔥 使用图增强: {use_graph_augmentation}")
        if use_graph_augmentation: 
            print(f"  - 🔥 增强方式: {augment_method}")
            print(f"  - 🔥 增强权重: {augment_weight}")
        print(f"  - 融合方式: {fusion_type}")
        print(f"  - GNN类型: {gnn_type}")
        print(f"  - GNN共享策略: {gnn_sharing_strategy}")
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 🔥 图增强器（生成增强边）
        if use_graph_augmentation: 
            self.graph_augmentor = GraphAugmentor(
                n_input=n_input,
                hidden_dim=128,
                k_augment=k_augment,
                augment_mode=augment_mode,
                dropout=dropout
            )
        
        if gnn_sharing_strategy == 'none':
            # 🔥 方案A:  完全分开（当前实现）
            self.spatial_conv1, self.spatial_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
            self.spatial_conv2, self.spatial_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
            
            self.augmented_spatial_conv1, self.augmented_spatial_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
            self.augmented_spatial_conv2, self.augmented_spatial_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        elif gnn_sharing_strategy == 'full':
            # 🔥 方案B: 完全共享
            self.shared_conv1, self.shared_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
            self.shared_conv2, self.shared_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        elif gnn_sharing_strategy == 'partial':
            # 🔥 方案C: 部分共享（推荐）
            # 第1层共享
            self.shared_conv1, self.shared_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
            
            # 第2层分开
            self.spatial_conv2, self.spatial_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
            self.augmented_spatial_conv2, self.augmented_spatial_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        # 表达图GNN（始终独立）
        self.expr_conv1, self.expr_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        self.expr_conv2, self.expr_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        # 🔥 门控融合（空间 + 表达）
        if fusion_type == 'independent':
            self.gate_spatial = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.Sigmoid()
            )
            self.gate_expr = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # 投影到latent
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
    
    def _make_gnn_layer(self, in_dim, out_dim, n_heads):
        """创建GNN层"""
        if self.gnn_type == 'gine':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(out_dim, out_dim)
            )
            conv = GINEConv(mlp, train_eps=True, edge_dim=1)
        elif self.gnn_type == 'gat':
            conv = GATConv(
                in_channels=in_dim,
                out_channels=out_dim // n_heads,
                heads=n_heads,
                dropout=0.1,
                edge_dim=1,
                concat=True
            )
        elif self.gnn_type == 'gcn':
            conv = GCNConv(in_dim, out_dim)
        else:
            raise ValueError(f"不支持的GNN:  {self.gnn_type}")
        
        ln = nn.LayerNorm(out_dim)
        return conv, ln
    
    def _apply_gnn(self, conv, ln, h, edge_index, edge_weight):
        """应用GNN层"""
        if self.gnn_type in ['gine', 'gat']:
            h_new = conv(h, edge_index, edge_attr=edge_weight.unsqueeze(-1))
        elif self.gnn_type == 'gcn':
            h_new = conv(h, edge_index, edge_weight=edge_weight)
        
        h_new = ln(h_new)
        return h_new

    def _apply_gnn_with_residual(self, conv, ln, h, edge_index, edge_weight):
        """应用GNN层 + 残差连接"""
        h_new = self._apply_gnn(conv, ln, h, edge_index, edge_weight)
        return F.relu(h_new) + h
    
    def _augment_spatial_graph(self, spatial_edge_index, spatial_edge_weight, 
                               augment_edge_index, augment_edge_weight):
        """
        🔥 增强空间图
        
        Args:
            spatial_edge_index: [2, E_s] 原始空间图
            spatial_edge_weight: [E_s,]
            augment_edge_index: [2, E_a] 增强边
            augment_edge_weight: [E_a,]
        
        Returns:
            augmented_edge_index: [2, E_new]
            augmented_edge_weight: [E_new,]
        """
        if self.augment_method == 'add':
            # 🔥 加法：合并两个图
            # 调整增强边的权重
            augment_edge_weight_scaled = augment_edge_weight * self.augment_weight
            
            # 合并边
            combined_edge_index = torch.cat([spatial_edge_index, augment_edge_index], dim=1)
            combined_edge_weight = torch.cat([spatial_edge_weight, augment_edge_weight_scaled], dim=0)
            
            # 🔥 关键：去重并合并权重（如果有重复边，权重相加）
            augmented_edge_index, augmented_edge_weight = coalesce(
                combined_edge_index,
                combined_edge_weight,
                reduce='add'  # 重复边的权重相加
            )
        
        elif self.augment_method == 'replace':
            # 替换：直接用增强图
            augmented_edge_index = augment_edge_index
            augmented_edge_weight = augment_edge_weight
        
        else:
            raise ValueError(f"不支持的增强方式: {self.augment_method}")
        
        return augmented_edge_index, augmented_edge_weight
    
    def forward(
        self,
        x: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_weight: torch.Tensor,
        expr_edge_index: torch.Tensor,
        expr_edge_weight: torch.Tensor
    ):
        """
        前向传播
        
        Returns:
            z_latent: [N, latent_dim] 最终latent表示
            z_spatial: [N, hidden_dim] 原始空间图的表示（用于图对比）
            z_augmented_spatial: [N, hidden_dim] 增强空间图的表示（用于图对比）
            z_expr: [N, hidden_dim] 表达图的表示（用于局部-全局对比）
            info: dict
        """
        info = {}
        h = self.input_proj(x)
        
        # 生成增强图
        if self.use_graph_augmentation:
            augment_edge_index, augment_edge_weight, _ = self.graph_augmentor(x)
            augmented_spatial_edge_index, augmented_spatial_edge_weight = \
                self._augment_spatial_graph(
                    spatial_edge_index, spatial_edge_weight,
                    augment_edge_index, augment_edge_weight
                )
        
        # ========================================
        # 🔥 根据共享策略处理空间图分支
        # ========================================
        if self.gnn_sharing_strategy == 'none': 
            # 完全分开
            h_spatial = h
            h_spatial = self._apply_gnn_with_residual(
                self.spatial_conv1, self.spatial_ln1,
                h_spatial, spatial_edge_index, spatial_edge_weight
            )
            h_spatial = self._apply_gnn_with_residual(
                self.spatial_conv2, self.spatial_ln2,
                h_spatial, spatial_edge_index, spatial_edge_weight
            )
            z_spatial = h_spatial
            
            if self.use_graph_augmentation:
                h_aug = h
                h_aug = self._apply_gnn_with_residual(
                    self.augmented_spatial_conv1, self.augmented_spatial_ln1,
                    h_aug, augmented_spatial_edge_index, augmented_spatial_edge_weight
                )
                h_aug = self._apply_gnn_with_residual(
                    self.augmented_spatial_conv2, self.augmented_spatial_ln2,
                    h_aug, augmented_spatial_edge_index, augmented_spatial_edge_weight
                )
                z_augmented_spatial = h_aug
        
        elif self.gnn_sharing_strategy == 'full': 
            # 完全共享
            h_spatial = h
            h_spatial = self._apply_gnn_with_residual(
                self.shared_conv1, self.shared_ln1,
                h_spatial, spatial_edge_index, spatial_edge_weight
            )
            h_spatial = self._apply_gnn_with_residual(
                self.shared_conv2, self.shared_ln2,
                h_spatial, spatial_edge_index, spatial_edge_weight
            )
            z_spatial = h_spatial
            
            if self.use_graph_augmentation:
                h_aug = h
                h_aug = self._apply_gnn_with_residual(
                    self.shared_conv1, self.shared_ln1,  # 🔥 共享
                    h_aug, augmented_spatial_edge_index, augmented_spatial_edge_weight
                )
                h_aug = self._apply_gnn_with_residual(
                    self.shared_conv2, self.shared_ln2,  # 🔥 共享
                    h_aug, augmented_spatial_edge_index, augmented_spatial_edge_weight
                )
                z_augmented_spatial = h_aug
        
        elif self.gnn_sharing_strategy == 'partial':
            # 部分共享（第1层共享，第2层分开）
            h_spatial = h
            h_spatial = self._apply_gnn_with_residual(
                self.shared_conv1, self.shared_ln1,  # 🔥 第1层共享
                h_spatial, spatial_edge_index, spatial_edge_weight
            )
            h_spatial = self._apply_gnn_with_residual(
                self.spatial_conv2, self.spatial_ln2,  # 🔥 第2层分开
                h_spatial, spatial_edge_index, spatial_edge_weight
            )
            z_spatial = h_spatial
            
            if self.use_graph_augmentation:
                h_aug = h
                h_aug = self._apply_gnn_with_residual(
                    self.shared_conv1, self.shared_ln1,  # 🔥 第1层共享
                    h_aug, augmented_spatial_edge_index, augmented_spatial_edge_weight
                )
                h_aug = self._apply_gnn_with_residual(
                    self.augmented_spatial_conv2, self.augmented_spatial_ln2,  # 🔥 第2层分开
                    h_aug, augmented_spatial_edge_index, augmented_spatial_edge_weight
                )
                z_augmented_spatial = h_aug
        
        # 计算鲁棒的空间表示
        if self.use_graph_augmentation:
            z_spatial_robust = (z_spatial + z_augmented_spatial) / 2
        else:
            z_augmented_spatial = None
            z_spatial_robust = z_spatial
        
        # ========================================
        # 表达图分支（始终独立）
        # ========================================
        h_expr = h
        h_expr = self._apply_gnn_with_residual(
            self.expr_conv1, self.expr_ln1,
            h_expr, expr_edge_index, expr_edge_weight
        )
        h_expr = self._apply_gnn_with_residual(
            self.expr_conv2, self.expr_ln2,
            h_expr, expr_edge_index, expr_edge_weight
        )
        z_expr = h_expr
        
        # ========================================
        # 🔥 步骤5: 融合空间表示和表达表示
        # ========================================
        if self.fusion_type == 'independent':
            gate_s = self.gate_spatial(z_spatial_robust)
            gate_e = self.gate_expr(z_expr)
            z_fused = gate_s * z_spatial_robust + gate_e * z_expr
            
            info['gate_spatial_mean'] = gate_s.mean().item()
            info['gate_expr_mean'] = gate_e.mean().item()
        else:
            # 简单相加
            z_fused = z_spatial_robust + z_expr
        
        # ========================================
        # 🔥 步骤6: 投影到latent
        # ========================================
        z_latent = self.to_latent(z_fused)
        
        return z_latent, z_spatial, z_augmented_spatial, z_expr, info


class UniversalDecoder(nn.Module):
    """
    通用解码器 - 优化版本
    🔥 修改：支持不同的 latent 维度和输出维度
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_output: int,  # 🔥 新增：输出维度（原始基因数，如 3000）
        hidden_dim: int = 256,
        dropout: float = 0.1,
        distribution: str = 'nb'
    ):
        super().__init__()
        
        self.distribution = distribution.lower()
        self.n_output = n_output
        
        print(f"📊 解码器配置:")
        print(f"  - 输出维度: {n_output}")  # 🔥 打印输出维度
        print(f"  - 使用分布: {self.distribution.upper()}")
        
        # 解码器骨干
        self.decoder_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 🔥 均值解码器：输出 n_output 维
        self.mean_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, n_output),
            nn.Softmax(dim=-1)
        )
        
        # Dispersion 参数（NB/ZINB）
        if self.distribution in ['nb', 'zinb']: 
            # 🔥 n_output 个基因的 dispersion
            self.px_r = nn.Parameter(torch.ones(n_output) * 4.0)
            print(f"  初始化 dispersion (θ): {torch.exp(self.px_r).mean().item():.2f}")
        
        # Dropout 参数（ZINB）
        if self.distribution == 'zinb': 
            self.dropout_decoder = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_output)
            )
        
        # Variance 参数（Gaussian）
        if self.distribution == 'gaussian': 
            self.logvar_decoder = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_output)
            )
    
    def forward(self, z: torch.Tensor, library_size: torch.Tensor):
        """
        解码 latent 表示
        
        Args:
            z: [N, latent_dim]
            library_size: [N, 1]
        
        Returns: 
            dict:  分布参数，输出维度为 [N, n_output]
        """
        h = self.decoder_backbone(z)
        
        # 均值（library-size 归一化）
        px_scale = self.mean_decoder(h)  # [N, n_output]
        px_rate = library_size * px_scale
        
        outputs = {
            'px_rate': px_rate,
            'px_scale': px_scale
        }
        
        # NB/ZINB:  dispersion
        if self.distribution in ['nb', 'zinb']:
            px_r = torch.exp(self.px_r)
            outputs['px_r'] = px_r
        
        # ZINB: dropout logits
        if self.distribution == 'zinb':
            outputs['px_dropout'] = self.dropout_decoder(h)
        
        # Gaussian: variance
        if self.distribution == 'gaussian':
            outputs['logvar'] = self.logvar_decoder(h)
            outputs['var'] = torch.exp(outputs['logvar'].clamp(min=-10, max=10))
        
        return outputs


class ContrastiveHead(nn.Module):
    """对比学习头部（保持不变）"""
    
    def __init__(self, latent_dim: int, projection_dim: int = 128):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, z: torch.Tensor):
        return F.normalize(self.projection(z), dim=-1)

class SpatialDomainModel(BaseModuleClass):
    """
    空间域识别模型 - 三图架构 + 双层对比学习
    
    架构: 
    1.三个图:  空间图、增强空间图、表达图
    2.图对比学习: 空间图 vs 增强空间图（结构鲁棒性）
    3.局部-全局对比学习: 
       - 在表达图表示上（域内一致性）
       - 在最终latent上（全局判别性）
    """
    
    def __init__(
        self,
        n_input:  int,
        n_output: int,
        hidden_dim: int = 256,
        latent_dim: int = 30,
        gnn_type:  str = 'gine',
        n_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = 'independent',
        reconstruction_loss: str = 'nb',
        # 图增强参数
        use_graph_augmentation: bool = True,
        k_augment: int = 20,
        augment_mode: str = 'topk',
        augment_method: str = 'add',
        augment_weight: float = 0.5,
        # 损失权重
        contrast_weight: float = 20.0,
        graph_contrast_weight: float = 10.0,
        local_contrast_weight: float = 2.0,
        global_contrast_weight: float = 1.0,
        # 对比学习参数
        temperature: float = 0.3,
        graph_contrast_temperature: float = 0.5,
        contrast_sample_rate: float = 0.5,
        local_margin:  float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.use_graph_augmentation = use_graph_augmentation
        self.contrast_weight = contrast_weight
        self.graph_contrast_weight = graph_contrast_weight
        self.local_contrast_weight = local_contrast_weight
        self.global_contrast_weight = global_contrast_weight
        self.temperature = temperature
        self.contrast_sample_rate = contrast_sample_rate
        self.local_margin = local_margin
        self.reconstruction_loss = reconstruction_loss
        
        print(f"\n🎯 模型配置 - 三图架构:")
        print(f"  - 🔥 使用图增强: {use_graph_augmentation}")
        print(f"  - 🔥 图对比权重: {graph_contrast_weight}")
        print(f"  - 🔥 局部-全局对比权重: {contrast_weight}")
        
        # 编码器
        self.encoder = TripleGraphEncoder(
            n_input=n_input,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout,
            fusion_type=fusion_type,
            use_graph_augmentation=use_graph_augmentation,
            k_augment=k_augment,
            augment_mode=augment_mode,
            augment_method=augment_method,
            augment_weight=augment_weight
        )
        
        # 解码器（保持不变）
        from .spatial_domain_model import UniversalDecoder
        self.decoder = UniversalDecoder(
            latent_dim=latent_dim,
            n_output=n_output,
            hidden_dim=hidden_dim,
            dropout=dropout,
            distribution=reconstruction_loss
        )
        
        # 🔥 图对比学习损失
        if use_graph_augmentation:
            self.graph_contrast_loss_fn = GraphContrastiveLoss(
                temperature=graph_contrast_temperature,
                contrast_mode='infonce'
            )
        
        # 🔥 局部-全局对比学习投影头
        from .spatial_domain_model import ContrastiveHead
        self.contrast_head_expr = ContrastiveHead(hidden_dim, projection_dim=128)  # 用于表达图表示
        self.contrast_head_latent = ContrastiveHead(latent_dim, projection_dim=128)  # 用于latent表示
        
        # Buffers
        self.register_buffer("_spatial_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_spatial_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_expr_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_expr_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_X_pca", torch.empty((0, n_input), dtype=torch.float32))
        self.register_buffer("_X_raw", torch.empty((0, n_output), dtype=torch.float32))
    
    def attach_dual_graphs(self, spatial_edge_index, spatial_edge_weight, expr_edge_index, expr_edge_weight):
        self.register_buffer("_spatial_edge_index", spatial_edge_index)
        self.register_buffer("_spatial_edge_weight", spatial_edge_weight)
        self.register_buffer("_expr_edge_index", expr_edge_index)
        self.register_buffer("_expr_edge_weight", expr_edge_weight)
        return self
    
    def attach_full_X(self, X_pca, X_raw):
        self.register_buffer("_X_pca", X_pca)
        self.register_buffer("_X_raw", X_raw)
        return self
    
    def forward_all(self):
        """全图前向传播"""
        if self._X_pca.numel() == 0:
            raise RuntimeError("需要先调用 attach_full_X")
        
        X_pca = self._X_pca
        library_size = self._X_raw.sum(dim=1, keepdim=True)
        
        # 编码
        z_latent, z_spatial, z_augmented_spatial, z_expr, encoder_info = self.encoder(
            X_pca,
            self._spatial_edge_index,
            self._spatial_edge_weight,
            self._expr_edge_index,
            self._expr_edge_weight
        )
        
        # 解码
        decoder_outputs = self.decoder(z_latent, library_size)
        
        return z_latent, z_spatial, z_augmented_spatial, z_expr, decoder_outputs, encoder_info
    
    def compute_graph_contrast_loss(self, z_spatial, z_augmented_spatial):
        """
        🔥 图对比学习损失
        
        目标:  z_spatial ≈ z_augmented_spatial
        作用: 让模型学到对图结构变化鲁棒的表示
        """
        if not self.use_graph_augmentation or z_augmented_spatial is None:
            return torch.tensor(0.0, device=z_spatial.device)
        
        loss = self.graph_contrast_loss_fn(z_spatial, z_augmented_spatial)
        return loss
    
    def contrastive_loss_on_representation(self, z, contrast_samples, projection_head, use_margin=True):
        """
        🔥 在特定表示上计算局部-全局对比损失
        
        Args: 
            z: [N, D] 表示（可以是z_expr或z_latent）
            contrast_samples: 对比样本字典
            projection_head: 投影头
            use_margin: 是否使用margin loss
        
        Returns:
            total_loss, local_loss, global_loss
        """
        device = z.device
        n_spots = z.size(0)
        
        n_sample = int(n_spots * self.contrast_sample_rate)
        sampled_indices = torch.randperm(n_spots, device=device)[:n_sample].cpu().numpy()
        
        z_proj = projection_head(z)
        
        local_loss = 0.0
        global_loss = 0.0
        n_valid_local = 0
        n_valid_global = 0
        
        for i in sampled_indices:
            anchor = z_proj[i: i+1]
            
            # 局部对比
            local_pos_idx = contrast_samples['local_pos'][i]
            local_neg_idx = contrast_samples['local_neg'][i]
            
            if len(local_pos_idx) > 0:
                local_pos = z_proj[local_pos_idx]
                pos_sim = torch.mm(anchor, local_pos.t())
                
                if use_margin:
                    pos_loss = F.softplus(self.local_margin - pos_sim).mean()
                else:
                    pos_loss = -pos_sim.mean()
                
                if len(local_neg_idx) > 0:
                    local_neg = z_proj[local_neg_idx]
                    neg_sim = torch.mm(anchor, local_neg.t())
                    
                    if use_margin:
                        neg_loss = F.softplus(neg_sim - self.local_margin).mean()
                    else:
                        neg_loss = neg_sim.mean()
                    
                    loss = pos_loss + neg_loss
                else:
                    loss = pos_loss
                
                local_loss += loss
                n_valid_local += 1
            
            # 全局对比
            global_pos_idx = contrast_samples['global_pos'][i]
            global_neg_idx = contrast_samples['global_neg'][i]
            
            if len(global_pos_idx) > 0 and len(global_neg_idx) > 0:
                global_pos = z_proj[global_pos_idx]
                global_neg = z_proj[global_neg_idx]
                
                pos_sim = torch.mm(anchor, global_pos.t()) / self.temperature
                neg_sim = torch.mm(anchor, global_neg.t()) / self.temperature
                
                logits = torch.cat([pos_sim.mean(dim=1, keepdim=True), neg_sim], dim=1)
                loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
                global_loss += loss
                n_valid_global += 1
        
        if n_valid_local > 0:
            local_loss = local_loss / n_valid_local
        else:
            local_loss = torch.tensor(0.0, device=device)
        
        if n_valid_global > 0:
            global_loss = global_loss / n_valid_global
        else:
            global_loss = torch.tensor(0.0, device=device)
        
        total_contrast_loss = (
            self.local_contrast_weight * local_loss +
            self.global_contrast_weight * global_loss
        )
        
        return total_contrast_loss, local_loss, global_loss
    
    def compute_reconstruction_loss(self, decoder_outputs, x_raw):
        """重构损失（保持不变）"""
        from .spatial_domain_model import NegativeBinomial, ZeroInflatedNegativeBinomial
        from torch.distributions import Poisson, Normal
        
        px_rate = decoder_outputs['px_rate']
        eps = 1e-8
        px_rate = torch.clamp(px_rate, min=eps)
        
        if self.reconstruction_loss == 'zinb':
            px_r = decoder_outputs['px_r']
            px_dropout = decoder_outputs['px_dropout']
            dist = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        elif self.reconstruction_loss == 'nb': 
            px_r = decoder_outputs['px_r']
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        elif self.reconstruction_loss == 'poisson':
            dist = Poisson(rate=px_rate)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        elif self.reconstruction_loss == 'gaussian':
            var = decoder_outputs['var']
            recon_loss = 0.5 * (
                torch.log(2 * torch.pi * var + eps) +
                (x_raw - px_rate) ** 2 / (var + eps)
            ).sum(dim=-1)
        else:
            raise ValueError(f"不支持的重构损失:  {self.reconstruction_loss}")
        
        return recon_loss
    
    @torch.no_grad()
    def get_latent_representation(self):
        """获取latent表示"""
        self.eval()
        z_latent, _, _, _, _, _ = self.forward_all()
        return z_latent.cpu().numpy()