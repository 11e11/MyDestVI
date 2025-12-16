"""空间域识别模型 - 门控融合版本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric. nn import GINEConv, GATConv, GCNConv
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch.distributions import Poisson, Normal
from scvi import REGISTRY_KEYS


class GatedFusion(nn.Module):
    """
    门控融合模块
    
    公式：
        gate = σ(W_g [h_spatial, h_expr])
        h_fused = gate ⊙ h_spatial + (1-gate) ⊙ h_expr
        
    或者更复杂的版本（推荐）：
        gate_s = σ(W_s h_spatial)
        gate_e = σ(W_e h_expr)
        h_fused = gate_s ⊙ h_spatial + gate_e ⊙ h_expr
    """
    
    def __init__(self, hidden_dim: int, fusion_type: str = 'simple'):
        """
        Args:
            hidden_dim: 特征维度
            fusion_type: 
                - 'simple': gate = σ(W[concat])
                - 'independent': 两个独立门控
                - 'adaptive': 自适应门控（带残差）
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'simple':
            # 简单门控：一个门控向量
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn. Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn. Sigmoid()  # 门控信号 [0, 1]
            )
        
        elif fusion_type == 'independent':
            # 独立门控：每个分支一个门控
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
        
        elif fusion_type == 'adaptive':
            # 自适应门控：学习融合权重 + 残差
            self.gate = nn.Sequential(
                nn. Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn. Tanh(),  # Tanh 允许负值，更灵活
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # 残差权重
            self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def forward(self, h_spatial: torch.Tensor, h_expr: torch.Tensor):
        """
        Args:
            h_spatial: [N, hidden_dim]
            h_expr: [N, hidden_dim]
        
        Returns:
            h_fused: [N, hidden_dim]
            gate_info: dict，包含门控信息用于可视化
        """
        if self.fusion_type == 'simple':
            # 拼接后计算门控
            concat = torch.cat([h_spatial, h_expr], dim=-1)
            gate = self.gate(concat)  # [N, hidden_dim]
            
            # 门控融合
            h_fused = gate * h_spatial + (1 - gate) * h_expr
            
            gate_info = {
                'gate_mean': gate.mean().item(),
                'gate_std': gate. std().item(),
                'spatial_weight': gate.mean().item(),
                'expr_weight': (1 - gate).mean().item()
            }
        
        elif self.fusion_type == 'independent':
            # 独立门控
            gate_s = self.gate_spatial(h_spatial)  # [N, hidden_dim]
            gate_e = self. gate_expr(h_expr)        # [N, hidden_dim]
            
            # 🔥 关键：两个门控可以同时激活，允许信息叠加
            h_fused = gate_s * h_spatial + gate_e * h_expr
            
            gate_info = {
                'gate_spatial_mean': gate_s.mean(). item(),
                'gate_expr_mean': gate_e.mean().item(),
                'spatial_weight': gate_s.mean(). item(),
                'expr_weight': gate_e.mean().item()
            }
        
        elif self.fusion_type == 'adaptive':
            # 自适应门控 + 残差
            concat = torch.cat([h_spatial, h_expr], dim=-1)
            gate = torch.sigmoid(self.gate(concat))
            
            # 融合 + 残差连接
            h_fused = gate * h_spatial + (1 - gate) * h_expr
            h_fused = h_fused + self.residual_weight * (h_spatial + h_expr) / 2
            
            gate_info = {
                'gate_mean': gate.mean().item(),
                'residual_weight': self.residual_weight.item(),
                'spatial_weight': gate.mean().item(),
                'expr_weight': (1 - gate).mean().item()
            }
        
        return h_fused, gate_info


class DualGINEncoder(nn.Module):
    """
    双图编码器 - 门控融合版本
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        gnn_type: str = 'gine',
        n_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = 'independent'  # 🔥 融合类型
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gnn_type = gnn_type. lower()
        
        print(f"🏗️ 编码器配置:")
        print(f"  - GNN类型: {gnn_type. upper()}")
        print(f"  - 融合方式: 门控融合 ({fusion_type})")
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn. ReLU(),
            nn. Dropout(dropout)
        )
        
        # 空间图 GNN（2层）
        self.spatial_conv1, self.spatial_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        self.spatial_conv2, self.spatial_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        # 表达图 GNN（2层）
        self.expr_conv1, self.expr_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        self. expr_conv2, self.expr_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        # 🔥 门控融合模块
        self.gated_fusion = GatedFusion(hidden_dim, fusion_type=fusion_type)
        
        # 投影到 latent
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn. Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
    
    def _make_gnn_layer(self, in_dim, out_dim, n_heads):
        """创建 GNN 层"""
        if self.gnn_type == 'gine':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn. LayerNorm(out_dim),
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
            raise ValueError(f"不支持的 GNN 类型: {self.gnn_type}")
        
        ln = nn.LayerNorm(out_dim)
        return conv, ln
    
    def _apply_gnn(self, conv, ln, h, edge_index, edge_weight):
        """应用 GNN 层"""
        if self.gnn_type == 'gine':
            h_new = conv(h, edge_index, edge_attr=edge_weight. unsqueeze(-1))
        elif self.gnn_type == 'gat':
            h_new = conv(h, edge_index, edge_attr=edge_weight. unsqueeze(-1))
        elif self.gnn_type == 'gcn':
            h_new = conv(h, edge_index, edge_weight=edge_weight)
        
        h_new = ln(h_new)
        return h_new
    
    def forward(
        self,
        x: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_weight: torch. Tensor,
        expr_edge_index: torch.Tensor,
        expr_edge_weight: torch.Tensor
    ):
        """
        Args:
            x: [N, n_genes]
            spatial_edge_index: [2, E_s]
            spatial_edge_weight: [E_s]
            expr_edge_index: [2, E_e]
            expr_edge_weight: [E_e]
        
        Returns:
            z: [N, latent_dim]
            gate_info: dict with gating statistics
        """
        # 输入投影
        h = self.input_proj(x)
        
        # ==== 空间路径（2层 + 残差）====
        h_spatial = h
        h_spatial_new = self._apply_gnn(
            self.spatial_conv1, self.spatial_ln1,
            h_spatial, spatial_edge_index, spatial_edge_weight
        )
        h_spatial = F.relu(h_spatial_new) + h_spatial  # 残差
        
        h_spatial_new = self._apply_gnn(
            self. spatial_conv2, self.spatial_ln2,
            h_spatial, spatial_edge_index, spatial_edge_weight
        )
        h_spatial = F. relu(h_spatial_new) + h_spatial  # 残差
        
        # ==== 表达路径（2层 + 残差）====
        h_expr = h
        h_expr_new = self._apply_gnn(
            self.expr_conv1, self.expr_ln1,
            h_expr, expr_edge_index, expr_edge_weight
        )
        h_expr = F.relu(h_expr_new) + h_expr  # 残差
        
        h_expr_new = self._apply_gnn(
            self.expr_conv2, self. expr_ln2,
            h_expr, expr_edge_index, expr_edge_weight
        )
        h_expr = F.relu(h_expr_new) + h_expr  # 残差
        
        # ==== 🔥 门控融合 ====
        h_fused, gate_info = self.gated_fusion(h_spatial, h_expr)
        
        # 投影到 latent
        z = self.to_latent(h_fused)
        
        return z, gate_info


class UniversalDecoder(nn.Module):
    """通用解码器 - 优化版本"""
    
    def __init__(
        self,
        latent_dim: int,
        n_genes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        distribution: str = 'nb'
    ):
        super().__init__()
        
        self.distribution = distribution.lower()
        self.n_genes = n_genes
        
        print(f"📊 解码器使用分布: {self.distribution. upper()}")
        
        # 解码器骨干
        self.decoder_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn. Dropout(dropout)
        )
        
        # 均值解码器
        self.mean_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, n_genes),
            nn. Softmax(dim=-1)  # 归一化
        )
        
        # Dispersion 参数（NB/ZINB）
        if self.distribution in ['nb', 'zinb']:
            # 🔥 初始化为合理值：log(θ) = 4, θ ≈ 54
            self.px_r = nn.Parameter(torch.ones(n_genes) * 4.0)
            print(f"  初始化 dispersion (θ): {torch.exp(self.px_r). mean().item():.2f}")
        
        # Dropout 参数（ZINB）
        if self. distribution == 'zinb':
            self.dropout_decoder = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_genes)
            )
        
        # Variance 参数（Gaussian）
        if self.distribution == 'gaussian':
            self.logvar_decoder = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_genes)
            )
    
    def forward(self, z: torch.Tensor, library_size: torch.Tensor):
        """
        解码 latent 表示
        
        Args:
            z: [N, latent_dim]
            library_size: [N, 1]
        
        Returns:
            dict: 分布参数
        """
        h = self.decoder_backbone(z)
        
        # 均值（library-size 归一化）
        px_scale = self.mean_decoder(h)
        px_rate = library_size * px_scale
        
        outputs = {
            'px_rate': px_rate,
            'px_scale': px_scale
        }
        
        # NB/ZINB: dispersion
        if self.distribution in ['nb', 'zinb']:
            px_r = torch.exp(self.px_r)
            outputs['px_r'] = px_r
        
        # ZINB: dropout logits
        if self.distribution == 'zinb':
            outputs['px_dropout'] = self.dropout_decoder(h)
        
        # Gaussian: variance
        if self. distribution == 'gaussian':
            outputs['logvar'] = self. logvar_decoder(h)
            outputs['var'] = torch.exp(outputs['logvar']. clamp(min=-10, max=10))
        
        return outputs


class ContrastiveHead(nn.Module):
    """对比学习头部"""
    
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
    空间域识别模型 - 门控融合版本
    """
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        gnn_type: str = 'gine',
        n_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = 'independent',  # 🔥 融合类型
        reconstruction_loss: str = 'nb',
        # Loss weights
        kl_weight: float = 1e-6,
        contrast_weight: float = 0.1,
        local_contrast_weight: float = 1.0,
        global_contrast_weight: float = 0.5,
        # Contrastive parameters
        temperature: float = 0.07,
        contrast_sample_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss.lower()
        self.kl_weight = kl_weight
        self.contrast_weight = contrast_weight
        self.local_contrast_weight = local_contrast_weight
        self. global_contrast_weight = global_contrast_weight
        self.temperature = temperature
        self.contrast_sample_rate = contrast_sample_rate
        
        print(f"\n🎯 模型配置:")
        print(f"  - GNN类型: {gnn_type.upper()}")
        print(f"  - 融合方式: 门控融合 ({fusion_type})")
        print(f"  - 重构损失: {self.reconstruction_loss.upper()}")
        print(f"  - Latent维度: {latent_dim}")
        print(f"  - 对比学习采样率: {contrast_sample_rate * 100:.1f}%")
        
        # Encoder
        self.encoder = DualGINEncoder(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout,
            fusion_type=fusion_type
        )
        
        # Decoder
        self.decoder = UniversalDecoder(
            latent_dim=latent_dim,
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            distribution=reconstruction_loss
        )
        
        # Contrastive Head
        self.contrast_head = ContrastiveHead(latent_dim, projection_dim=128)
        
        # Buffers
        self.register_buffer("_spatial_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_spatial_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_expr_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_expr_edge_weight", torch. empty(0, dtype=torch. float32))
        self.register_buffer("_X_all", torch.empty((0, n_genes), dtype=torch.float32))
        self.register_buffer("_X_raw", torch.empty((0, n_genes), dtype=torch.float32))  # 原始数据
    
    def attach_dual_graphs(self, spatial_edge_index, spatial_edge_weight, expr_edge_index, expr_edge_weight):
        self. register_buffer("_spatial_edge_index", spatial_edge_index)
        self.register_buffer("_spatial_edge_weight", spatial_edge_weight)
        self.register_buffer("_expr_edge_index", expr_edge_index)
        self.register_buffer("_expr_edge_weight", expr_edge_weight)
        return self
    
    def attach_full_X(self, X, X_raw=None):
        """
        Args:
            X: 归一化+log变换后的数据 (用于编码器)
            X_raw: 原始计数数据 (用于NB/ZINB/Poisson损失)
        """
        self.register_buffer("_X_all", X)
        
        # 🔥 保存原始数据
        if X_raw is not None:
            self.register_buffer("_X_raw", X_raw)
        else:
            # 如果没提供,假设X就是原始数据(向后兼容)
            self.register_buffer("_X_raw", X)
        return self
    
    def forward_all(self):
        """全图前向传播"""
        if self._X_all.numel() == 0:
            raise RuntimeError("需要先调用 attach_full_X")
        
        # 🔥 编码器使用log变换数据
        X_log = self._X_all
        
        # 🔥 library size从原始数据计算
        if self.reconstruction_loss in ['nb', 'zinb', 'poisson']:
            library_size = self._X_raw.sum(dim=1, keepdim=True)
        else:
            # Gaussian等连续分布可以用log数据
            library_size = X_log.sum(dim=1, keepdim=True)
        
        # 编码
        z, gate_info = self.encoder(
            X_log,  # 🔥 编码器输入log数据
            self._spatial_edge_index,
            self._spatial_edge_weight,
            self._expr_edge_index,
            self._expr_edge_weight
        )
        
        # 解码
        decoder_outputs = self.decoder(z, library_size)
        
        return z, decoder_outputs, gate_info
    
    def compute_reconstruction_loss(self, x_log: torch.Tensor, decoder_outputs: dict, x_raw: torch.Tensor = None):
        """
        计算重构损失
        
        Args:
            x_log: log变换后的数据 (编码器输入,用于Gaussian)
            decoder_outputs: 解码器输出
            x_raw: 原始计数数据 (用于NB/ZINB/Poisson)
        """
        px_rate = decoder_outputs['px_rate']
        eps = 1e-8
        px_rate = torch.clamp(px_rate, min=eps)
        
        # 🔥 根据分布类型选择目标数据
        if self.reconstruction_loss in ['zinb', 'nb', 'poisson']:
            if x_raw is None:
                raise ValueError(f"{self.reconstruction_loss.upper()} 损失需要原始计数数据!")
            target = x_raw  # 使用原始数据
        else:
            # Gaussian等连续分布使用log数据
            target = x_log
        
        if self.reconstruction_loss == 'zinb':
            px_r = decoder_outputs['px_r']
            px_dropout = decoder_outputs['px_dropout']
            dist = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            recon_loss = -dist.log_prob(target).sum(dim=-1)
        
        elif self.reconstruction_loss == 'nb':
            px_r = decoder_outputs['px_r']
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
            recon_loss = -dist.log_prob(target).sum(dim=-1)
        
        elif self.reconstruction_loss == 'poisson':
            dist = Poisson(rate=px_rate)
            recon_loss = -dist.log_prob(target).sum(dim=-1)
        
        elif self.reconstruction_loss == 'gaussian':
            var = decoder_outputs['var']
            recon_loss = 0.5 * (
                torch.log(2 * torch.pi * var + eps) +
                (target - px_rate) ** 2 / (var + eps)
            ).sum(dim=-1)
        
        else:
            raise ValueError(f"不支持的重构损失: {self.reconstruction_loss}")
        
        return recon_loss
    
    def contrastive_loss_fast(self, z: torch.Tensor, contrast_samples: dict):
        """加速版对比学习"""
        device = z.device
        n_spots = z.size(0)
        
        # 随机采样 anchor
        n_sample = max(int(n_spots * self.contrast_sample_rate), 100)
        sampled_indices = torch.randperm(n_spots, device=device)[:n_sample]. cpu(). numpy()
        
        z_proj = self.contrast_head(z)
        
        local_loss = 0.0
        global_loss = 0.0
        n_valid_local = 0
        n_valid_global = 0
        
        for i in sampled_indices:
            anchor = z_proj[i:i+1]
            
            # 局部对比
            local_pos_idx = contrast_samples['local_pos'][i]
            local_neg_idx = contrast_samples['local_neg'][i]
            
            if len(local_pos_idx) > 0 and len(local_neg_idx) > 0:
                local_pos = z_proj[local_pos_idx]
                local_neg = z_proj[local_neg_idx]
                
                pos_sim = torch.mm(anchor, local_pos. t()) / self.temperature
                neg_sim = torch.mm(anchor, local_neg.t()) / self. temperature
                
                logits = torch.cat([pos_sim. mean(dim=1, keepdim=True), neg_sim], dim=1)
                loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
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
        
        local_loss = local_loss / max(n_valid_local, 1)
        global_loss = global_loss / max(n_valid_global, 1)
        
        total_contrast_loss = (
            self.local_contrast_weight * local_loss +
            self.global_contrast_weight * global_loss
        )
        
        return total_contrast_loss, local_loss, global_loss
    
    @torch.no_grad()
    def get_latent_representation(self):
        """获取 latent 表示"""
        self.eval()
        z, _, _ = self.forward_all()
        return z. cpu().numpy()