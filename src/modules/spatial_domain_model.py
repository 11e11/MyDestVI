"""空间域识别模型 - 门控融合版本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv, GCNConv
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch.distributions import Poisson, Normal
from scvi import REGISTRY_KEYS


class GatedFusion(nn.Module):
    """门控融合模块（保持不变）"""
    
    def __init__(self, hidden_dim:  int, fusion_type: str = 'simple'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'simple':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn. Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn. Sigmoid()
            )
        
        elif fusion_type == 'independent':
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
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn. Tanh(),
                nn. Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        else:
            raise ValueError(f"不支持的融合类型:  {fusion_type}")
    
    def forward(self, h_spatial:  torch.Tensor, h_expr: torch.Tensor):
        if self.fusion_type == 'simple':
            concat = torch.cat([h_spatial, h_expr], dim=-1)
            gate = self.gate(concat)
            h_fused = gate * h_spatial + (1 - gate) * h_expr
            gate_info = {
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'spatial_weight': gate.mean().item(),
                'expr_weight': (1 - gate).mean().item()
            }
        
        elif self.fusion_type == 'independent':
            gate_s = self.gate_spatial(h_spatial)
            gate_e = self.gate_expr(h_expr)
            h_fused = gate_s * h_spatial + gate_e * h_expr
            gate_info = {
                'gate_spatial_mean': gate_s.mean().item(),
                'gate_expr_mean': gate_e.mean().item(),
                'spatial_weight': gate_s.mean().item(),
                'expr_weight': gate_e.mean().item()
            }
        
        elif self.fusion_type == 'adaptive':
            concat = torch.cat([h_spatial, h_expr], dim=-1)
            gate = torch.sigmoid(self.gate(concat))
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
    🔥 修改：支持不同的输入和输出维度
    """
    
    def __init__(
        self,
        n_input: int,  # 🔥 新增：输入维度（PCA 维度，如 50）
        hidden_dim: int = 256,
        latent_dim: int = 64,
        gnn_type: str = 'gine',
        n_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = 'independent'
    ):
        super().__init__()
        self.n_input = n_input
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gnn_type = gnn_type. lower()
        
        print(f"🏗️ 编码器配置:")
        print(f"  - 输入维度: {n_input}")  # 🔥 打印输入维度
        print(f"  - GNN类型: {gnn_type. upper()}")
        print(f"  - 融合方式: 门控融合 ({fusion_type})")
        
        # 🔥 输入投影：从 n_input 维到 hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 空间图 GNN（2层）
        self.spatial_conv1, self.spatial_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        self.spatial_conv2, self.spatial_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        # 表达图 GNN（2层）
        self.expr_conv1, self.expr_ln1 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        self.expr_conv2, self. expr_ln2 = self._make_gnn_layer(hidden_dim, hidden_dim, n_heads)
        
        # 门控融合模块
        self.gated_fusion = GatedFusion(hidden_dim, fusion_type=fusion_type)
        
        # 投影到 latent
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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
                nn. Linear(out_dim, out_dim)
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
        x: torch.Tensor,  # 🔥 现在是 [N, n_input] 维
        spatial_edge_index: torch.Tensor,
        spatial_edge_weight: torch.Tensor,
        expr_edge_index: torch.Tensor,
        expr_edge_weight: torch.Tensor
    ):
        """
        Args:
            x: [N, n_input]  # 🔥 PCA 降维后的数据
        
        Returns:
            z: [N, latent_dim]
            gate_info: dict
        """
        # 输入投影
        h = self.input_proj(x)
        
        # 空间路径（2层 + 残差）
        h_spatial = h
        h_spatial_new = self._apply_gnn(
            self.spatial_conv1, self.spatial_ln1,
            h_spatial, spatial_edge_index, spatial_edge_weight
        )
        h_spatial = F.relu(h_spatial_new) + h_spatial
        
        h_spatial_new = self._apply_gnn(
            self. spatial_conv2, self.spatial_ln2,
            h_spatial, spatial_edge_index, spatial_edge_weight
        )
        h_spatial = F. relu(h_spatial_new) + h_spatial
        
        # 表达路径（2层 + 残差）
        h_expr = h
        h_expr_new = self._apply_gnn(
            self.expr_conv1, self.expr_ln1,
            h_expr, expr_edge_index, expr_edge_weight
        )
        h_expr = F.relu(h_expr_new) + h_expr
        
        h_expr_new = self._apply_gnn(
            self.expr_conv2, self.expr_ln2,
            h_expr, expr_edge_index, expr_edge_weight
        )
        h_expr = F.relu(h_expr_new) + h_expr
        
        # 门控融合
        h_fused, gate_info = self.gated_fusion(h_spatial, h_expr)
        
        # 投影到 latent
        z = self.to_latent(h_fused)
        
        return z, gate_info


class UniversalDecoder(nn.Module):
    """
    通用解码器 - 针对 PCA 重构优化
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_output: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        distribution: str = 'gaussian'
    ):
        super().__init__()
        
        self.distribution = distribution. lower()
        self.n_output = n_output
        
        print(f"📊 解码器配置:")
        print(f"  - 输出维度: {n_output}")
        print(f"  - 使用分布: {self.distribution. upper()}")
        
        # 解码器骨干
        self.decoder_backbone = nn.Sequential(
            nn. Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn. Dropout(dropout)
        )
        
        # 🔥 对于 Gaussian 损失，直接输出重构值
        if self.distribution == 'gaussian':
            self.output_layer = nn.Linear(hidden_dim // 2, n_output)
        
        # 保留 NB 支持
        elif self.distribution in ['nb', 'zinb']:
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_output),
                nn. Softmax(dim=-1)
            )
            self.px_r = nn.Parameter(torch.ones(n_output) * 4.0)
    
    def forward(self, z: torch.Tensor, library_size: torch.Tensor):
        """
        解码 latent 表示
        
        Args: 
            z: [N, latent_dim]
            library_size: [N, 1]（对 PCA 重构无用，保留接口兼容性）
        
        Returns:  
            dict: 分布参数
        """
        h = self.decoder_backbone(z)
        
        if self. distribution == 'gaussian':
            # 🔥 直接输出重构的 PCA
            px_rate = self.output_layer(h)  # [N, n_output]
            
            outputs = {
                'px_rate': px_rate,
                'px_scale': px_rate  # 保持接口一致
            }
        
        elif self.distribution in ['nb', 'zinb']: 
            # NB 分支（保留）
            px_scale = self.mean_decoder(h)
            px_rate = library_size * px_scale
            px_r = torch.exp(self.px_r)
            
            outputs = {
                'px_rate': px_rate,
                'px_scale': px_scale,
                'px_r':  px_r
            }
        
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
    空间域识别模型 - PCA 输入输出版本
    """
    
    def __init__(
        self,
        n_input: int,   # PCA 维度（如 50）
        n_output: int,  # 🔥 输出维度（与 n_input 相同）
        hidden_dim: int = 256,
        latent_dim: int = 30,
        gnn_type: str = 'gine',
        n_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = 'independent',
        reconstruction_loss: str = 'gaussian',  # 🔥 默认改为 gaussian
        # Loss weights
        contrast_weight: float = 0.1,
        local_contrast_weight: float = 1.0,
        global_contrast_weight: float = 0.5,
        # Contrastive parameters
        temperature: float = 0.1,
        contrast_sample_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss. lower()
        self.contrast_weight = contrast_weight
        self.local_contrast_weight = local_contrast_weight
        self.global_contrast_weight = global_contrast_weight
        self. temperature = temperature
        self.contrast_sample_rate = contrast_sample_rate
        
        print(f"\n🎯 模型配置:")
        print(f"  - 输入维度: {n_input} (PCA)")
        print(f"  - 输出维度: {n_output} (PCA)")
        print(f"  - Latent 维度: {latent_dim}")
        print(f"  - 重构损失: {self.reconstruction_loss. upper()}")
        print(f"  - Temperature: {temperature}")
        
        # Encoder
        self.encoder = DualGINEncoder(
            n_input=n_input,
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
            n_output=n_output,
            hidden_dim=hidden_dim,
            dropout=dropout,
            distribution=reconstruction_loss
        )
        
        # Contrastive Head
        self.  contrast_head = ContrastiveHead(latent_dim, projection_dim=128)
        
        # Buffers
        self.register_buffer("_spatial_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_spatial_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_expr_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_expr_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_X_pca", torch.empty((0, n_input), dtype=torch.float32))
    
    def attach_dual_graphs(self, spatial_edge_index, spatial_edge_weight, expr_edge_index, expr_edge_weight):
        self. register_buffer("_spatial_edge_index", spatial_edge_index)
        self.register_buffer("_spatial_edge_weight", spatial_edge_weight)
        self.register_buffer("_expr_edge_index", expr_edge_index)
        self.register_buffer("_expr_edge_weight", expr_edge_weight)
        return self
    
    def attach_full_X(self, X_pca):
        """
        🔥 修改：只需要 PCA 数据
        
        Args:
            X_pca:  PCA 数据 [N, n_input]
        """
        self.register_buffer("_X_pca", X_pca)
        return self
    
    def forward_all(self):
        """全图前向传播"""
        if self._X_pca.numel() == 0:
            raise RuntimeError("需要先调用 attach_full_X")
        
        X_pca = self._X_pca
        
        # 🔥 library size 设为 1（不需要了，因为输出是 PCA）
        library_size = torch.ones((X_pca.shape[0], 1), device=X_pca.device)
        
        # 编码
        z, gate_info = self.encoder(
            X_pca,
            self._spatial_edge_index,
            self._spatial_edge_weight,
            self._expr_edge_index,
            self._expr_edge_weight
        )
        
        # 解码
        decoder_outputs = self.decoder(z, library_size)
        
        return z, decoder_outputs, gate_info
    
    def compute_reconstruction_loss(self, decoder_outputs:  dict, x_target: torch.  Tensor):
        """
        计算重构损失
        
        Args:
            decoder_outputs: 解码器输出
            x_target: 目标数据（PCA）
        """
        if self.reconstruction_loss == 'gaussian':
            # 🔥 MSE 损失（最简单有效）
            px_rate = decoder_outputs['px_rate']  # 实际上就是重构的 PCA
            recon_loss = F.mse_loss(px_rate, x_target, reduction='none').sum(dim=-1)
        
        elif self.reconstruction_loss == 'nb':
            # 保留 NB 支持（如果你还想用）
            px_rate = decoder_outputs['px_rate']
            px_r = decoder_outputs['px_r']
            eps = 1e-8
            px_rate = torch.clamp(px_rate, min=eps)
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
            recon_loss = -dist.log_prob(x_target).sum(dim=-1)
        
        else:
            raise ValueError(f"不支持的重构损失:  {self.reconstruction_loss}")
        
        return recon_loss
    
    def contrastive_loss_fast(self, z:  torch.Tensor, contrast_samples: dict):
        """加速版对比学习 - 修复数值稳定性"""
        device = z.device
        n_spots = z.size(0)
        
        n_sample = max(int(n_spots * self.contrast_sample_rate), 100)
        sampled_indices = torch.randperm(n_spots, device=device)[:n_sample].cpu().numpy()
        
        z_proj = self.contrast_head(z)
        
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
                
                if len(local_neg_idx) > 0:
                    local_neg = z_proj[local_neg_idx]
                    pos_sim = torch.mm(anchor, local_pos. t()) / self.temperature
                    neg_sim = torch.mm(anchor, local_neg.t()) / self.temperature
                    logits = torch.cat([pos_sim. mean(dim=1, keepdim=True), neg_sim], dim=1)
                    loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
                else:
                    # 🔥 只有正样本：使用 softplus（数值稳定）
                    pos_sim = torch.mm(anchor, local_pos.t()) / self.temperature
                    loss = torch.mean(F.softplus(-pos_sim))
                
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
            local_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        if n_valid_global > 0:
            global_loss = global_loss / n_valid_global
        else: 
            global_loss = torch. tensor(0.0, device=device, dtype=torch.float32)
        
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