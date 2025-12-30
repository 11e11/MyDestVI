"""空间域识别模型 - Attention-GIB + 双视图对比学习（修复版）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .gib_pruner import AttentionGIBPruner
from .contrasive_loss import NodeContrastiveLoss


class SpatialDomainModel(nn.Module):
    """
    Attention-GIB 双视图对比学习模型（修复版）
    
    🔥 核心改进：
    1.边权重与 Attention Prob 解耦
    2.引入 Entropy Loss 促进二元决策
    """
    
    def __init__(
        self,
        n_input:  int,
        n_output:  int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        # Attention-GIB 参数
        gib_num_heads: int = 4,
        gib_temperature: float = 1.0,
        gib_use_gumbel: bool = True,
        gib_gumbel_tau: float = 0.5,
        gib_entropy_weight: float = 0.01,  # 🔥 NEW
        # GCN 参数
        gnn_num_layers: int = 3,
        dropout: float = 0.1,
        # 损失权重
        contrastive_weight: float = 1.0,
        contrastive_temperature: float = 0.1,
        reconstruction_weight: float = 1.0,
        gib_loss_weight: float = 0.1,  # 🔥 改名：原 sparsity_weight
        # 重构损失类型
        reconstruction_loss:  str = 'zinb',
        **kwargs
    ):
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.gib_loss_weight = gib_loss_weight
        self.reconstruction_loss = reconstruction_loss
        
        print(f"\n🎯 Attention-GIB 双视图对比学习模型 (修复版):")
        print(f"  - Input (PCA): {n_input}, Output (Genes): {n_output}")
        print(f"  - Hidden:  {hidden_dim}, Latent: {latent_dim}")
        print(f"  - GNN Layers: {gnn_num_layers}")
        print(f"  - Contrastive Weight:  {contrastive_weight}")
        print(f"  - Reconstruction Weight: {reconstruction_weight}")
        print(f"  - GIB Loss Weight: {gib_loss_weight}")
        print(f"  - 🔥 边权重与 Prob 解耦")
        
        # Step 2: 基础特征投影
        self.shared_encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Step 3: Attention-GIB 剪枝模块
        self.attention_gib = AttentionGIBPruner(
            input_dim=n_input,
            hidden_dim=64,
            num_heads=gib_num_heads,
            dropout=dropout,
            temperature=gib_temperature,
            # use_gumbel=gib_use_gumbel,
            # gumbel_tau=gib_gumbel_tau,
            entropy_weight=gib_entropy_weight  # 🔥 NEW
        )
        
        # Step 4: 双视图 GCN 编码器
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(gnn_num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # 投影到 Latent 空间
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Step 5: 损失函数
        self.contrastive_loss_fn = NodeContrastiveLoss(
            temperature=contrastive_temperature,
            negative_mode='all'
        )
        
        # 重构解码器
        self._build_decoder(latent_dim, n_output, hidden_dim, dropout, reconstruction_loss)
        
        # Buffers
        self.register_buffer("_spatial_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_spatial_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_X_pca", torch.empty((0, n_input), dtype=torch.float32))
        self.register_buffer("_X_raw", torch.empty((0, n_output), dtype=torch.float32))
    
    def _build_decoder(self, latent_dim, n_output, hidden_dim, dropout, distribution):
        """构建解码器"""
        self.decoder_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decoder_mean = nn.Sequential(
            nn.Linear(hidden_dim // 2, n_output),
            nn.Softmax(dim=-1)
        )
        
        if distribution in ['nb', 'zinb']:
            self.px_r = nn.Parameter(torch.ones(n_output) * 4.0)
        
        if distribution == 'zinb': 
            self.decoder_dropout = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_output)
            )
    
    def attach_spatial_graph(self, edge_index, edge_weight):
        """附加空间图"""
        self.register_buffer("_spatial_edge_index", edge_index)
        self.register_buffer("_spatial_edge_weight", edge_weight)
        return self
    
    def attach_full_X(self, X_pca, X_raw):
        """附加特征"""
        self.register_buffer("_X_pca", X_pca)
        self.register_buffer("_X_raw", X_raw)
        return self
    
    def encode_with_gcn(self, h_base, edge_index, edge_weight):
        """使用 GCN 编码"""
        h = h_base
        
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            h = gcn(h, edge_index, edge_weight)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        z = self.to_latent(h)
        return z
    
    def forward_all(self):
        """完整前向传播"""
        X_pca = self._X_pca
        N = X_pca.size(0)
        device = X_pca.device
        
        library_size = self._X_raw.sum(dim=1, keepdim=True)
        info = {}
        
        # Step 2: 基础特征投影
        h_base = self.shared_encoder(X_pca)
        
        # Step 3: Attention-GIB 剪枝
        pruned_edge_index, pruned_edge_weight, attention_probs, gib_loss = self.attention_gib(
            X_pca,
            self._spatial_edge_index,
            self._spatial_edge_weight
        )
        
        info['gib_keep_rate'] = (attention_probs > 0.05).float().mean().item()
        info['gib_max_prob'] = attention_probs.max().item()
        info['gib_min_prob'] = attention_probs.min().item()
        info['gib_loss'] = gib_loss.item()
        info['_gib_loss_tensor'] = gib_loss
        # 在 info 字典中添加
        info['gib_effective_sparsity'] = self.attention_gib.get_effective_sparsity(
            attention_probs, 
            threshold=0.05  # 权重 < 0.05 视为"被剪枝"
        )
        
        # Step 4: 双视图 GCN 编码
        # 视图 1: 结构视图 (原始空间图)
        z_spatial = self.encode_with_gcn(
            h_base,
            self._spatial_edge_index,
            self._spatial_edge_weight
        )
        
        # 视图 2: 特征视图 (剪枝后的图)
        z_pruned = self.encode_with_gcn(
            h_base,
            pruned_edge_index,
            pruned_edge_weight
        )
        
        # 解码
        decoder_outputs = self._decode(z_spatial, library_size)
        
        return z_spatial, z_pruned, decoder_outputs, info
    
    def _decode(self, z, library_size):
        """解码器"""
        h = self.decoder_backbone(z)
        px_scale = self.decoder_mean(h)
        px_rate = library_size * px_scale
        
        outputs = {'px_rate': px_rate, 'px_scale': px_scale}
        
        if hasattr(self, 'px_r'):
            outputs['px_r'] = torch.exp(self.px_r)
        if hasattr(self, 'decoder_dropout'):
            outputs['px_dropout'] = self.decoder_dropout(h)
        
        return outputs
    
    def compute_reconstruction_loss(self, decoder_outputs, x_raw):
        """重构损失"""
        px_rate = decoder_outputs['px_rate']
        eps = 1e-8
        px_rate = torch.clamp(px_rate, min=eps)
        
        if self.reconstruction_loss == 'zinb':
            px_r = decoder_outputs['px_r']
            px_dropout = decoder_outputs['px_dropout']
            from scvi.distributions import ZeroInflatedNegativeBinomial
            dist = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        elif self.reconstruction_loss == 'nb':
            px_r = decoder_outputs['px_r']
            from scvi.distributions import NegativeBinomial
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        elif self.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(px_rate, x_raw, reduction='none').sum(dim=-1)
        else:
            raise ValueError(f"Unsupported loss:  {self.reconstruction_loss}")
        
        return recon_loss
    
    def compute_contrastive_loss(self, z_spatial, z_pruned):
        """对比学习损失"""
        return self.contrastive_loss_fn(z_spatial, z_pruned)
    
    @torch.no_grad()
    def get_latent_representation(self):
        """获取 latent 表示"""
        self.eval()
        z_spatial, _, _, _ = self.forward_all()
        return z_spatial.cpu().numpy()
