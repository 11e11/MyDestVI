"""空间域识别模型 - GIB + 对比学习版本（双重重构）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from scvi.module.base import BaseModuleClass
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

from .gib_pruner import GIBPruner
from .contrasive_loss import NodeContrastiveLoss
from .dense_gcn import DenseGCNEncoder
from .prototype_learning import LearnablePrototypes, PrototypeLoss
from .losses import SpatialConsistencyLoss


class SpatialDomainModelGIB(BaseModuleClass):
    """
    GIB + 对比学习 + 原型学习（双重重构版本）
    
    🔥 关键改进：双重重构（Dual Reconstruction）
    - Anchor View:   完整空间图 → 重构 + 原型 + 空间一致性
    - Learner View: GIB 剪枝图 → 重构（🔥 NEW）+ 对比学习
    
    目标：强制 GIB 感受"剪掉关键边 = 重构失败"的压力
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_dim: int = 256,
        latent_dim:  int = 32,
        n_domains: int = 7,
        # GIB 参数
        gib_hidden: int = 64,
        gib_min_keep: float = 0.05,
        gib_max_keep: float = 0.8,
        gib_sparsity_weight: float = 1.0,
        # 对比学习参数
        contrastive_temperature: float = 0.1,
        contrastive_weight: float = 1.0,
        # 原型学习参数
        prototype_dim: int = 128,
        prototype_temperature: float = 0.15,
        prototype_weight: float = 20.0,
        # 🔥 NEW:  双重重构参数
        learner_recon_weight: float = 0.5,  # Learner 分支的重构损失权重
        # 空间一致性参数
        spatial_consistency_weight: float = 0.1,
        # 其他
        gnn_num_layers: int = 3,
        dropout: float = 0.1,
        reconstruction_loss: str = 'zinb',
        **kwargs
    ):
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        self.latent_dim = latent_dim
        self.n_domains = n_domains
        self.contrastive_weight = contrastive_weight
        self.prototype_weight = prototype_weight
        self.gib_sparsity_weight = gib_sparsity_weight
        self.learner_recon_weight = learner_recon_weight  # 🔥 NEW
        self.spatial_consistency_weight = spatial_consistency_weight
        self.reconstruction_loss = reconstruction_loss
        
        print(f"\n🎯 GIB 对比学习模型（双重重构版）:")
        print(f"  - Input: {n_input}, Output: {n_output}")
        print(f"  - Latent:  {latent_dim}, Domains: {n_domains}")
        print(f"  - Contrastive weight: {contrastive_weight}")
        print(f"  - Prototype weight: {prototype_weight}")
        print(f"  - GIB sparsity weight: {gib_sparsity_weight}")
        print(f"  - 🔥 Learner recon weight: {learner_recon_weight} (NEW! )")
        print(f"  - Spatial consistency weight: {spatial_consistency_weight}")
        
        # 1.输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2.GIB 剪枝器
        self.gib_pruner = GIBPruner(
            node_dim=n_input,
            hidden_dim=gib_hidden,
            min_keep_rate=gib_min_keep,
            max_keep_rate=gib_max_keep,
            temperature=0.3
        )
        
        # 3.共享 GNN 编码器
        self.gnn_encoder = DenseGCNEncoder(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout
        )
        
        # 4.投影到 Latent
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 5.原型学习
        self.latent_to_prototype = nn.Sequential(
            nn.Linear(latent_dim, prototype_dim),
            nn.ReLU(),
            nn.Linear(prototype_dim, prototype_dim)
        )
        
        self.prototypes = LearnablePrototypes(
            n_prototypes=n_domains,
            prototype_dim=prototype_dim,
            temperature=prototype_temperature,
            momentum=0.99
        )
        
        self.prototype_loss_fn = PrototypeLoss(temperature=prototype_temperature)
        
        # 6.对比学习损失
        self.contrastive_loss_fn = NodeContrastiveLoss(
            temperature=contrastive_temperature,
            negative_mode='all'
        )
        
        # 7.空间一致性损失
        self.spatial_consistency_loss_fn = SpatialConsistencyLoss(
            loss_type='mse',
            reduction='mean'
        )
        
        # 🔥 8.解码器（共享，但可以接收两个视图的 latent）
        self._build_decoder(latent_dim, n_output, hidden_dim, dropout, reconstruction_loss)
        
        # Buffers
        self.register_buffer("_spatial_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_spatial_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_X_pca", torch.empty((0, n_input), dtype=torch.float32))
        self.register_buffer("_X_raw", torch.empty((0, n_output), dtype=torch.float32))
    
    def _build_decoder(self, latent_dim, n_output, hidden_dim, dropout, distribution):
        """构建解码器（共享）"""
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
    
    def forward_all(self):
        """
        完整前向传播（双重重构版本）
        
        Returns:
            z_anchor: anchor view 的 latent
            z_learner: learner view 的 latent
            decoder_outputs_anchor: anchor 的解码器输出
            decoder_outputs_learner: 🔥 NEW:  learner 的解码器输出
            info: 统计信息
        """
        X_pca = self._X_pca
        N = X_pca.size(0)
        device = X_pca.device
        
        library_size = self._X_raw.sum(dim=1, keepdim=True)
        info = {}
        
        # 1.GIB 剪枝生成增强图
        aug_edge_index, aug_edge_weight, keep_probs, sparsity_loss = self.gib_pruner(
            X_pca,
            self._spatial_edge_index,
            self._spatial_edge_weight
        )
        
        info['gib_keep_rate'] = (keep_probs > 0.5).float().mean().item()
        info['gib_avg_prob'] = keep_probs.mean().item()
        info['gib_sparsity_loss'] = sparsity_loss.item()
        
        # 2.构建稠密邻接矩阵
        A_anchor = torch.zeros((N, N), device=device)
        A_anchor[self._spatial_edge_index[0], self._spatial_edge_index[1]] = self._spatial_edge_weight
        
        A_learner = torch.zeros((N, N), device=device)
        A_learner[aug_edge_index[0], aug_edge_index[1]] = aug_edge_weight
        
        # 3.输入投影
        h = self.input_proj(X_pca)
        
        # 4.双视图编码
        h_anchor = self.gnn_encoder(h, A_anchor)
        h_learner = self.gnn_encoder(h, A_learner)
        
        # 5.投影到 Latent
        z_anchor = self.to_latent(h_anchor)
        z_learner = self.to_latent(h_learner)
        
        # 🔥 6.双重解码
        decoder_outputs_anchor = self._decode(z_anchor, library_size)
        decoder_outputs_learner = self._decode(z_learner, library_size)  # 🔥 NEW
        
        # 传递 sparsity_loss
        info['_gib_sparsity_loss_tensor'] = sparsity_loss
        
        return z_anchor, z_learner, decoder_outputs_anchor, decoder_outputs_learner, info
    
    def _decode(self, z, library_size):
        """解码器（共享）"""
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
        else:
            raise ValueError(f"Unsupported loss:  {self.reconstruction_loss}")
        
        return recon_loss
    
    def compute_contrastive_loss(self, z_anchor, z_learner):
        """对比学习损失"""
        return self.contrastive_loss_fn(z_anchor, z_learner)
    
    def compute_prototype_loss(self, z_latent):
        """原型损失"""
        z_proto = self.latent_to_prototype(z_latent)
        Q, labels = self.prototypes(z_proto, update=True)
        prototypes = self.prototypes.prototypes
        loss, loss_dict = self.prototype_loss_fn(z_proto, prototypes, Q, labels)
        return loss, loss_dict, Q
    
    def compute_spatial_consistency_loss(self, Q):
        """空间一致性损失"""
        return self.spatial_consistency_loss_fn(Q, self._spatial_edge_index)
    
    @torch.no_grad()
    def get_latent_representation(self):
        """获取 latent（使用 anchor view）"""
        self.eval()
        z_anchor, _, _, _, _ = self.forward_all()  # 🔥 注意：现在返回 5 个值
        return z_anchor.cpu().numpy()