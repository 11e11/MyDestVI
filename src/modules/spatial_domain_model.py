"""空间域识别模型 - STAGUE 模式（语义感知 GIB）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.module.base import BaseModuleClass

from .gib_pruner import GIBPruner
from .contrasive_loss import NodeContrastiveLoss
from .dense_gcn import DenseGCNEncoder
from .prototype_learning import LearnablePrototypes, PrototypeLoss


class SpatialDomainModelGIB(BaseModuleClass):
    """
    STAGUE 模式：语义感知的 GIB 剪枝
    
    🔥 核心改进：
    1.第一步：在原始图上编码，得到语义特征（Anchor）
    2.第二步：用语义特征指导 GIB 剪枝，生成 Learner 图
    3.第三步：在 Learner 图上二次编码，得到纯净特征（Learner）
    """
    
    def __init__(
        self,
        n_input:  int,
        n_output: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        n_domains: int = 7,
        # GIB 参数
        gib_hidden:  int = 64,
        gib_min_keep:  float = 0.05,
        gib_max_keep:  float = 0.8,
        gib_sparsity_weight: float = 1.0,
        gib_warmup_epochs: int = 50,  # 🔥 NEW: GIB 预热期
        # 对比学习参数
        contrastive_temperature: float = 0.1,
        contrastive_weight: float = 1.0,
        # 原型学习参数
        prototype_dim: int = 128,
        prototype_temperature: float = 0.15,
        prototype_weight: float = 20.0,
        # 其他
        gnn_num_layers: int = 3,
        dropout:  float = 0.1,
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
        self.gib_warmup_epochs = gib_warmup_epochs  # 🔥 NEW
        self.reconstruction_loss = reconstruction_loss
        
        # 🔥 用于跟踪当前 epoch
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))
        
        print(f"\n🎯 STAGUE 模式（语义感知 GIB）:")
        print(f"  - Input: {n_input}, Output: {n_output}")
        print(f"  - Latent:  {latent_dim}, Domains: {n_domains}")
        print(f"  - 🔥 GIB Warmup: {gib_warmup_epochs} epochs")
        print(f"  - Contrastive weight: {contrastive_weight}")
        print(f"  - Prototype weight: {prototype_weight}")
        print(f"  - GIB sparsity weight: {gib_sparsity_weight}")
        
        # 1.输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2.🔥 GIB 剪枝器（输入是 hidden_dim 的语义特征）
        self.gib_pruner = GIBPruner(
            node_dim=hidden_dim,  # 🔥 改为 hidden_dim（不再是 n_input）
            hidden_dim=gib_hidden,
            min_keep_rate=gib_min_keep,
            max_keep_rate=gib_max_keep,
            temperature=0.3,
            use_soft_mask=True
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
        
        # 7.解码器（仅用于 Anchor）
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
    
    def set_epoch(self, epoch):
        """🔥 NEW: 设置当前 epoch（用于判断是否在 warmup 期）"""
        self.current_epoch = torch.tensor(epoch, dtype=torch.long, device=self.current_epoch.device)
    
    def forward_all(self):
        """
        完整前向传播（STAGUE 两步走）
        
        🔥 流程：
        1.第一步：在原始图上编码 → Anchor 语义特征
        2.第二步：用 Anchor 语义特征指导 GIB 剪枝 → Learner 图
        3.第三步：在 Learner 图上二次编码 → Learner latent
        
        Returns:
            z_anchor: anchor view 的 latent
            z_learner: learner view 的 latent
            decoder_outputs: anchor 的解码器输出
            info: 统计信息
        """
        X_pca = self._X_pca
        N = X_pca.size(0)
        device = X_pca.device
        
        library_size = self._X_raw.sum(dim=1, keepdim=True)
        info = {}
        
        # 🔥 判断是否在 GIB warmup 期
        in_gib_warmup = self.current_epoch < self.gib_warmup_epochs
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 第一步：在原始图上编码（Anchor View）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1.构建原始图的稠密邻接矩阵
        A_anchor = torch.zeros((N, N), device=device)
        A_anchor[self._spatial_edge_index[0], self._spatial_edge_index[1]] = self._spatial_edge_weight
        
        # 2.输入投影
        h = self.input_proj(X_pca)  # [N, hidden_dim]
        
        # 3.在原始图上编码，得到 Anchor 语义特征
        h_anchor = self.gnn_encoder(h, A_anchor)  # [N, hidden_dim]
        
        # 4.投影到 Latent
        z_anchor = self.to_latent(h_anchor)  # [N, latent_dim]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 第二步：用 Anchor 语义特征指导 GIB 剪枝
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if in_gib_warmup: 
            # Warmup 期：直接使用原始图（不剪枝）
            aug_edge_index = self._spatial_edge_index
            aug_edge_weight = self._spatial_edge_weight
            keep_probs = torch.ones(self._spatial_edge_index.size(1), device=device)
            sparsity_loss = torch.tensor(0.0, device=device)
            
            info['gib_keep_rate'] = 1.0
            info['gib_avg_prob'] = 1.0
            info['gib_sparsity_loss'] = 0.0
            info['in_gib_warmup'] = True
        else:
            # 正式训练期：使用 GIB 剪枝
            aug_edge_index, aug_edge_weight, keep_probs, sparsity_loss = self.gib_pruner(
                h_anchor,  # 🔥 输入是 Anchor 语义特征（不是原始 PCA）
                self._spatial_edge_index,
                self._spatial_edge_weight
            )
            
            info['gib_keep_rate'] = (keep_probs > 0.5).float().mean().item()
            info['gib_avg_prob'] = keep_probs.mean().item()
            info['gib_sparsity_loss'] = sparsity_loss.item()
            info['in_gib_warmup'] = False
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 第三步：在 Learner 图上二次编码
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1.构建 Learner 图的稠密邻接矩阵
        A_learner = torch.zeros((N, N), device=device)
        A_learner[aug_edge_index[0], aug_edge_index[1]] = aug_edge_weight
        
        # 2.在 Learner 图上编码（使用原始投影特征 h）
        h_learner = self.gnn_encoder(h, A_learner)  # [N, hidden_dim]
        
        # 3.投影到 Latent
        z_learner = self.to_latent(h_learner)  # [N, latent_dim]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 解码（仅 Anchor）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        decoder_outputs = self._decode(z_anchor, library_size)
        
        # 传递 sparsity_loss
        info['_gib_sparsity_loss_tensor'] = sparsity_loss
        
        return z_anchor, z_learner, decoder_outputs, info
    
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
        """重构损失（仅 Anchor）"""
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
            raise ValueError(f"Unsupported loss: {self.reconstruction_loss}")
        
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
    
    @torch.no_grad()
    def get_latent_representation(self):
        """获取 latent（使用 anchor view）"""
        self.eval()
        z_anchor, _, _, _ = self.forward_all()
        return z_anchor.cpu().numpy()