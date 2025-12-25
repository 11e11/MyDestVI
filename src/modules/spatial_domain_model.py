"""空间域识别模型 - V2修复版"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from scvi.module.base import BaseModuleClass
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch.distributions import Poisson

from .differentiable_graph_learning import DifferentiableGraphLearner
from .dense_gcn import DenseGCNEncoder
from .prototype_learning import LearnablePrototypes, PrototypeLoss
from .losses import GraphSparsityLoss


class SpatialDomainModelV2(BaseModuleClass):
    """空间域识别模型V2 - 修复版"""
    
    def __init__(
        self,
        n_input:  int,
        n_output: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        n_domains: int = 7,
        # 图学习参数
        use_graph_learning: bool = False,  # 🔥 关键参数
        k_learned_neighbors: int = 15,
        graph_learning_hidden:  int = 128,
        # 原型学习参数
        use_prototype: bool = True,
        prototype_dim: int = 128,
        prototype_temperature: float = 0.1,
        prototype_momentum: float = 0.99,
        # GNN参数
        gnn_num_layers: int = 3,
        dropout: float = 0.1,
        # 损失权重
        prototype_weight: float = 20.0,
        graph_sparsity_weight: float = 0.01,
        # 其他
        reconstruction_loss:  str = 'nb',
        **kwargs
    ):
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        self.latent_dim = latent_dim
        self.n_domains = n_domains
        self.use_graph_learning = use_graph_learning  # 🔥 保存状态
        self.use_prototype = use_prototype
        self.prototype_weight = prototype_weight
        self.graph_sparsity_weight = graph_sparsity_weight
        self.reconstruction_loss = reconstruction_loss
        
        print(f"\n🎯 空间域识别模型 V2 (修复版):")
        print(f"  - 输入维度: {n_input}")
        print(f"  - 输出维度: {n_output}")
        print(f"  - Latent维度: {latent_dim}")
        print(f"  - 域数量: {n_domains}")
        print(f"  - 🔥 图学习: {use_graph_learning}")
        print(f"  - 🔥 原型学习:  {use_prototype}")
        
        # 1.输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2.可微图学习器（条件创建）
        if use_graph_learning:
            self.graph_learner = DifferentiableGraphLearner(
                n_input=n_input,
                hidden_dim=graph_learning_hidden,
                k_neighbors=k_learned_neighbors,
                temperature=1.0,
                use_gumbel=True
            )
        
        # 3.GNN编码器
        # 3a.空间图编码器
        self.spatial_encoder = DenseGCNEncoder(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout
        )
        
        # 3b.学习图编码器（条件创建）
        if use_graph_learning:
            self.learned_graph_encoder = DenseGCNEncoder(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=gnn_num_layers,
                dropout=dropout
            )
        
        # 3c.表达图编码器
        self.expr_conv1 = self._make_sparse_gnn_layer(hidden_dim, hidden_dim)
        self.expr_bn1 = nn.BatchNorm1d(hidden_dim)
        self.expr_conv2 = self._make_sparse_gnn_layer(hidden_dim, hidden_dim)
        self.expr_bn2 = nn.BatchNorm1d(hidden_dim)
        self.expr_conv3 = self._make_sparse_gnn_layer(hidden_dim, hidden_dim)
        self.expr_bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 🔥 4.修复：条件融合门控
        if use_graph_learning:
            # 三路融合
            print("  - 融合策略:  三路（空间 + 学习图 + 表达）")
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
                nn.Softmax(dim=-1)
            )
        else:
            # 双路融合
            print("  - 融合策略: 双路（空间 + 表达）")
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )
        
        # 5.投影到Latent
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 6.原型学习模块
        if use_prototype: 
            self.latent_to_prototype = nn.Sequential(
                nn.Linear(latent_dim, prototype_dim),
                nn.ReLU(),
                nn.Linear(prototype_dim, prototype_dim)
            )
            
            self.prototypes = LearnablePrototypes(
                n_prototypes=n_domains,
                prototype_dim=prototype_dim,
                temperature=prototype_temperature,
                momentum=prototype_momentum
            )
            
            self.prototype_loss_fn = PrototypeLoss(
                temperature=prototype_temperature
            )
        
        # 7.解码器
        self._build_decoder(latent_dim, n_output, hidden_dim, dropout, reconstruction_loss)
        
        # 8.辅助损失
        if use_graph_learning:
            self.graph_sparsity_loss_fn = GraphSparsityLoss(loss_type='l1')
        
        # 9.Buffers
        self.register_buffer("_spatial_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_spatial_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_expr_edge_index", torch.empty((2, 0), dtype=torch.long))
        self.register_buffer("_expr_edge_weight", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_X_pca", torch.empty((0, n_input), dtype=torch.float32))
        self.register_buffer("_X_raw", torch.empty((0, n_output), dtype=torch.float32))
    
    def _make_sparse_gnn_layer(self, in_dim, out_dim):
        """创建稀疏GNN层"""
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim)
        )
        return GINEConv(mlp, train_eps=True, edge_dim=1)
    
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
    
    def attach_dual_graphs(self, spatial_edge_index, spatial_edge_weight, 
                          expr_edge_index, expr_edge_weight):
        """附加图数据"""
        self.register_buffer("_spatial_edge_index", spatial_edge_index)
        self.register_buffer("_spatial_edge_weight", spatial_edge_weight)
        self.register_buffer("_expr_edge_index", expr_edge_index)
        self.register_buffer("_expr_edge_weight", expr_edge_weight)
        return self
    
    def attach_full_X(self, X_pca, X_raw):
        """附加特征数据"""
        self.register_buffer("_X_pca", X_pca)
        self.register_buffer("_X_raw", X_raw)
        return self
    
    def forward_all(self):
        """🔥 修复：完整前向传播"""
        if self._X_pca.numel() == 0:
            raise RuntimeError("需要先调用 attach_full_X")
        
        X_pca = self._X_pca
        N = X_pca.size(0)
        device = X_pca.device
        
        library_size = self._X_raw.sum(dim=1, keepdim=True)
        
        info = {}
        
        # 1.输入投影
        h = self.input_proj(X_pca)
        
        # 2.学习图结构（条件执行）
        if self.use_graph_learning:
            A_learned, graph_stats = self.graph_learner(
                X_pca,
                self._spatial_edge_index,
                self._spatial_edge_weight
            )
            info.update(graph_stats)
        else:
            A_learned = None
        
        # 3.构建空间图的稠密邻接矩阵
        A_spatial = torch.zeros((N, N), device=device)
        edge_index = self._spatial_edge_index
        edge_weight = self._spatial_edge_weight
        A_spatial[edge_index[0], edge_index[1]] = edge_weight
        
        # 4.三流编码
        # 4a.空间图编码
        h_spatial = self.spatial_encoder(h, A_spatial)
        
        # 4b.学习图编码（条件执行）
        if self.use_graph_learning:
            h_learned = self.learned_graph_encoder(h, A_learned)
        
        # 4c.表达图编码
        h_expr = h
        h_expr = self.expr_conv1(h_expr, self._expr_edge_index, 
                                 edge_attr=self._expr_edge_weight.unsqueeze(-1))
        h_expr = self.expr_bn1(h_expr)
        h_expr = F.relu(h_expr) + h
        
        h_expr = self.expr_conv2(h_expr, self._expr_edge_index,
                                 edge_attr=self._expr_edge_weight.unsqueeze(-1))
        h_expr = self.expr_bn2(h_expr)
        h_expr = F.relu(h_expr) + h_expr
        
        h_expr = self.expr_conv3(h_expr, self._expr_edge_index,
                                 edge_attr=self._expr_edge_weight.unsqueeze(-1))
        h_expr = self.expr_bn3(h_expr)
        h_expr = F.relu(h_expr) + h_expr
        
        # 🔥 5.修复：条件融合
        if self.use_graph_learning:
            # 三路融合
            h_concat = torch.cat([h_spatial, h_learned, h_expr], dim=-1)
            gates = self.fusion_gate(h_concat)
            
            h_fused = (
                gates[: , 0:1] * h_spatial +
                gates[:, 1:2] * h_learned +
                gates[:, 2:3] * h_expr
            )
            
            info['gate_spatial'] = gates[: , 0].mean().item()
            info['gate_learned'] = gates[:, 1].mean().item()
            info['gate_expr'] = gates[:, 2].mean().item()
        else:
            # 🔥 双路融合
            h_concat = torch.cat([h_spatial, h_expr], dim=-1)
            gates = self.fusion_gate(h_concat)
            
            h_fused = (
                gates[:, 0:1] * h_spatial +
                gates[: , 1:2] * h_expr
            )
            
            info['gate_spatial'] = gates[:, 0].mean().item()
            info['gate_expr'] = gates[:, 1].mean().item()
        
        # 6.投影到Latent
        z_latent = self.to_latent(h_fused)
        
        # 7.解码
        decoder_outputs = self._decode(z_latent, library_size)
        
        return z_latent, decoder_outputs, A_learned, info
    
    def _decode(self, z, library_size):
        """解码器前向传播"""
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
        """计算重构损失"""
        px_rate = decoder_outputs['px_rate']
        eps = 1e-8
        px_rate = torch.clamp(px_rate, min=eps)
        
        if self.reconstruction_loss == 'nb':
            px_r = decoder_outputs['px_r']
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        elif self.reconstruction_loss == 'zinb':
            px_r = decoder_outputs['px_r']
            px_dropout = decoder_outputs['px_dropout']
            dist = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        elif self.reconstruction_loss == 'poisson':
            dist = Poisson(rate=px_rate)
            recon_loss = -dist.log_prob(x_raw).sum(dim=-1)
        else:
            raise ValueError(f"不支持的重构损失:  {self.reconstruction_loss}")
        
        return recon_loss
    
    def compute_prototype_loss(self, z_latent):
        """计算原型损失"""
        if not self.use_prototype:
            return torch.tensor(0.0), {}, None
        
        z_proto = self.latent_to_prototype(z_latent)
        Q, labels = self.prototypes(z_proto, update=True)
        prototypes = self.prototypes.prototypes
        loss, loss_dict = self.prototype_loss_fn(z_proto, prototypes, Q, labels)
        
        return loss, loss_dict, Q
    
    def compute_graph_sparsity_loss(self, A_learned):
        """计算图稀疏性损失"""
        if not self.use_graph_learning or A_learned is None:
            return torch.tensor(0.0)
        
        return self.graph_sparsity_loss_fn(A_learned)
    
    @torch.no_grad()
    def get_latent_representation(self):
        """获取latent表示"""
        self.eval()
        z_latent, _, _, _ = self.forward_all()
        return z_latent.cpu().numpy()