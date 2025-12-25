"""图增强模块：基于表达特征生成增强图"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAugmentor(nn.Module):
    """
    图增强器：基于节点特征生成增强图
    
    核心思想：
    - 用Metric Network学习节点相似度
    - 基于相似度构建增强图（TopK或阈值）
    - 增强图捕获长距离的功能连接
    """
    
    def __init__(
        self,
        n_input: int,
        hidden_dim: int = 128,
        k_augment: int = 20,  # TopK中的K
        augment_mode: str = 'topk',  # 'topk' or 'threshold'
        threshold:  float = 0.5,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.k_augment = k_augment
        self.augment_mode = augment_mode
        self.threshold = threshold
        
        print(f"🎨 初始化图增强器:")
        print(f"  - 输入维度: {n_input}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - 增强模式:   {augment_mode}")
        if augment_mode == 'topk':
            print(f"  - TopK:  {k_augment}")
        else: 
            print(f"  - 阈值: {threshold}")
        
        # Metric Network
        layers = []
        layers.append(nn.Linear(n_input, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 4))
        
        self.metric_net = nn.Sequential(*layers)
    
    def compute_similarity_matrix(self, x: torch.Tensor):
        """
        计算节点相似度矩阵
        
        Args:
            x:  [N, n_input]
        
        Returns:
            S: [N, N] 余弦相似度矩阵
        """
        h = self.metric_net(x)
        h_norm = F.normalize(h, p=2, dim=-1)
        S = torch.mm(h_norm, h_norm.t())
        return S
    
    def generate_augmented_graph_topk(self, similarity_matrix: torch.Tensor, k: int):
        """
        TopK模式：保留每个节点最相似的K个邻居
        
        Args:
            similarity_matrix:  [N, N]
            k: TopK中的K
        
        Returns: 
            edge_index: [2, E]
            edge_weight: [E,]
        """
        N = similarity_matrix.size(0)
        device = similarity_matrix.device
        
        # 对每个节点，找到TopK最相似的邻居
        topk_values, topk_indices = torch.topk(similarity_matrix, k=k+1, dim=1)  # +1因为包含自己
        
        # 构建边列表（排除自环）
        edge_list = []
        edge_weights = []
        
        for i in range(N):
            for j in range(1, k+1):  # 跳过自己（索引0）
                neighbor = topk_indices[i, j].item()
                weight = topk_values[i, j].item()
                
                # 添加双向边
                edge_list.append([i, neighbor])
                edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=device)
        
        # 去重（因为是双向的，可能有重复）
        edge_index, edge_weight = self._remove_duplicate_edges(edge_index, edge_weight)
        
        return edge_index, edge_weight
    
    def generate_augmented_graph_threshold(self, similarity_matrix: torch.Tensor, threshold: float):
        """
        阈值模式：保留相似度>阈值的边
        
        Args:
            similarity_matrix:  [N, N]
            threshold: 阈值
        
        Returns:
            edge_index: [2, E]
            edge_weight: [E,]
        """
        N = similarity_matrix.size(0)
        device = similarity_matrix.device
        
        # 找到所有 > threshold 的边
        mask = similarity_matrix > threshold
        mask.fill_diagonal_(False)  # 排除自环
        
        edge_index = mask.nonzero().t()  # [2, E]
        
        src, dst = edge_index[0], edge_index[1]
        edge_weight = similarity_matrix[src, dst]
        
        return edge_index, edge_weight
    
    def _remove_duplicate_edges(self, edge_index, edge_weight):
        """去除重复边（保留权重较大的）"""
        # 将边转换为唯一标识
        N = edge_index.max().item() + 1
        edge_ids = edge_index[0] * N + edge_index[1]
        
        # 去重
        unique_ids, inverse_indices = torch.unique(edge_ids, return_inverse=True)
        
        # 对重复边，保留权重最大的
        new_edge_weight = torch.zeros(unique_ids.size(0), device=edge_weight.device)
        new_edge_weight.scatter_reduce_(
            0, inverse_indices, edge_weight, 
            reduce='amax', include_self=False
        )
        
        # 重建edge_index
        new_edge_index = torch.stack([
            unique_ids // N,
            unique_ids % N
        ])
        
        return new_edge_index, new_edge_weight
    
    def forward(self, x: torch.Tensor):
        """
        生成增强图
        
        Args:
            x: [N, n_input] 节点特征
        
        Returns: 
            augmented_edge_index: [2, E_aug]
            augmented_edge_weight: [E_aug,]
            similarity_matrix: [N, N] (用于可视化)
        """
        # 计算相似度
        similarity_matrix = self.compute_similarity_matrix(x)
        
        # 生成增强图
        if self.augment_mode == 'topk':
            edge_index, edge_weight = self.generate_augmented_graph_topk(
                similarity_matrix, self.k_augment
            )
        elif self.augment_mode == 'threshold':
            edge_index, edge_weight = self.generate_augmented_graph_threshold(
                similarity_matrix, self.threshold
            )
        else:
            raise ValueError(f"不支持的增强模式:  {self.augment_mode}")
        
        return edge_index, edge_weight, similarity_matrix


class GraphContrastiveLoss(nn.Module):
    """
    图对比学习损失
    
    目标：让空间图和增强图学到的表示接近
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        contrast_mode: str = 'infonce'  # 'infonce' or 'cosine'
    ):
        super().__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        
        print(f"🔥 图对比学习损失:")
        print(f"  - Temperature: {temperature}")
        print(f"  - 模式:   {contrast_mode}")
    
    def forward(self, z_spatial: torch.Tensor, z_augmented: torch.Tensor):
        """
        计算图对比损失
        
        Args:
            z_spatial:  [N, D] 空间图学到的表示
            z_augmented: [N, D] 增强图学到的表示
        
        Returns:
            loss: 标量
        """
        N = z_spatial.size(0)
        device = z_spatial.device
        
        if self.contrast_mode == 'infonce':
            # InfoNCE损失（类似SimCLR）
            # 归一化
            z1 = F.normalize(z_spatial, p=2, dim=-1)
            z2 = F.normalize(z_augmented, p=2, dim=-1)
            
            # 计算相似度矩阵
            sim_matrix = torch.mm(z1, z2.t()) / self.temperature  # [N, N]
            
            # 对角线是正样本，其他是负样本
            labels = torch.arange(N, device=device)
            
            # 双向对比
            loss_1to2 = F.cross_entropy(sim_matrix, labels)
            loss_2to1 = F.cross_entropy(sim_matrix.t(), labels)
            
            loss = (loss_1to2 + loss_2to1) / 2
        
        elif self.contrast_mode == 'cosine':
            # 简单余弦相似度损失
            z1 = F.normalize(z_spatial, p=2, dim=-1)
            z2 = F.normalize(z_augmented, p=2, dim=-1)
            
            # 最大化对应节点的相似度
            loss = -torch.mean(torch.sum(z1 * z2, dim=-1))
        
        else: 
            raise ValueError(f"不支持的对比模式:  {self.contrast_mode}")
        
        return loss