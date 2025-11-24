"""双图构建工具：空间图 + 表达图"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse


def build_spatial_graph(adata, k=6, spatial_key='spatial'):
    """
    构建空间邻域图（基于物理坐标）
    
    Returns:
        edge_index: [2, E] 长整型
        edge_weight: [E] 浮点型（基于距离的 RBF 核）
    """
    coords = adata.obsm[spatial_key]
    
    # k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    edge_list = []
    edge_weights = []
    
    for i in range(len(coords)):
        neighbors = indices[i, 1:]  # 排除自己
        dists = distances[i, 1:]
        
        # RBF 权重
        sigma = dists.std() if dists.std() > 0 else 1.0
        weights = np.exp(-dists**2 / (2 * sigma**2))
        
        for j, w in zip(neighbors, weights):
            edge_list.extend([[i, j], [j, i]])  # 无向图
            edge_weights.extend([w, w])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    return edge_index, edge_weight


def build_expression_graph(adata, k=10, hvg_only=True, n_hvg=2000):
    """
    构建表达相似图（基于基因表达谱）
    
    Args:
        hvg_only: 是否只用高变基因计算相似度
        n_hvg: 高变基因数量
    
    Returns:
        edge_index: [2, E]
        edge_weight: [E]（余弦相似度）
    """
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    
    # 只用 HVG 计算相似度
    if hvg_only:
        if 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable'].values
        else:
            # 简单选择方差最大的基因
            variances = np.var(X, axis=0)
            hvg_indices = np.argsort(variances)[-n_hvg:]
            hvg_mask = np.zeros(X.shape[1], dtype=bool)
            hvg_mask[hvg_indices] = True
        X = X[:, hvg_mask]
    
    # 计算余弦相似度矩阵
    n_spots = X.shape[0]
    edge_list = []
    edge_weights = []
    
    # 批量计算相似度
    sim_matrix = cosine_similarity(X)
    
    for i in range(n_spots):
        # 找到相似度 top-k 的邻居（排除自己）
        sims = sim_matrix[i]
        top_k_idx = np.argsort(sims)[-(k+1):-1]  # 排除自己
        top_k_sims = sims[top_k_idx]
        
        # 只保留相似度 > 阈值的边
        threshold = 0.3
        valid_mask = top_k_sims > threshold
        
        for j, sim in zip(top_k_idx[valid_mask], top_k_sims[valid_mask]):
            if i < j:  # 避免重复边
                edge_list.extend([[i, j], [j, i]])
                edge_weights.extend([sim, sim])
    
    if len(edge_list) == 0:
        # 如果没有边，返回空图
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    return edge_index, edge_weight


def build_dual_graphs(adata, k_spatial=6, k_expr=10, spatial_key='spatial'):
    """
    一键构建双图
    
    Returns:
        dict: {
            'spatial_edge_index': [2, E_s],
            'spatial_edge_weight': [E_s],
            'expr_edge_index': [2, E_e],
            'expr_edge_weight': [E_e]
        }
    """
    spatial_ei, spatial_ew = build_spatial_graph(adata, k=k_spatial, spatial_key=spatial_key)
    expr_ei, expr_ew = build_expression_graph(adata, k=k_expr)
    
    print(f"✅ 空间图构建完成：{spatial_ei.shape[1]} 条边")
    print(f"✅ 表达图构建完成：{expr_ei.shape[1]} 条边")
    
    return {
        'spatial_edge_index': spatial_ei,
        'spatial_edge_weight': spatial_ew,
        'expr_edge_index': expr_ei,
        'expr_edge_weight': expr_ew
    }