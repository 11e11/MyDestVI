"""双图构建工具：空间图 + 表达图"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse


def build_spatial_graph(adata, k=6, spatial_key='spatial'):
    coords = adata.obsm[spatial_key].copy()  # [N,2]
    # 坐标归一化
    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-6)

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    edge_list = []
    edge_attrs = []   # 多维 edge_attr
    
    for i in range(len(coords)):
        neighbors = indices[i, 1:] 
        dists = distances[i, 1:]
        
        sigma = dists.std() if dists.std() > 1e-6 else 1.0
        rbf_w = np.exp(-dists**2 / (2 * sigma**2))
        
        for j, dist_ij, w_ij in zip(neighbors, dists, rbf_w):
            dx = coords[j,0] - coords[i,0]
            dy = coords[j,1] - coords[i,1]
            
            # 构建4维 edge_attr
            attr = [dist_ij, w_ij, dx, dy]
            
            # 无向图：i->j, j->i
            edge_list.append([i, j])
            edge_list.append([j, i])
            edge_attrs.append(attr)
            edge_attrs.append(attr)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    return edge_index, edge_attr



def build_expression_graph(adata, k=10, hvg_only=True, n_hvg=2000):
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    
    # 选择 HVG
    if hvg_only:
        if 'highly_variable' in adata.var:
            hvg_mask = adata.var['highly_variable'].values
        else:
            variances = np.var(X, axis=0)
            hvg_idx = np.argsort(variances)[-n_hvg:]
            hvg_mask = np.zeros(X.shape[1], dtype=bool)
            hvg_mask[hvg_idx] = True
        X = X[:, hvg_mask]
    
    sim_matrix = cosine_similarity(X)
    
    # Pearson
    corr = np.corrcoef(X)
    
    edge_list = []
    edge_attrs = []
    
    for i in range(X.shape[0]):
        sims = sim_matrix[i]
        top_idx = np.argsort(sims)[-(k+1):-1]  # 排除自己
        
        for rank, j in enumerate(top_idx):
            sim_ij = sims[j]
            corr_ij = corr[i, j]
            
            # 3维 edge_attr
            attr = [sim_ij, 1 - corr_ij, rank / k]  # rank归一化
            
            if i < j:
                edge_list.extend([[i, j], [j, i]])
                edge_attrs.extend([attr, attr])
    
    if len(edge_list) == 0:
        return torch.empty((2,0)), torch.empty((0,))
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    return edge_index, edge_attr



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
    spatial_ei, spatial_attr = build_spatial_graph(adata, k=k_spatial, spatial_key=spatial_key)
    expr_ei, expr_attr = build_expression_graph(adata, k=k_expr)
    
    print(f"✅ 空间图构建完成：{spatial_ei.shape[1]} 条边")
    print(f"✅ 表达图构建完成：{expr_ei.shape[1]} 条边")
    
    return {
        'spatial_edge_index': spatial_ei,
        'spatial_edge_attr': spatial_attr,
        'expr_edge_index': expr_ei,
        'expr_edge_attr': expr_attr
    }