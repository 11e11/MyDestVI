"""双图构建工具：空间图 + 表达图"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected 


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


# def build_expression_graph(adata, k=10, hvg_only=True, n_hvg=2000, layer='normalized_log'):
#     """
#     构建表达相似图（基于基因表达谱）
    
#     Args:
#         hvg_only: 是否只用高变基因计算相似度
#         n_hvg: 高变基因数量
    
#     Returns:
#         edge_index: [2, E]
#         edge_weight: [E]（余弦相似度）
#     """
#     # # 🔥 优先尝试读取 layer
#     if layer in adata.layers:
#         X = adata.layers[layer]
#         print(f"📊 Building Expression Graph using layer: {layer}")
#     else:
#         X = adata.X
#         print(f"⚠️ Layer {layer} not found, using .X for graph (Check if this is Normalized!)")
#     # X = adata.X.toarray() if issparse(adata.X) else adata.X
    
#     # 只用 HVG 计算相似度
#     if hvg_only:
#         if 'highly_variable' in adata.var.columns:
#             hvg_mask = adata.var['highly_variable'].values
#         else:
#             # 简单选择方差最大的基因
#             variances = np.var(X, axis=0)
#             hvg_indices = np.argsort(variances)[-n_hvg:]
#             hvg_mask = np.zeros(X.shape[1], dtype=bool)
#             hvg_mask[hvg_indices] = True
#         X = X[:, hvg_mask]
    
#     # 计算余弦相似度矩阵
#     n_spots = X.shape[0]
#     edge_list = []
#     edge_weights = []
    
#     # 批量计算相似度
#     sim_matrix = cosine_similarity(X)
    
#     for i in range(n_spots):
#         # 找到相似度 top-k 的邻居（排除自己）
#         sims = sim_matrix[i]
#         top_k_idx = np.argsort(sims)[-(k+1):-1]  # 排除自己
#         top_k_sims = sims[top_k_idx]
        
#         # 只保留相似度 > 阈值的边
#         threshold = 0.3
#         valid_mask = top_k_sims > threshold
        
#         for j, sim in zip(top_k_idx[valid_mask], top_k_sims[valid_mask]):
#             if i < j:  # 避免重复边
#                 edge_list.extend([[i, j], [j, i]])
#                 edge_weights.extend([sim, sim])
    
#     if len(edge_list) == 0:
#         # 如果没有边，返回空图
#         return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    
#     edge_index = torch.tensor(edge_list, dtype=torch.long).t()
#     edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
#     return edge_index, edge_weight



def build_expression_graph(adata, k=10, hvg_only=True, n_hvg=2000, layer='normalized_log'):
    """
    构建表达相似图（基于基因表达谱）
    """
    # 1. 获取数据
    if layer in adata.layers:
        X = adata.layers[layer]
        print(f"📊 Building Expression Graph using layer: {layer}")
    else:
        X = adata.X
        print(f"⚠️ Layer {layer} not found, using .X for graph (Check if this is Normalized!)")
    
    # 确保 X 是 dense 格式，避免 sklearn 或 numpy 在稀疏矩阵上报错
    # 如果内存不够，这一步可以不做，但 sklearn 的 brute force 最好用 dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    # 2. HVG 筛选
    if hvg_only:
        if 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable'].values
        else:
            print("⚠️ 'highly_variable' not found, selecting top variance genes...")
            variances = np.var(X, axis=0)
            hvg_indices = np.argsort(variances)[-n_hvg:]
            hvg_mask = np.zeros(X.shape[1], dtype=bool)
            hvg_mask[hvg_indices] = True
        X = X[:, hvg_mask]
    
    n_spots = X.shape[0]
    
    # 3. 快速计算 KNN
    print("🚀 Using fast NearestNeighbors search...")
    # algorithm='auto' 通常比 'brute' 更智能，会自动选择 KDTree 或 BallTree（如果适用）
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='auto')
    nbrs.fit(X)
    
    # distances: [N, k+1], indices: [N, k+1]
    distances, indices = nbrs.kneighbors(X)
    
    # 转换距离为相似度 (Cosine Sim = 1 - Cosine Dist)
    similarities = 1 - distances
    
    # 4. 构建边列表 (Vectorized)
    # 排除第一列（因为第一列通常是自己，dist=0, sim=1）
    source_nodes = np.repeat(np.arange(n_spots), k)
    target_nodes = indices[:, 1:].flatten()
    weights = similarities[:, 1:].flatten()
    
    # 5. 阈值过滤
    threshold = 0.3
    mask = weights > threshold
    
    # 过滤后的数据
    sources = source_nodes[mask]
    targets = target_nodes[mask]
    edge_weights = weights[mask]
    
    if len(sources) == 0:
        print("⚠️ No edges found with current threshold!")
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

    # 6. 构建 Tensor (修复 Numpy 警告)
    # 直接用 [arr, arr] 转 tensor 会很慢且报警告，推荐 np.stack
    edge_index = torch.tensor(np.stack([sources, targets]), dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    # 🔥 关键改进：KNN 图通常是有向的 (A在B的Top10里，但B不一定在A的Top10里)
    # 图神经网络通常希望图是无向的 (Undirected)。
    # 如果你需要双向连接，请取消下面这行的注释：
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
    
    print(f"✅ Graph built: {edge_index.shape[1]} edges")
    return edge_index, edge_weight


def build_dual_graphs(adata, k_spatial=6, k_expr=10, spatial_key='spatial', encoder_layer='normalized_log'):
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
    expr_ei, expr_ew = build_expression_graph(adata, k=k_expr, layer=encoder_layer)
    
    print(f"✅ 空间图构建完成：{spatial_ei.shape[1]} 条边")
    print(f"✅ 表达图构建完成：{expr_ei.shape[1]} 条边")
    
    return {
        'spatial_edge_index': spatial_ei,
        'spatial_edge_weight': spatial_ew,
        'expr_edge_index': expr_ei,
        'expr_edge_weight': expr_ew
    }
def build_dual_graphs_with_pseudo(
    adata_st,
    X_pseudo: np.ndarray = None,
    k_spatial: int = 6,
    k_expr: int = 10,
    expr_threshold: float = 0.5,
    spatial_key: str = "spatial",
    allow_pseudo_in_graph: bool = False,  # 🔥 新增：控制伪点是否进图
):
    """
    构建双图
    
    🔥 新策略：伪点默认不进图（allow_pseudo_in_graph=False）
    """
    from scipy.sparse import issparse
    
    N_real = adata_st.n_obs
    
    # ===== 准备表达特征 =====
    if 'normalized_log' in adata_st.  layers:
        X_real = adata_st.layers['normalized_log']
    else:
        X_real = adata_st.X
    
    if issparse(X_real):
        X_real = X_real. toarray()
    X_real = np.asarray(X_real, dtype=np.float32)
    
    if X_pseudo is not None and allow_pseudo_in_graph:
        # 🔥 旧方案：伪点加入图（会导致震荡）
        N_pseudo = len(X_pseudo)
        library_pseudo = X_pseudo.sum(axis=1, keepdims=True)
        X_pseudo_norm = np.log1p(X_pseudo / (library_pseudo + 1e-6) * 10000)
        X_all_norm = np.vstack([X_real, X_pseudo_norm])
        N_total = N_real + N_pseudo
        print(f"⚠️ 伪点加入图模式（可能导致震荡）：{N_real} 真实 + {N_pseudo} 伪点")
    else:
        # 🔥 新方案：伪点不进图
        X_all_norm = X_real
        N_total = N_real
        N_pseudo = 0 if X_pseudo is None else len(X_pseudo)
        if N_pseudo > 0:
            print(f"✅ 伪点不进图模式（推荐）：{N_real} 真实点用于图，{N_pseudo} 伪点仅用于监督")
    
    # ===== 空间图（只连真实点）=====
    coords = adata_st.obsm[spatial_key]
    nbrs_spatial = NearestNeighbors(n_neighbors=k_spatial + 1, metric='euclidean').fit(coords)
    distances_s, indices_s = nbrs_spatial.  kneighbors(coords)
    
    src_s, dst_s, weights_s = [], [], []
    for i in range(N_real):
        for j in range(1, k_spatial + 1):
            neighbor = indices_s[i, j]
            dist = distances_s[i, j]
            src_s.append(i)
            dst_s.append(neighbor)
            sigma = distances_s[:, 1:]. std() if distances_s[:, 1:].std() > 0 else 1.0
            weights_s.append(np.exp(-dist**2 / (2 * sigma**2)))
    
    spatial_edge_index = torch.tensor([src_s, dst_s], dtype=torch.long)
    spatial_edge_weight = torch.tensor(weights_s, dtype=torch.float32)
    
    # ===== 表达图（只连真实点）=====
    nbrs_expr = NearestNeighbors(n_neighbors=k_expr + 1, metric='cosine').fit(X_all_norm)
    distances_e, indices_e = nbrs_expr. kneighbors(X_all_norm)
    
    src_e, dst_e, weights_e = [], [], []
    for i in range(N_total):
        for j in range(1, k_expr + 1):
            neighbor = indices_e[i, j]
            dist = distances_e[i, j]
            if dist < expr_threshold:
                src_e.append(i)
                dst_e.append(neighbor)
                weights_e.append(np.exp(-dist * 2))
    
    expr_edge_index = torch.tensor([src_e, dst_e], dtype=torch.long)
    expr_edge_weight = torch.tensor(weights_e, dtype=torch.float32)
    
    print(f"✅ 空间图：{len(src_s)} 条边（只连真实点）")
    print(f"✅ 表达图：{len(src_e)} 条边（只连真实点）")
    
    return {
        'spatial_edge_index': spatial_edge_index,
        'spatial_edge_weight': spatial_edge_weight,
        'expr_edge_index': expr_edge_index,
        'expr_edge_weight': expr_edge_weight,
        'N_real': N_real,
        'N_pseudo': N_pseudo,
        'N_total': N_real,  # 🔥 图只包含真实点
    }