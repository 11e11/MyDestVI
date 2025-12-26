"""双图构建工具：空间图 + 表达图"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected 


# def build_spatial_graph(adata, k=6, spatial_key='spatial'):
#     """
#     构建空间邻域图（基于物理坐标）
    
#     Returns:
#         edge_index: [2, E] 长整型
#         edge_weight: [E] 浮点型（基于距离的 RBF 核）
#     """
#     coords = adata.obsm[spatial_key]
    
#     # k-NN
#     nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
#     distances, indices = nbrs.kneighbors(coords)
    
#     edge_list = []
#     edge_weights = []
    
#     for i in range(len(coords)):
#         neighbors = indices[i, 1:]  # 排除自己
#         dists = distances[i, 1:]
        
#         # RBF 权重
#         sigma = dists.std() if dists.std() > 0 else 1.0
#         weights = np.exp(-dists**2 / (2 * sigma**2))
        
#         for j, w in zip(neighbors, weights):
#             edge_list.extend([[i, j], [j, i]])  # 无向图
#             edge_weights.extend([w, w])
    
#     edge_index = torch.tensor(edge_list, dtype=torch.long).t()
#     edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
#     return edge_index, edge_weight
def build_spatial_graph_only(adata, k=6, spatial_key='spatial'):
    """
    只构建空间KNN图
    
    Returns:
        edge_index: [2, E]
        edge_weight: [E,]
    """
    from sklearn.neighbors import NearestNeighbors
    import torch
    
    spatial_coords = adata. obsm[spatial_key]
    
    # KNN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)
    
    # 构建边
    src = []
    dst = []
    weights = []
    
    for i in range(len(spatial_coords)):
        for j_idx in range(1, k+1):  # 跳过自己
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            
            src.append(i)
            dst.append(j)
            weights.append(np.exp(-d**2 / (2 * d. mean()**2)))  # Gaussian kernel
    
    edge_index = torch.LongTensor([src, dst])
    edge_weight = torch.FloatTensor(weights)
    
    return edge_index, edge_weight

def build_spatial_graph(adata, k=6, spatial_key='spatial', use_global_sigma=True):
    """
    构建空间邻域图 - 向量化 + 可选全局/局部 sigma
    
    Args:
        use_global_sigma: 
            - True: 使用全局 sigma（所有边权重一致的尺度）
            - False:  使用局部 sigma（考虑每个节点的邻域密度）
    """
    coords = adata.obsm[spatial_key]
    n_spots = coords.shape[0]
    
    # 1. k-NN 搜索
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # 排除自己
    distances = distances[:, 1:]  # [N, k]
    indices = indices[: , 1:]      # [N, k]
    
    # 2. 计算 RBF 权重
    if use_global_sigma: 
        # 🔥 方案 1：全局 sigma（推荐用于空间图）
        sigma = np.std(distances)  # 标量
        if sigma == 0:
            sigma = 1.0
        weights_matrix = np.exp(-distances**2 / (2 * sigma**2))
        print(f"  使用全局 sigma: {sigma:.3f}")
    else:
        # 🔥 方案 2：局部 sigma（每个节点独立）
        sigmas = np.std(distances, axis=1)  # [N,]
        sigmas[sigmas == 0] = 1.0
        sigmas_expanded = sigmas[:, np.newaxis]  # [N, 1]
        weights_matrix = np.exp(-distances**2 / (2 * sigmas_expanded**2))
        print(f"  使用局部 sigma: mean={sigmas.mean():.3f}, std={sigmas.std():.3f}")
    
    # 3. 构建边列表（向量化）
    source_nodes = np.repeat(np.arange(n_spots), k)
    target_nodes = indices. flatten()
    edge_weights = weights_matrix.flatten()
    
    # 4. 转为 Tensor
    edge_index = torch.tensor(np.stack([source_nodes, target_nodes]), dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    # 5. 转无向图（去重 + 平均权重）
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
    
    print(f"✅ Spatial Graph:  {edge_index.shape[1]} edges (k={k})")
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



# def build_expression_graph(adata, k=10, use_rep='X_pca', n_pcs=50):
#     """
#     🔥 改进：使用 PCA 表示构建表达相似图
    
#     Args:
#         adata: AnnData 对象
#         k: 邻居数量
#         use_rep: 使用的表示空间
#             - 'X_pca':  使用 PCA 降维后的数据（推荐）
#             - 'normalized_log':  使用归一化后的 log 数据（原方法）
#         n_pcs: 使用的 PCA 主成分数量（仅当 use_rep='X_pca' 时有效）
    
#     Returns:
#         edge_index: [2, E]
#         edge_weight: [E]（余弦相似度）
#     """
#     n_spots = adata.n_obs
    
#     # 🔥 根据 use_rep 选择数据
#     if use_rep == 'X_pca': 
#         if 'X_pca' not in adata.obsm:
#             raise ValueError("adata. obsm 中没有 'X_pca'，请先运行 sc.tl.pca()")
        
#         X = adata.obsm['X_pca'][: , :n_pcs]
#         print(f"📊 Building Expression Graph using PCA representation ({n_pcs} PCs)")
    
#     elif use_rep == 'normalized_log':
#         if 'normalized_log' in adata.layers:
#             X = adata.layers['normalized_log']
#         else:
#             X = adata.X
        
#         # 确保 X 是 dense 格式
#         if hasattr(X, "toarray"):
#             X = X. toarray()
        
#         # HVG 筛选
#         if 'highly_variable' in adata.var.columns:
#             hvg_mask = adata.var['highly_variable']. values
#             X = X[: , hvg_mask]
#             print(f"📊 Building Expression Graph using HVG ({hvg_mask.sum()} genes)")
#         else:
#             print(f"📊 Building Expression Graph using all genes ({X.shape[1]} genes)")
    
#     else:
#         raise ValueError(f"不支持的 use_rep:  {use_rep}")
    
#     # 快速计算 KNN
#     print("🚀 Using fast NearestNeighbors search...")
#     nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='auto')
#     nbrs.fit(X)
    
#     # distances:  [N, k+1], indices: [N, k+1]
#     distances, indices = nbrs.kneighbors(X)
    
#     # 转换距离为相似度 (Cosine Sim = 1 - Cosine Dist)
#     similarities = 1 - distances
    
#     # 构建边列表 (Vectorized)
#     # 排除第一列（自己）
#     source_nodes = np.repeat(np.arange(n_spots), k)
#     target_nodes = indices[: , 1:].flatten()
#     weights = similarities[: , 1:].flatten()
    
#     # 🔥 阈值过滤（可选）
#     threshold = 0.0  # 🔥 改为 0，保留所有边（PCA 数据已经去噪）
#     if threshold > 0:
#         mask = weights > threshold
#         sources = source_nodes[mask]
#         targets = target_nodes[mask]
#         edge_weights = weights[mask]
#     else:
#         sources = source_nodes
#         targets = target_nodes
#         edge_weights = weights
    
#     if len(sources) == 0:
#         print("⚠️ No edges found with current threshold!")
#         return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    
#     # 构建 Tensor
#     edge_index = torch.tensor(np.stack([sources, targets]), dtype=torch.long)
#     edge_weight = torch.tensor(edge_weights, dtype=torch. float32)
    
#     # 转无向图 (取平均权重)
#     edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
    
#     print(f"✅ Expression Graph: {edge_index.shape[1]} edges (k={k})")
#     return edge_index, edge_weight

def build_expression_graph(adata, k=10, use_rep='X_pca', n_pcs=50):
    """
    🔥 优化版：针对 PCA 数据使用 Euclidean + RBF，针对原始数据使用 Cosine
    """
    import numpy as np
    import torch
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.utils import to_undirected

    n_spots = adata.n_obs
    
    # 1. 准备数据和确定度量方式
    if use_rep == 'X_pca': 
        if 'X_pca' not in adata.obsm:
            raise ValueError("adata.obsm 中没有 'X_pca'")
        
        # 取前 n_pcs
        X = adata.obsm['X_pca'][:, :n_pcs]
        
        # 🔥 PCA 空间：使用欧氏距离 + RBF 核
        metric = 'euclidean'
        use_rbf = True
        print(f"📊 Building Graph: PCA (shape: {X.shape}) | Metric: Euclidean | Weight: RBF")
        
    elif use_rep == 'normalized_log':
        # ... (获取数据的逻辑保持不变) ...
        if 'normalized_log' in adata.layers:
            X = adata.layers['normalized_log']
        else:
            X = adata.X
        if hasattr(X, "toarray"): X = X.toarray()
        
        # HVG 筛选逻辑保持不变...
        if 'highly_variable' in adata.var.columns:
            X = X[:, adata.var['highly_variable'].values]
            
        # 🔥 原始高维空间：使用余弦距离 + 线性权重
        metric = 'cosine'
        use_rbf = False
        print(f"📊 Building Graph: Gene Expr (shape: {X.shape}) | Metric: Cosine | Weight: Linear")
        
    else:
        raise ValueError(f"不支持的 use_rep: {use_rep}")
    
    # 2. 计算 KNN
    print(f"🚀 Computing KNN (k={k}, metric={metric})...")
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 排除自己
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # 3. 计算权重 (核心修改)
    if use_rbf:
        # 🔥 RBF 核权重计算 (w = exp(-d^2 / 2σ^2))
        # 自适应 sigma: 取平均距离，或者每个点第 k 个邻居的距离
        mean_dist = np.mean(distances)
        sigma = mean_dist if mean_dist > 0 else 1.0
        
        # 计算权重
        weights = np.exp(- (distances**2) / (2 * sigma**2))
        
        print(f"  - RBF Sigma: {sigma:.4f}")
        print(f"  - Weights: Min={weights.min():.4f}, Mean={weights.mean():.4f}, Max={weights.max():.4f}")
        
    else:
        # 🔥 原始的线性权重 (Cosine)
        # Cosine distance 范围 0~2，相似度 = 1 - dist
        weights = 1 - distances
        # 截断负值（虽然理论上 KNN 找出来的 dist 不会超过 1，但保险起见）
        weights = np.clip(weights, a_min=0, a_max=1)

    # 4. 构建 Tensor
    source_nodes = np.repeat(np.arange(n_spots), k)
    target_nodes = indices.flatten()
    edge_weights = weights.flatten()
    
    edge_index = torch.tensor(np.stack([source_nodes, target_nodes]), dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    # 5. 转无向图
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
    
    print(f"✅ Expression Graph Built: {edge_index.shape[1]} edges")
    return edge_index, edge_weight


def build_dual_graphs(adata, k_spatial=6, k_expr=10, spatial_key='spatial', 
                     use_pca_for_expr=True, n_pcs=50):
    """
    一键构建双图
    
    Args:
        adata: AnnData 对象
        k_spatial: 空间图邻居数
        k_expr: 表达图邻居数
        spatial_key: 空间坐标在 obsm 中的 key
        use_pca_for_expr: 🔥 是否使用 PCA 构建表达图（推荐 True）
        n_pcs: 🔥 使用的 PCA 主成分数量
    
    Returns:
        dict: {
            'spatial_edge_index': [2, E_s],
            'spatial_edge_weight': [E_s],
            'expr_edge_index': [2, E_e],
            'expr_edge_weight': [E_e]
        }
    """
    # 构建空间图
    spatial_ei, spatial_ew = build_spatial_graph(
        adata, 
        k=k_spatial, 
        spatial_key=spatial_key,
        use_global_sigma=True
    )
    
    # 🔥 构建表达图（使用 PCA 或 HVG）
    if use_pca_for_expr:
        expr_ei, expr_ew = build_expression_graph(
            adata, 
            k=k_expr, 
            use_rep='X_pca',
            n_pcs=n_pcs
        )
    else:
        expr_ei, expr_ew = build_expression_graph(
            adata, 
            k=k_expr, 
            use_rep='normalized_log'
        )
    
    print(f"✅ 空间图构建完成：{spatial_ei.shape[1]} 条边")
    print(f"✅ 表达图构建完成：{expr_ei. shape[1]} 条边")
    
    return {
        'spatial_edge_index': spatial_ei,
        'spatial_edge_weight': spatial_ew,
        'expr_edge_index': expr_ei,
        'expr_edge_weight': expr_ew
    }