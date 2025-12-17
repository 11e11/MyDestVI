"""双图构建工具：空间图 + 表达图"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric. utils import to_undirected 


def build_spatial_graph(adata, k=6, spatial_key='spatial'):
    """
    构建空间邻域图（基于物理坐标）
    使用 RBF 核
    """
    coords = adata.obsm[spatial_key]
    n_spots = coords.shape[0]
    
    # k-NN 搜索（欧氏距离）
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # 排除自己
    distances = distances[:, 1:]  # [N, k]
    indices = indices[:, 1:]      # [N, k]
    
    # 🔥 RBF 权重（全局 sigma）
    mean_dist = np.mean(distances)
    sigma = mean_dist if mean_dist > 0 else 1.0
    weights_matrix = np.exp(- (distances**2) / (2 * sigma**2))
    
    print(f"  Spatial Graph:  Euclidean + RBF (σ={sigma:.3f})")
    
    # 构建边列表（向量化）
    source_nodes = np.repeat(np.arange(n_spots), k)
    target_nodes = indices. flatten()
    edge_weights = weights_matrix.flatten()
    
    # 转为 Tensor
    edge_index = torch.tensor(np.stack([source_nodes, target_nodes]), dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch. float32)
    
    # 转无向图（去重 + 平均权重）
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
    
    print(f"✅ Spatial Graph:   {edge_index.shape[1]} edges (k={k})")
    return edge_index, edge_weight


def build_expression_graph(adata, k=10, use_rep='X_pca', n_pcs=50):
    """
    🔥 改进版：使用 PCA + Euclidean + RBF 构建表达图
    
    Args:
        adata: AnnData 对象
        k:  邻居数量
        use_rep: 使用的表示空间
            - 'X_pca':   使用 PCA 降维后的数据（推荐，Euclidean）
            - 'normalized_log':   使用归一化后的 log 数据（Cosine）
        n_pcs: 使用的 PCA 主成分数量
    
    Returns:
        edge_index: [2, E]
        edge_weight: [E]
    """
    n_spots = adata.n_obs
    
    # 1. 根据 use_rep 选择数据和度量
    if use_rep == 'X_pca':  
        if 'X_pca' not in adata. obsm:
            raise ValueError("adata.obsm 中没有 'X_pca'，请先运行 sc.tl.pca()")
        
        X = adata.obsm['X_pca'][:, :n_pcs]
        
        # 🔥 关键：PCA 使用欧氏距离
        metric = 'euclidean'
        use_rbf = True
        
        print(f"📊 Building Expression Graph:")
        print(f"  - Data: PCA (shape: {X.shape})")
        print(f"  - Metric:  Euclidean Distance")
        print(f"  - Weight: RBF Kernel")
    
    elif use_rep == 'normalized_log':
        if 'normalized_log' in adata.layers:
            X = adata. layers['normalized_log']
        else:
            X = adata.X
        
        # 确保 X 是 dense 格式
        if hasattr(X, "toarray"):
            X = X.toarray()
        
        # HVG 筛选
        if 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable']. values
            X = X[:, hvg_mask]
            print(f"📊 Building Expression Graph using HVG ({hvg_mask.sum()} genes)")
        
        # 原始基因空间使用余弦相似度
        metric = 'cosine'
        use_rbf = False
        
        print(f"  - Data: Log-normalized ({X.shape[1]} features)")
        print(f"  - Metric: Cosine Distance")
    
    else:
        raise ValueError(f"不支持的 use_rep:   {use_rep}")
    
    # 2. 计算 KNN
    print(f"🚀 Computing KNN (k={k})...")
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')
    nbrs.fit(X)
    
    distances, indices = nbrs.kneighbors(X)
    
    # 3. 计算权重
    # 排除第一列（自己）
    distances = distances[:, 1:]  # [N, k]
    indices = indices[:, 1:]      # [N, k]
    
    if use_rbf:
        # 🔥 RBF 权重（用于 PCA）
        mean_dist = np.mean(distances)
        sigma = mean_dist if mean_dist > 0 else 1.0
        
        # 可选：使用更大的 sigma 让图更平滑
        # sigma = 2 * mean_dist
        
        weights = np.exp(- (distances**2) / (2 * sigma**2))
        print(f"  - RBF σ: {sigma:.4f}")
        print(f"  - Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    else:
        # 线性权重（用于原始基因空间 + Cosine）
        # 注意：cosine 返回的是距离（0-2），需要转为相似度
        weights = 1 - distances
        # 可选：阈值过滤
        # threshold = 0.3
        # weights[weights < threshold] = 0
    
    # 4. 构建边列表 (Vectorized)
    source_nodes = np.repeat(np.arange(n_spots), k)
    target_nodes = indices.flatten()
    edge_weights = weights.flatten()
    
    # 5. 可选：过滤权重过低的边
    if use_rbf:
        # RBF 权重已经自然衰减，一般不需要额外过滤
        threshold = 0.0
    else:
        # Cosine 可能需要阈值
        threshold = 0.0
    
    if threshold > 0:
        mask = edge_weights > threshold
        source_nodes = source_nodes[mask]
        target_nodes = target_nodes[mask]
        edge_weights = edge_weights[mask]
    
    if len(source_nodes) == 0:
        print("⚠️ No edges found!")
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    
    # 6. 构建 Tensor
    edge_index = torch.tensor(np.stack([source_nodes, target_nodes]), dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    # 7. 转无向图 (取平均权重)
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
    
    print(f"✅ Expression Graph:  {edge_index.shape[1]} edges")
    return edge_index, edge_weight


def build_dual_graphs(adata, k_spatial=6, k_expr=10, spatial_key='spatial', 
                     use_pca_for_expr=True, n_pcs=50):
    """
    一键构建双图
    
    Args:
        adata: AnnData 对象（🔥 注意：PCA 必须是标准化后的！）
        k_spatial: 空间图邻居数
        k_expr: 表达图邻居数
        spatial_key: 空间坐标在 obsm 中的 key
        use_pca_for_expr: 是否使用 PCA 构建表达图
        n_pcs: 使用的 PCA 主成分数量
    
    Returns:
        dict: 包含空间图和表达图的边和权重
    """
    print("\n" + "="*60)
    print("📊 构建双图")
    print("="*60)
    
    # 构建空间图
    print("\n🗺️ 构建空间图...")
    spatial_ei, spatial_ew = build_spatial_graph(
        adata, 
        k=k_spatial, 
        spatial_key=spatial_key
    )
    
    # 构建表达图
    print("\n🧬 构建表达图...")
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
    
    print("\n" + "="*60)
    print(f"✅ 双图构建完成")
    print(f"  - 空间图: {spatial_ei.shape[1]} 条边")
    print(f"  - 表达图: {expr_ei.shape[1]} 条边")
    print("="*60 + "\n")
    
    return {
        'spatial_edge_index': spatial_ei,
        'spatial_edge_weight': spatial_ew,
        'expr_edge_index': expr_ei,
        'expr_edge_weight': expr_ew
    }