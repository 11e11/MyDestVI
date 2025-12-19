"""多尺度对比学习采样器 - 自适应平滑版本"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse


class MultiScaleContrastiveSampler:
    """
    🔥 改进版：移除局部正/负样本，使用自适应平滑
    
    核心改变：
    1.不再采样"局部正/负样本"
    2.只保留"空间邻居"信息（用于自适应平滑）
    3.保留"全局正/负样本"（用于长距离对比）
    """
    
    def __init__(
        self,
        adata,
        spatial_key='spatial',
        expr_layer='normalized_log',
        k_spatial=6,
        # 🔥 移除局部正/负样本参数
        # local_pos=3,
        # local_neg=3,
        # 全局对比参数
        global_pos=3,
        global_neg1=5,
        global_neg2=5,
        global_sim_percentile=90
    ):
        """
        Args:
            adata: AnnData 对象
            spatial_key:  空间坐标的 key
            expr_layer: 表达数据的 layer
            k_spatial: 空间邻居数量（用于自适应平滑）
            global_pos: 全局正样本数量
            global_neg1: 全局最不相似负样本数量
            global_neg2: 全局随机负样本数量
            global_sim_percentile: 全局相似度阈值（百分位数）
        """
        self.adata = adata
        self.spatial_key = spatial_key
        self.expr_layer = expr_layer
        self.k_spatial = k_spatial
        self.global_pos = global_pos
        self.global_neg1 = global_neg1
        self.global_neg2 = global_neg2
        self.global_sim_percentile = global_sim_percentile
        
        self.n_spots = adata.n_obs
        
        # 获取表达数据
        if expr_layer in adata.layers:
            self.expr_data = adata.layers[expr_layer]
        else: 
            self.expr_data = adata.X
        
        if issparse(self.expr_data):
            self.expr_data = self.expr_data.toarray()
        
        print("🎯 初始化多尺度对比学习采样器（自适应平滑版本）")
        print("="*60)
        
        # 🔥 1.构建空间邻居（用于自适应平滑）
        print("📍 构建空间邻居...")
        self.spatial_neighbors = self._build_spatial_neighbors()
        
        # 🔥 2.计算表达相似度矩阵（用于全局对比）
        print("🧬 计算表达相似度矩阵...")
        self.expr_sim_matrix = self._compute_expression_similarity()
        
        # 🔥 3.预计算全局对比样本
        print("⚡ 预计算全局对比样本...")
        self.global_pos_samples = []
        self.global_neg_samples = []
        self._sample_all_global()
        
        print("✅ 采样器初始化完成！")
        print("="*60)
    
    def _build_spatial_neighbors(self):
        """
        构建空间邻居索引
        
        Returns: 
            dict: {spot_idx: [neighbor_indices]}
        """
        coords = self.adata.obsm[self.spatial_key]
        nbrs = NearestNeighbors(n_neighbors=self.k_spatial + 1, metric='euclidean').fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        spatial_neighbors = {}
        for i in range(self.n_spots):
            # 排除自己（第一个）
            spatial_neighbors[i] = indices[i, 1:].tolist()
        
        print(f"  平均空间邻居数: {self.k_spatial}")
        
        return spatial_neighbors
    
    def _compute_expression_similarity(self):
        """
        计算表达相似度矩阵（用于全局对比采样）
        
        使用高变基因计算余弦相似度
        """
        # 使用高变基因
        if 'highly_variable' in self.adata.var.columns:
            hvg_mask = self.adata.var['highly_variable'].values
            X_hvg = self.expr_data[: , hvg_mask]
            n_hvg = hvg_mask.sum()
            print(f"  使用 {n_hvg} 个高变基因计算相似度")
        else:
            X_hvg = self.expr_data
            print(f"  使用所有 {X_hvg.shape[1]} 个基因计算相似度")
        
        # 计算余弦相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(X_hvg)
        
        print(f"  相似度矩阵形状: {sim_matrix.shape}")
        
        # 计算全局阈值
        self.global_sim_threshold = np.percentile(sim_matrix, self.global_sim_percentile)
        print(f"  全局相似度阈值 (P{self.global_sim_percentile}): {self.global_sim_threshold:.3f}")
        
        return sim_matrix
    
    def _sample_global_positives(self, anchor_idx):
        """
        采样全局正样本：表达高度相似但空间距离远
        
        策略：
        1.筛选表达相似度 > 阈值的 spot
        2.排除空间邻居
        3.随机选择 global_pos 个
        """
        # 获取所有相似的 spot
        similarities = self.expr_sim_matrix[anchor_idx]
        high_sim_mask = similarities > self.global_sim_threshold
        candidates = np.where(high_sim_mask)[0]
        
        # 排除自己
        candidates = candidates[candidates != anchor_idx]
        
        # 排除空间邻居
        spatial_neighbors_set = set(self.spatial_neighbors[anchor_idx])
        candidates = np.array([c for c in candidates if c not in spatial_neighbors_set])
        
        if len(candidates) == 0:
            return np.array([], dtype=np.int64)
        
        # 随机选择
        n_select = min(self.global_pos, len(candidates))
        selected = np.random.choice(candidates, n_select, replace=False)
        
        return selected
    
    def _sample_global_negatives(self, anchor_idx):
        """
        采样全局负样本：表达不相似且空间距离远
        
        策略：
        1.最不相似的 global_neg1 个
        2.随机采样 global_neg2 个（排除邻居和高相似）
        """
        similarities = self.expr_sim_matrix[anchor_idx]
        
        # 1.最不相似的负样本
        # 排除自己
        valid_indices = np.arange(self.n_spots)
        valid_indices = valid_indices[valid_indices != anchor_idx]
        valid_sims = similarities[valid_indices]
        
        # 选择最小的 global_neg1 个
        n_neg1 = min(self.global_neg1, len(valid_indices))
        neg1_local_idx = np.argpartition(valid_sims, n_neg1)[:n_neg1]
        neg1_indices = valid_indices[neg1_local_idx]
        
        # 2.随机负样本
        # 排除自己、邻居、高相似
        spatial_neighbors_set = set(self.spatial_neighbors[anchor_idx])
        high_sim_mask = similarities > self.global_sim_threshold
        
        candidates = []
        for i in range(self.n_spots):
            if i != anchor_idx and i not in spatial_neighbors_set and not high_sim_mask[i]: 
                candidates.append(i)
        
        if len(candidates) > 0:
            n_neg2 = min(self.global_neg2, len(candidates))
            neg2_indices = np.random.choice(candidates, n_neg2, replace=False)
        else:
            neg2_indices = np.array([], dtype=np.int64)
        
        # 合并
        all_neg = np.concatenate([neg1_indices, neg2_indices])
        return all_neg
    
    def _sample_all_global(self):
        """
        预计算所有 spot 的全局对比样本
        """
        for i in range(self.n_spots):
            if i % 500 == 0:
                print(f"  已处理 {i}/{self.n_spots} 个 spot...")
            
            # 全局正样本
            global_pos = self._sample_global_positives(i)
            self.global_pos_samples.append(global_pos)
            
            # 全局负样本
            global_neg = self._sample_global_negatives(i)
            self.global_neg_samples.append(global_neg)
    
    def get_samples(self):
        """
        获取采样结果
        
        Returns: 
            dict: {
                'spatial_neighbors': dict,  # 🔥 空间邻居（用于自适应平滑）
                'global_pos':  list,         # 全局正样本
                'global_neg': list          # 全局负样本
            }
        """
        return {
            'spatial_neighbors':  self.spatial_neighbors,
            'global_pos': self.global_pos_samples,
            'global_neg': self.global_neg_samples
        }
    
    def get_statistics(self):
        """打印采样统计"""
        print("\n📊 采样统计:")
        
        # 空间邻居
        n_spatial = np.mean([len(neighbors) for neighbors in self.spatial_neighbors.values()])
        print(f"  空间邻居平均数: {n_spatial:.2f}")
        
        # 全局正样本
        n_global_pos = np.mean([len(pos) for pos in self.global_pos_samples])
        print(f"  全局正样本平均数: {n_global_pos:.2f}")
        
        # 全局负样本
        n_global_neg = np.mean([len(neg) for neg in self.global_neg_samples])
        print(f"  全局负样本平均数: {n_global_neg:.2f}")
        
        print()