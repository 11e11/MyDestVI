"""多尺度对比学习采样器 - 全图版本"""
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics. pairwise import cosine_similarity
import scipy.sparse as sp


class MultiScaleContrastiveSampler:
    """
    多尺度对比学习采样策略（全图预计算版本）
    
    局部策略：
    - 正样本：一跳邻居内相似的 local_pos 个 Spot
    - 负样本：两跳和一跳邻居内不相似的 local_neg 个 Spot
    
    全局策略：
    - 正样本：不邻接的最相似的 global_pos 个 Spot
    - 负样本：全局最不相似的 global_neg1 个 + 随机的 global_neg2 个
    """
    
    def __init__(
        self,
        adata,
        spatial_key='spatial',
        expr_layer='normalized_log',
        k_spatial=6,
        local_pos=3,
        local_neg=5,
        global_pos=3,
        global_neg1=5,
        global_neg2=5,
        local_sim_percentile=60,  # 相似度阈值（百分位数）
    ):
        """
        Args:
            local_pos: 局部正样本数量
            local_neg: 局部负样本数量
            global_pos: 全局正样本数量
            global_neg1: 全局最不相似负样本数量
            global_neg2: 全局随机负样本数量
            local_sim_percentile: 局部相似度阈值（百分位数），高于此值为正样本
        """
        self.adata = adata
        self.n_spots = adata.n_obs
        self.k_spatial = k_spatial
        self.local_pos = local_pos
        self.local_neg = local_neg
        self.global_pos = global_pos
        self. global_neg1 = global_neg1
        self.global_neg2 = global_neg2
        
        print("\n" + "="*60)
        print("🎯 初始化多尺度对比学习采样器")
        print("="*60)
        
        # 1. 构建空间邻域
        print("📍 构建空间邻域...")
        coords = adata. obsm[spatial_key]
        self.spatial_nbrs = NearestNeighbors(n_neighbors=k_spatial+1). fit(coords)
        self. spatial_dists, self.spatial_indices = self. spatial_nbrs.kneighbors(coords)
        
        # 构建邻接矩阵（用于快速查询）
        self.adj_matrix = sp.lil_matrix((self.n_spots, self.n_spots), dtype=bool)
        for i in range(self.n_spots):
            neighbors = self.spatial_indices[i, 1:]  # 排除自己
            self.adj_matrix[i, neighbors] = True
        self.adj_matrix = self.adj_matrix.tocsr()
        
        # 2. 构建两跳邻域
        print("📍 构建两跳邻域...")
        self._build_two_hop_neighbors()
        
        # 3.  计算表达相似度矩阵
        print("🧬 计算表达相似度矩阵...")
        if expr_layer in adata.layers:
            X = adata.layers[expr_layer]
        else:
            X = adata.X
        
        if hasattr(X, 'toarray'):
            X = X. toarray()
        
        # 只用HVG计算相似度
        if 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable']. values
            X = X[:, hvg_mask]
            print(f"  使用 {hvg_mask.sum()} 个高变基因计算相似度")
        
        self.expr_sim_matrix = cosine_similarity(X)
        print(f"  相似度矩阵形状: {self.expr_sim_matrix.shape}")
        
        # 计算相似度阈值
        self.local_sim_threshold = np.percentile(self.expr_sim_matrix, local_sim_percentile)
        print(f"  局部相似度阈值 (P{local_sim_percentile}): {self.local_sim_threshold:.3f}")
        
        # 4. 预计算所有样本（全图）
        print("⚡ 预计算所有对比学习样本...")
        self._precompute_all_samples()
        
        print("✅ 采样器初始化完成！")
        print("="*60 + "\n")
    
    def _build_two_hop_neighbors(self):
        """构建两跳邻域（包含一跳邻居）"""
        self.one_hop_neighbors = {}
        self.two_hop_union = {}  # 一跳 + 二跳的并集
        
        for i in range(self.n_spots):
            one_hop = set(self.spatial_indices[i, 1:]. tolist())
            self.one_hop_neighbors[i] = one_hop
            
            # 收集所有二跳邻居
            two_hop = set()
            for j in one_hop:
                two_hop.update(self.spatial_indices[j, 1:].tolist())
            
            # 两跳并集 = 一跳 + 二跳（不包括自己）
            union = (one_hop | two_hop) - {i}
            self. two_hop_union[i] = list(union)
    
    def _precompute_all_samples(self):
        """预计算全图的对比学习样本"""
        self.local_pos_samples = {}
        self.local_neg_samples = {}
        self.global_pos_samples = {}
        self. global_neg_samples = {}
        
        for i in range(self.n_spots):
            # 局部正样本
            self.local_pos_samples[i] = self._sample_local_positives(i)
            
            # 局部负样本
            self.local_neg_samples[i] = self._sample_local_negatives(i)
            
            # 全局正样本
            self.global_pos_samples[i] = self._sample_global_positives(i)
            
            # 全局负样本
            self.global_neg_samples[i] = self._sample_global_negatives(i)
            
            if i % 500 == 0:
                print(f"  已处理 {i}/{self.n_spots} 个 spot...")
    
    def _sample_local_positives(self, anchor_idx):
        """
        局部正样本：一跳邻居内相似的 local_pos 个 Spot
        """
        one_hop = list(self.one_hop_neighbors[anchor_idx])
        
        if len(one_hop) == 0:
            return np.array([], dtype=np.int64)
        
        # 计算与一跳邻居的相似度
        similarities = self.expr_sim_matrix[anchor_idx, one_hop]
        
        # 筛选相似度高于阈值的
        high_sim_mask = similarities > self.local_sim_threshold
        candidates = np.array(one_hop)[high_sim_mask]
        
        if len(candidates) == 0:
            # 如果没有高相似度的，选择最相似的几个
            n_select = min(self.local_pos, len(one_hop))
            top_indices = np.argsort(similarities)[-n_select:]
            return np.array(one_hop)[top_indices]
        
        # 从候选中随机选择 local_pos 个
        n_select = min(self.local_pos, len(candidates))
        selected = np.random.choice(candidates, n_select, replace=False)
        
        return selected
    
    def _sample_local_negatives(self, anchor_idx):
        """
        局部负样本：两跳和一跳邻居内不相似的 local_neg 个 Spot
        """
        two_hop_union = self.two_hop_union. get(anchor_idx, [])
        
        if len(two_hop_union) == 0:
            return np.array([], dtype=np.int64)
        
        # 计算相似度
        similarities = self. expr_sim_matrix[anchor_idx, two_hop_union]
        
        # 选择相似度最低的 local_neg 个
        n_select = min(self.local_neg, len(two_hop_union))
        neg_indices = np.argsort(similarities)[:n_select]
        
        return np.array(two_hop_union)[neg_indices]
    
    # def _sample_local_negatives(self, anchor_idx):
    #     """简化：不使用局部负样本"""
    #     return np.array([], dtype=np.int64)
    
    def _sample_global_positives(self, anchor_idx):
        """
        全局正样本：不邻接的最相似的 global_pos 个 Spot
        """
        # 获取非邻居 mask
        neighbors = self.adj_matrix[anchor_idx]. toarray().ravel(). astype(bool)
        neighbors[anchor_idx] = True  # 排除自己
        non_neighbors_mask = ~neighbors
        
        non_neighbor_indices = np.where(non_neighbors_mask)[0]
        
        if len(non_neighbor_indices) == 0:
            return np.array([], dtype=np.int64)
        
        # 计算与非邻居的相似度
        similarities = self.expr_sim_matrix[anchor_idx, non_neighbor_indices]
        
        # 选择最相似的 global_pos 个
        n_select = min(self.global_pos, len(non_neighbor_indices))
        top_indices = np.argsort(similarities)[-n_select:]
        
        return non_neighbor_indices[top_indices]
    
    def _sample_global_negatives(self, anchor_idx):
        """
        全局负样本：
        - global_neg1 个最不相似的 Spot
        - global_neg2 个随机选择的 Spot
        """
        # 排除自己
        all_indices = np.arange(self. n_spots)
        mask = all_indices != anchor_idx
        candidates = all_indices[mask]
        
        similarities = self.expr_sim_matrix[anchor_idx, candidates]
        
        # 1. 最不相似的 global_neg1 个
        n_hard = min(self.global_neg1, len(candidates))
        hard_neg_indices = np.argsort(similarities)[:n_hard]
        hard_negatives = candidates[hard_neg_indices]
        
        # 2. 随机选择 global_neg2 个（排除已选的硬负样本）
        remaining = np.setdiff1d(candidates, hard_negatives)
        n_random = min(self.global_neg2, len(remaining))
        
        if n_random > 0:
            random_negatives = np.random.choice(remaining, n_random, replace=False)
            all_negatives = np.concatenate([hard_negatives, random_negatives])
        else:
            all_negatives = hard_negatives
        
        return all_negatives
    
    def get_samples(self, indices=None):
        """
        获取预计算的样本
        
        Args:
            indices: 要获取样本的索引，None 表示全部
        
        Returns:
            dict: {
                'local_pos': list of arrays,
                'local_neg': list of arrays,
                'global_pos': list of arrays,
                'global_neg': list of arrays
            }
        """
        if indices is None:
            indices = range(self.n_spots)
        
        return {
            'local_pos': [self.local_pos_samples[i] for i in indices],
            'local_neg': [self.local_neg_samples[i] for i in indices],
            'global_pos': [self. global_pos_samples[i] for i in indices],
            'global_neg': [self.global_neg_samples[i] for i in indices]
        }
    
    def get_statistics(self):
        """获取采样统计信息"""
        stats = {
            'local_pos_mean': np.mean([len(v) for v in self.local_pos_samples.values()]),
            'local_neg_mean': np.mean([len(v) for v in self. local_neg_samples.values()]),
            'global_pos_mean': np.mean([len(v) for v in self.global_pos_samples.values()]),
            'global_neg_mean': np.mean([len(v) for v in self.global_neg_samples.values()]),
        }
        
        print("\n📊 采样统计:")
        print(f"  局部正样本平均数: {stats['local_pos_mean']:.2f}")
        print(f"  局部负样本平均数: {stats['local_neg_mean']:.2f}")
        print(f"  全局正样本平均数: {stats['global_pos_mean']:.2f}")
        print(f"  全局负样本平均数: {stats['global_neg_mean']:.2f}")
        
        return stats