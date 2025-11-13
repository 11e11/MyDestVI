import copy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Union
import pandas as pd

class ContrastivePairs:
    """
    计算空间转录组数据中每个spot的正负对
    正对：空间邻近且基因表达相似的spot
    负对：基因表达最不相似的spot
    """
    
    def __init__(self, neg_rate: float = 0.05, pos_rate: float = 0.05, neighbor_k: int = 10,pos_max_per_spot: int | None = None,  
        neg_max_per_spot: int | None = None):
        """
        初始化对比学习样本对生成器
        
        Parameters:
        -----------
        neg_rate : float
            负样本比例，用于确定表达相似度的低阈值
        pos_rate : float  
            正样本比例，用于确定邻域内表达相似度的高阈值
        neighbor_k : int
            考虑的空间邻域大小（最近邻数量）
        """
        self.neg_rate = neg_rate
        self.pos_rate = pos_rate
        self.neighbor_k = neighbor_k
        self.pos_max_per_spot = pos_max_per_spot
        self.neg_max_per_spot = neg_max_per_spot
    
    def compute_pairs(self, adata, spatial_key: str = 'spatial', 
                    pos_key: str = 'pos_indices', neg_key: str = 'neg_indices'):
        """
        计算每个spot的正负对索引并存入anndata
        
        Parameters:
        -----------
        adata : AnnData
            包含基因表达和空间位置信息的anndata对象
        spatial_key : str
            adata.obsm中空间坐标的键名
        pos_key : str
            存储正对索引的键名
        neg_key : str
            存储负对索引的键名
            
        Returns:
        --------
        adata : AnnData
            添加了正负对索引信息的anndata对象
        """
        # 获取空间位置和基因表达信息
        if spatial_key not in adata.obsm:
            raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")
            
        spot_pos_all = adata.obsm[spatial_key]
        spot_info_all = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        
        # 计算余弦相似度矩阵
        expr = copy.deepcopy(spot_info_all)
        similarity_matrix = cosine_similarity(expr)
        
        # 计算空间距离矩阵
        distance = np.sum((spot_pos_all - spot_pos_all[:, np.newaxis]) ** 2, axis=2)
        
        # 找到每个spot的空间邻居（包括自己）
        indices_1_all = np.argpartition(distance, self.neighbor_k + 1)[:, :self.neighbor_k + 1]
        
        # 计算邻域内的表达相似度分布，确定正样本阈值
        near_similarity_all = similarity_matrix[np.arange(len(spot_pos_all))[:, np.newaxis], indices_1_all].flatten()
        sorted_lst = np.sort(near_similarity_all)[::-1]
        # 去掉对角线元素（每个spot与自己的相似度为1）
        sorted_lst = sorted_lst[len(spot_pos_all):]
        top_thres = sorted_lst[int(len(sorted_lst) * self.pos_rate)]
        
        # 计算全局表达相似度分布，确定负样本阈值
        all_similarity = similarity_matrix.flatten()
        low_thres = np.sort(all_similarity)[int(len(all_similarity) * self.neg_rate)]
        
        # 计算每个spot的负样本数量上限
        sigma = len(spot_pos_all) * self.neg_rate
        neg_num = min(int(sigma), 200)  # 全局上限200，这是每个anchor的最大负样本数量

        # 为每个spot计算正负对索引
        pos_indices_list = []
        neg_indices_list = []
        pos_statistics = []
        neg_statistics = []
            
     
        # 记录使用备选策略的spot数量
        fallback_spots_count = 0
            
        for i in range(len(spot_pos_all)):
            # 正对：先尝试原始策略 - 空间邻近且表达相似
            indices_2 = np.where(similarity_matrix[i, :] > top_thres)[0]
            pos_indices = np.intersect1d(indices_1_all[i], indices_2)
            # 移除自己
            pos_indices = pos_indices[pos_indices != i]
            
            # # 组合式选择：如果没有找到正样本，使用相对相似度选择
            # if len(pos_indices) == 0:
            #     fallback_spots_count += 1
            #     # 获取空间邻居（排除自己）
            #     neighbors = indices_1_all[i]
            #     neighbors = neighbors[neighbors != i]
                
            #     if len(neighbors) > 0:
            #         # 计算与邻居的相似度
            #         neighbor_sims = similarity_matrix[i, neighbors]
            #         # 选择相对最相似的前min(3, len(neighbors))个
            #         top_k = min(3, len(neighbors))
            #         top_indices = np.argsort(-neighbor_sims)[:top_k]  # 降序排序取前top_k个
            #         pos_indices = neighbors[top_indices]
            # 若有候选，则按相似度降序排序再截断
            if len(pos_indices) > 0:
                sims = similarity_matrix[i, pos_indices]
                order = np.argsort(-sims)
                pos_indices = pos_indices[order]
                if self.pos_max_per_spot is not None:
                    pos_indices = pos_indices[: self.pos_max_per_spot]
            else:
                # 备选策略：从邻居里取最相似的若干
                neighbors = indices_1_all[i]
                neighbors = neighbors[neighbors != i]
                if len(neighbors) > 0:
                    neighbor_sims = similarity_matrix[i, neighbors]
                    order = np.argsort(-neighbor_sims)
                    top_k = 3 if self.pos_max_per_spot is None else min(3, self.pos_max_per_spot)
                    pos_indices = neighbors[order[:top_k]]
                    fallback_spots_count += 1
            
            pos_indices_list.append(pos_indices)
            pos_statistics.append(len(pos_indices))
            
            # 修正负样本逻辑
            # # 1. 找到所有相似度低于low_thres的候选负样本
            # negative_candidates = np.where(similarity_matrix[i, :] <= low_thres)[0]
            # negative_candidates = negative_candidates[negative_candidates != i]  # 移除自己
            
            # if len(negative_candidates) > 0:
            #     # 2. 在候选中按相似度排序，取最不相似的前neg_num个
            #     candidate_similarities = similarity_matrix[i, negative_candidates]
            #     sorted_indices = np.argsort(candidate_similarities)  # 从小到大排序（最不相似在前）
                
            #     # 取前neg_num个最不相似的
            #     actual_neg_num = min(len(negative_candidates), neg_num)
            #     selected_neg_indices = negative_candidates[sorted_indices[:actual_neg_num]]
                
            #     neg_indices_list.append(selected_neg_indices)
            #     neg_statistics.append(len(selected_neg_indices))
            # else:
            #     # 3. 如果没有满足条件的候选，取全局最不相似的min(neg_num, 1)个
            #     all_similarities = similarity_matrix[i, :]
            #     # 排除自己
            #     valid_indices = np.arange(len(all_similarities))
            #     valid_indices = valid_indices[valid_indices != i]
            #     valid_similarities = all_similarities[valid_indices]
                
            #     # 取最不相似的几个
            #     sorted_indices = np.argsort(valid_similarities)
            #     fallback_num = min(len(valid_indices), neg_num, 1)  # 至少取1个，最多neg_num个
            #     selected_neg_indices = valid_indices[sorted_indices[:fallback_num]]
                
            #     neg_indices_list.append(selected_neg_indices)
            #     neg_statistics.append(len(selected_neg_indices))
            
            
            # === 负样本 ===
            negative_candidates = np.where(similarity_matrix[i, :] <= low_thres)[0]
            negative_candidates = negative_candidates[negative_candidates != i]
            if len(negative_candidates) > 0:
                cand_sims = similarity_matrix[i, negative_candidates]
                order = np.argsort(cand_sims)  # 从小到大（越不相似越靠前）
                base_max = min(len(negative_candidates), neg_num)
                if self.neg_max_per_spot is not None:
                    base_max = min(base_max, self.neg_max_per_spot)
                selected_neg_indices = negative_candidates[order[:base_max]]
            else:
                all_sims = similarity_matrix[i, :]
                valid_idx = np.arange(len(all_sims))
                valid_idx = valid_idx[valid_idx != i]
                valid_sims = all_sims[valid_idx]
                order = np.argsort(valid_sims)
                fallback_num = min(len(valid_idx), neg_num, 1)
                if self.neg_max_per_spot is not None:
                    fallback_num = min(fallback_num, self.neg_max_per_spot)
                selected_neg_indices = valid_idx[order[:fallback_num]]

            neg_indices_list.append(selected_neg_indices)
            neg_statistics.append(len(selected_neg_indices))
        
        # 存储到anndata的obsm中
        # 由于长度不一致，需要特殊处理
        max_pos_len = max(len(x) for x in pos_indices_list) if pos_indices_list else 0
        max_neg_len = max(len(x) for x in neg_indices_list) if neg_indices_list else 0
        
        # 用-1填充，表示无效索引
        pos_matrix = np.full((len(spot_pos_all), max_pos_len), -1, dtype=int)
        neg_matrix = np.full((len(spot_pos_all), max_neg_len), -1, dtype=int)
        
        for i, (pos_idx, neg_idx) in enumerate(zip(pos_indices_list, neg_indices_list)):
            if len(pos_idx) > 0:
                pos_matrix[i, :len(pos_idx)] = pos_idx
            if len(neg_idx) > 0:
                neg_matrix[i, :len(neg_idx)] = neg_idx
        
        adata.obsm[pos_key] = pos_matrix
        adata.obsm[neg_key] = neg_matrix
        
        # 也可以在obs中存储每个spot的正负对数量，方便训练时使用
        adata.obs[f'{pos_key}_count'] = pos_statistics
        adata.obs[f'{neg_key}_count'] = neg_statistics
        
        # 存储一些统计信息到uns中，方便调试
        adata.uns['contrastive_pairs_params'] = {
            'neg_rate': self.neg_rate,
            'pos_rate': self.pos_rate, 
            'neighbor_k': self.neighbor_k,
            'top_thres': top_thres,
            'low_thres': low_thres,
            'avg_pos_pairs': np.mean([len(x) for x in pos_indices_list]),
            'avg_neg_pairs': np.mean([len(x) for x in neg_indices_list])
        }
        
        adata.uns['contrastive_pairs_params'].update({
        'fallback_spots_count': fallback_spots_count,
        'fallback_spots_ratio': fallback_spots_count / len(spot_pos_all),
        })

        print(f"Contrastive pairs computed:")
        print(f"  Positive threshold: {top_thres:.4f}")
        print(f"  Negative threshold: {low_thres:.4f}")  
        print(f"  Average positive pairs per spot: {adata.uns['contrastive_pairs_params']['avg_pos_pairs']:.2f}")
        print(f"  Average negative pairs per spot: {adata.uns['contrastive_pairs_params']['avg_neg_pairs']:.2f}")
        print(f"  Spots using fallback strategy: {fallback_spots_count}/{len(spot_pos_all)} ({adata.uns['contrastive_pairs_params']['fallback_spots_ratio']:.2%})")
        
        return adata




def make_contrastive_pairs(adata, neg_rate: float = 0.05, pos_rate: float = 0.05, 
                          neighbor_k: int = 10, **kwargs):
    """
    便捷函数：为anndata计算对比学习的正负样本对

    支持额外参数：
      - 传给 ContrastivePairs.__init__：
          pos_max_per_spot: int | None
          neg_max_per_spot: int | None
      - 传给 compute_pairs：
          spatial_key: str
          pos_key: str
          neg_key: str
    """
    # 仅挑 __init__ 支持的键
    init_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {"pos_max_per_spot", "neg_max_per_spot"}
    }
    # 仅挑 compute_pairs 支持的键
    compute_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {"spatial_key", "pos_key", "neg_key"}
    }

    cp = ContrastivePairs(
        neg_rate=neg_rate,
        pos_rate=pos_rate,
        neighbor_k=neighbor_k,
        **init_kwargs,
    )
    adata = cp.compute_pairs(adata, **compute_kwargs)

    # 记录上限到 uns（若提供）
    if init_kwargs:
        adata.uns.setdefault("contrastive_pairs_params", {})
        adata.uns["contrastive_pairs_params"].update({
            "pos_max_per_spot": init_kwargs.get("pos_max_per_spot", None),
            "neg_max_per_spot": init_kwargs.get("neg_max_per_spot", None),
        })
    return adata



import numpy as np

def retrofit_padding_for_contrastive_pairs(adata, pad_value: int = -1):
    """
    将 obsm['pos_indices']/['neg_indices']（ragged 列表）转换为二维 padding 数组：
      - obsm['pos_indices_padded'], obsm['neg_indices_padded']  (int64, [n_obs, Kmax])
      - obs['pos_indices_count'], obs['neg_indices_count']      (int64, [n_obs])
    """
    if "pos_indices" not in adata.obsm or "neg_indices" not in adata.obsm:
        raise ValueError("pos_indices/neg_indices 不存在，先运行 make_contrastive_pairs。")

    def _make_padded(ragged):
        ragged = list(ragged)
        n = len(ragged)
        maxk = max(len(x) if hasattr(x, "__len__") else 1 for x in ragged) if n > 0 else 0
        padded = np.full((n, maxk), pad_value, dtype=np.int64)
        counts = np.zeros(n, dtype=np.int64)
        for i, row in enumerate(ragged):
            if hasattr(row, "__len__"):
                arr = np.asarray(row, dtype=np.int64)
            else:
                arr = np.asarray([int(row)], dtype=np.int64)
            L = int(arr.size)
            if L > 0:
                padded[i, :L] = arr
            counts[i] = L
        return padded, counts

    pos_padded, pos_counts = _make_padded(adata.obsm["pos_indices"])
    neg_padded, neg_counts = _make_padded(adata.obsm["neg_indices"])

    adata.obsm["pos_indices_padded"] = pos_padded
    adata.obsm["neg_indices_padded"] = neg_padded
    adata.obs["pos_indices_count"] = pos_counts
    adata.obs["neg_indices_count"] = neg_counts
    return adata