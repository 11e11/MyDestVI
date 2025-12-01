# """
# 伪点生成模块 - 参考 LETSmix/Harmodeconv
# """
# import numpy as np
# import torch
# from typing import Tuple, Optional


# class PseudoSpotGenerator:
#     """
#     从单细胞数据生成伪 spot（已知细胞类型比例）
#     """
#     def __init__(
#         self,
#         adata_sc,
#         celltype_key: str = "cell_type",
#         n_cells_per_spot: Tuple[int, int] = (5, 20),  # 每个伪spot的细胞数范围
#         prior_alpha: Optional[np.ndarray] = None,      # Dirichlet先验（可选）
#     ):
#         self.adata_sc = adata_sc
#         self.celltype_key = celltype_key
#         self.n_cells_per_spot = n_cells_per_spot
        
#         # 提取细胞类型信息
#         self.celltypes = np.array(adata_sc.obs[celltype_key]. cat.categories)
#         self.n_celltypes = len(self.celltypes)
#         self.celltype_to_idx = {ct: i for i, ct in enumerate(self.celltypes)}
        
#         # 按细胞类型分组索引
#         self.celltype_indices = {}
#         for ct in self.celltypes:
#             self.celltype_indices[ct] = np.where(
#                 adata_sc.obs[celltype_key] == ct
#             )[0]
        
#         # Dirichlet 先验（默认均匀）
#         if prior_alpha is None:
#             self.prior_alpha = np.ones(self.n_celltypes)
#         else:
#             self. prior_alpha = prior_alpha
        
#         print(f"✅ 伪点生成器初始化：{self.n_celltypes} 个细胞类型")
#         for ct in self.celltypes:
#             print(f"   {ct}: {len(self.celltype_indices[ct])} 个细胞")
    
#     def fit_prior_from_real_st(self, adata_st, proportion_key: str = "cell_type_fractions"):
#         """
#         从真实 ST 数据学习细胞类型比例的先验分布
        
#         策略：用真实 ST 的比例分布估计 Dirichlet 的 alpha 参数
#         """
#         if proportion_key not in adata_st.obsm:
#             print(f"⚠ 未找到 {proportion_key}，使用均匀先验")
#             return
        
#         proportions = adata_st.obsm[proportion_key]  # [N, n_celltypes]
        
#         # 估计 Dirichlet 参数（方法：Moment Matching）
#         mean_prop = proportions.mean(axis=0)  # 平均比例
#         var_prop = proportions.var(axis=0)    # 方差
        
#         # Dirichlet 的均值 = alpha_i / sum(alpha)
#         # 方差 = alpha_i * (sum(alpha) - alpha_i) / (sum(alpha)^2 * (sum(alpha) + 1))
#         # 简化估计：alpha_i ≈ mean_i * concentration
#         concentration = 10.0  # 可调参数，越大分布越集中
#         self.prior_alpha = mean_prop * concentration
        
#         print(f"✅ 从 {adata_st.n_obs} 个真实 spots 学习先验分布")
#         print(f"   Alpha: {self.prior_alpha}")
    
#     def generate_pseudo_spots(
#         self,
#         n_pseudo: int = 1000,
#         random_state: int = 42
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         生成伪 spot
        
#         Returns:
#             X_pseudo: [n_pseudo, n_genes] 混合后的表达矩阵（raw counts）
#             proportions_pseudo: [n_pseudo, n_celltypes] 真实比例（监督信号）
#             n_cells_pseudo: [n_pseudo] 每个伪spot包含的细胞数
#         """
#         np.random.seed(random_state)
        
#         X_sc = self.adata_sc.X
#         if hasattr(X_sc, 'toarray'):
#             X_sc = X_sc.toarray()
#         X_sc = np.asarray(X_sc, dtype=np.float32)
        
#         n_genes = X_sc.shape[1]
#         X_pseudo = []
#         proportions_pseudo = []
#         n_cells_pseudo = []
        
#         for i in range(n_pseudo):
#             # 1. 采样细胞数
#             n_cells = np. random.randint(
#                 self.n_cells_per_spot[0],
#                 self.n_cells_per_spot[1] + 1
#             )
            
#             # 2. 从 Dirichlet 采样比例
#             proportions = np.random.dirichlet(self.prior_alpha)
            
#             # 3.  按比例确定每个细胞类型的细胞数
#             celltype_counts = np.random.multinomial(n_cells, proportions)
            
#             # 4. 从每个细胞类型采样细胞
#             selected_cells = []
#             for ct_idx, ct in enumerate(self.celltypes):
#                 n_ct = celltype_counts[ct_idx]
#                 if n_ct > 0:
#                     ct_indices = self.celltype_indices[ct]
#                     if len(ct_indices) == 0:
#                         continue
#                     # 有放回采样（允许同一个细胞被多次选中）
#                     sampled = np.random.choice(ct_indices, size=n_ct, replace=True)
#                     selected_cells. extend(sampled)
            
#             if len(selected_cells) == 0:
#                 # 如果没有采样到细胞，跳过
#                 continue
            
#             # 5. 混合表达（求和，模拟 spot 的总表达）
#             x_mixed = X_sc[selected_cells]. sum(axis=0)
            
#             # 6. 计算实际比例（可能与采样比例略有不同，因为取整）
#             actual_proportions = celltype_counts / celltype_counts.sum()
            
#             X_pseudo.append(x_mixed)
#             proportions_pseudo. append(actual_proportions)
#             n_cells_pseudo.append(len(selected_cells))
        
#         X_pseudo = np.array(X_pseudo, dtype=np.float32)
#         proportions_pseudo = np. array(proportions_pseudo, dtype=np.float32)
#         n_cells_pseudo = np. array(n_cells_pseudo, dtype=np.int32)
        
#         print(f"✅ 生成 {len(X_pseudo)} 个伪 spots")
#         print(f"   平均细胞数: {n_cells_pseudo.mean():.1f} ± {n_cells_pseudo. std():.1f}")
#         print(f"   表达量范围: [{X_pseudo.sum(axis=1).min():.0f}, {X_pseudo.sum(axis=1).max():.0f}]")
        
#         return X_pseudo, proportions_pseudo, n_cells_pseudo
# """
# 伪点生成模块 - 改进版（基于真实细胞类型频率）
# """
# import numpy as np
# import torch
# from typing import Tuple, Optional


# class PseudoSpotGenerator:
#     """
#     从单细胞数据生成伪 spot（已知细胞类型比例）
    
#     改进策略：
#     1. 计算 SC 数据中各细胞类型的全局频率
#     2. 每个伪点只包含 3-5 种细胞类型（而非全部）
#     3. 高频细胞类型（如 Acinar）出现概率高，低频细胞类型（如 pDC）出现概率低
#     """
#     def __init__(
#         self,
#         adata_sc,
#         celltype_key: str = "cell_type",
#         n_cells_per_spot: Tuple[int, int] = (5, 20),
#         n_types_per_spot: Tuple[int, int] = (3, 5),  # 🔥 新增：每个spot包含的细胞类型数范围
#     ):
#         self.adata_sc = adata_sc
#         self.celltype_key = celltype_key
#         self.n_cells_per_spot = n_cells_per_spot
#         self.n_types_per_spot = n_types_per_spot
        
#         # 提取细胞类型信息
#         self.celltypes = np.array(adata_sc. obs[celltype_key]. cat.categories)
#         self.n_celltypes = len(self.celltypes)
#         self.celltype_to_idx = {ct: i for i, ct in enumerate(self.celltypes)}
        
#         # 按细胞类型分组索引
#         self.celltype_indices = {}
#         self.celltype_counts = {}
#         for ct in self.celltypes:
#             indices = np.where(adata_sc.obs[celltype_key] == ct)[0]
#             self.celltype_indices[ct] = indices
#             self.celltype_counts[ct] = len(indices)
        
#         # 🔥 计算全局细胞类型频率 P_global
#         total_cells = sum(self.celltype_counts. values())
#         self.P_global = np.array([
#             self.celltype_counts[ct] / total_cells 
#             for ct in self.celltypes
#         ])
        
#         print(f"✅ 伪点生成器初始化：{self.n_celltypes} 个细胞类型")
#         print(f"📊 全局细胞类型频率 P_global:")
#         for ct, prob in zip(self.celltypes, self.P_global):
#             print(f"   {ct}: {prob:.4f} ({self.celltype_counts[ct]} 个细胞)")
    
#     def generate_pseudo_spots(
#         self,
#         n_pseudo: int = 1000,
#         random_state: int = 42,
#         diversity_bias: float = 1.0,  # 🔥 新增：多样性偏置（>1 增加低频类型出现概率）
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         生成伪 spot（改进版）
        
#         Args:
#             n_pseudo: 生成伪点数量
#             random_state: 随机种子
#             diversity_bias: 多样性偏置因子
#                 - 1.0: 严格按 P_global 采样（高频类型主导）
#                 - >1.0: 增加低频类型的出现概率（例如 1.5）
#                 - <1.0: 进一步强化高频类型（不推荐）
        
#         Returns:
#             X_pseudo: [n_pseudo, n_genes] 混合后的表达矩阵（raw counts）
#             proportions_pseudo: [n_pseudo, n_celltypes] 真实比例
#             n_cells_pseudo: [n_pseudo] 每个伪spot包含的细胞数
#         """
#         np.random.seed(random_state)
        
#         X_sc = self.adata_sc.X
#         if hasattr(X_sc, 'toarray'):
#             X_sc = X_sc.toarray()
#         X_sc = np.asarray(X_sc, dtype=np.float32)
        
#         n_genes = X_sc.shape[1]
#         X_pseudo = []
#         proportions_pseudo = []
#         n_cells_pseudo = []
#         celltype_participation = np.zeros(self.n_celltypes)  # 统计每个细胞类型参与的伪点数
        
#         for i in range(n_pseudo):
#             # 1. 采样细胞数
#             n_cells = np.random.randint(
#                 self.n_cells_per_spot[0],
#                 self.n_cells_per_spot[1] + 1
#             )
            
#             # 2. 🔥 采样细胞类型数量（3-5 种）
#             n_types = np.random.randint(
#                 self.n_types_per_spot[0],
#                 self.n_types_per_spot[1] + 1
#             )
#             n_types = min(n_types, self.n_celltypes)  # 不能超过总类型数
            
#             # 3. 🔥 根据 P_global 选择哪些细胞类型参与
#             # 应用 diversity_bias 调整概率
#             adjusted_probs = self.P_global ** (1.0 / diversity_bias)
#             adjusted_probs = adjusted_probs / adjusted_probs.sum()  # 重新归一化
            
#             # 无放回采样 n_types 种细胞类型
#             selected_ct_indices = np.random.choice(
#                 self.n_celltypes,
#                 size=n_types,
#                 replace=False,
#                 p=adjusted_probs
#             )
            
#             # 4. 🔥 为选中的细胞类型分配比例
#             # 方案A：Dirichlet 分布（更自然）
#             alpha = np.ones(n_types) * 2.0  # alpha=2.0 产生相对均匀但有变化的比例
#             type_proportions = np.random. dirichlet(alpha)
            
#             # 方案B：根据 P_global 加权的 Dirichlet（可选）
#             # alpha = self.P_global[selected_ct_indices] * 10.0
#             # type_proportions = np.random.dirichlet(alpha)
            
#             # 5. 根据比例确定每个细胞类型的细胞数
#             celltype_counts = np.random.multinomial(n_cells, type_proportions)
            
#             # 6.  从每个细胞类型采样细胞
#             selected_cells = []
#             for ct_idx, n_ct in zip(selected_ct_indices, celltype_counts):
#                 if n_ct > 0:
#                     ct_name = self.celltypes[ct_idx]
#                     ct_indices = self.celltype_indices[ct_name]
#                     if len(ct_indices) == 0:
#                         continue
#                     # 有放回采样（允许同一个细胞被多次选中）
#                     sampled = np.random.choice(ct_indices, size=n_ct, replace=True)
#                     selected_cells.extend(sampled)
#                     celltype_participation[ct_idx] += 1
            
#             if len(selected_cells) == 0:
#                 continue
            
#             # 7. 混合表达（求和）
#             x_mixed = X_sc[selected_cells]. sum(axis=0)
            
#             # 8. 🔥 可选：调整库大小（使其接近真实 spot 的库大小）
#             # 如果需要，可以归一化到目标库大小
#             # target_library = 10000  # 例如
#             # current_library = x_mixed.sum()
#             # x_mixed = x_mixed * (target_library / current_library)
            
#             # 9. 计算实际比例（全局，包含未参与的类型）
#             actual_proportions = np.zeros(self.n_celltypes, dtype=np.float32)
#             for ct_idx, n_ct in zip(selected_ct_indices, celltype_counts):
#                 actual_proportions[ct_idx] = n_ct / len(selected_cells)
            
#             X_pseudo.append(x_mixed)
#             proportions_pseudo. append(actual_proportions)
#             n_cells_pseudo.append(len(selected_cells))
        
#         X_pseudo = np.array(X_pseudo, dtype=np.float32)
#         proportions_pseudo = np. array(proportions_pseudo, dtype=np.float32)
#         n_cells_pseudo = np. array(n_cells_pseudo, dtype=np.int32)
        
#         # 统计信息
#         print(f"✅ 生成 {len(X_pseudo)} 个伪 spots")
#         print(f"📊 伪点统计:")
#         print(f"   平均细胞数: {n_cells_pseudo.mean():.1f} ± {n_cells_pseudo.std():.1f}")
#         print(f"   表达量范围: [{X_pseudo.sum(axis=1).min():.0f}, {X_pseudo.sum(axis=1).max():.0f}]")
#         print(f"   平均非零细胞类型数: {(proportions_pseudo > 0).sum(axis=1).mean():.1f}")
        
#         print(f"📊 各细胞类型参与的伪点数:")
#         for ct, count in zip(self.celltypes, celltype_participation):
#             participation_rate = count / n_pseudo
#             print(f"   {ct}: {int(count)} 个伪点 ({participation_rate:.2%})")
        
#         return X_pseudo, proportions_pseudo, n_cells_pseudo
    
#     def fit_prior_from_real_st(self, adata_st, proportion_key: str = "cell_type_fractions"):
#         """
#         从真实 ST 数据学习细胞类型共现模式（可选）
        
#         注意：这个方法在新策略下可能不需要，因为我们已经用 P_global 了
#         """
#         if proportion_key not in adata_st.obsm:
#             print(f"⚠ 未找到 {proportion_key}，跳过先验学习")
#             return
        
#         proportions = adata_st.obsm[proportion_key]  # [N, n_celltypes]
        
#         # 统计每个细胞类型在真实 spot 中的平均比例
#         mean_proportions = proportions.mean(axis=0)
        
#         print(f"📊 真实 ST 数据中的平均细胞类型比例:")
#         for ct, prop in zip(self.celltypes, mean_proportions):
#             print(f"   {ct}: {prop:.4f}")
        
#         # 可以用这个信息调整 P_global（但通常不需要）
#         # self.P_global = 0.5 * self.P_global + 0. 5 * mean_proportions
#         # self.P_global = self.P_global / self.P_global.sum()
"""
伪点生成模块 - "均匀采样"或"加权采样"策略
"""
import numpy as np
import torch
from typing import Optional, Tuple


class PseudoSpotGeneratorUniform:
    """
    从单细胞数据生成伪 spot（"均匀采样"或"加权采样"策略）

    新策略：
    - 每个伪点包含的细胞类型不受 SC 中全局比例影响。
    - 所有细胞类型被平等对待或通过 `boost_weights` 加权。
    """
    def __init__(
        self,
        adata_sc,
        celltype_key: str = "cell_type",
        n_cells_per_spot: Tuple[int, int] = (5, 20),
        n_types_per_spot: Tuple[int, int] = (3, 5),  # 每个伪点包含的细胞类型数量范围
        boost_weights: Optional[np.ndarray] = None,  # 各细胞类型的采样概率权重
    ):
        self.adata_sc = adata_sc
        self.celltype_key = celltype_key
        self.n_cells_per_spot = n_cells_per_spot
        self.n_types_per_spot = n_types_per_spot
        
        # 提取细胞类型信息
        self.celltypes = np.array(adata_sc.obs[celltype_key].cat.categories)
        self.n_celltypes = len(self.celltypes)
        self.celltype_to_idx = {ct: i for i, ct in enumerate(self.celltypes)}
        
        # 按细胞类型分组索引
        self.celltype_indices = {ct: np.where(adata_sc.obs[celltype_key] == ct)[0] for ct in self.celltypes}
        
        # 采样权重（默认为均匀平等采样）
        if boost_weights is not None:
            # 使用加权采样
            assert len(boost_weights) == self.n_celltypes, "boost_weights 的长度必须与细胞类型数量相等"
            self.sampling_probs = np.array(boost_weights, dtype=np.float32)
        else:
            # 均匀采样
            self.sampling_probs = np.ones(self.n_celltypes, dtype=np.float32)
        
        # 归一化权重
        self.sampling_probs = self.sampling_probs / self.sampling_probs.sum()
        
        print(f"✅ 伪点生成器初始化：{self.n_celltypes} 个细胞类型")
        print(f"📊 采样概率（均匀或加权）:")
        for ct, prob in zip(self.celltypes, self.sampling_probs):
            print(f"   {ct}: {prob:.4f} ({len(self.celltype_indices[ct])} 个细胞)")
    
    def generate_pseudo_spots(
        self,
        n_pseudo: int = 1000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成伪 spot
        
        Returns:
            X_pseudo: [n_pseudo, n_genes] 混合后的表达矩阵（raw counts）
            proportions_pseudo: [n_pseudo, n_celltypes] 真实比例
            n_cells_pseudo: [n_pseudo] 每个伪点包含的细胞数
        """
        np.random.seed(random_state)
        
        X_sc = self.adata_sc.X
        if hasattr(X_sc, 'toarray'):
            X_sc = X_sc.toarray()
        X_sc = np.asarray(X_sc, dtype=np.float32)
        
        n_genes = X_sc.shape[1]
        X_pseudo = []
        proportions_pseudo = []
        n_cells_pseudo = []
        
        for i in range(n_pseudo):
            # 1. 采样细胞数
            n_cells = np.random.randint(
                self.n_cells_per_spot[0],
                self.n_cells_per_spot[1] + 1
            )
            
            # 2. 采样细胞类型数量（3-5 种）
            n_types = np.random.randint(
                self.n_types_per_spot[0],
                self.n_types_per_spot[1] + 1
            )
            n_types = min(n_types, self.n_celltypes)  # 不能超过总类型数
            
            # 3. 根据均匀或加权采样确定细胞类型
            selected_ct_indices = np.random.choice(
                self.n_celltypes,
                size=n_types,
                replace=False,
                p=self.sampling_probs
            )
            
            # 4. 为选中的细胞类型分配比例（Dirichlet 分布）
            alpha = np.ones(n_types) * 2.0
            type_proportions = np.random.dirichlet(alpha)
            
            # 5. 根据比例确定每个细胞类型的细胞数
            celltype_counts = np.random.multinomial(n_cells, type_proportions)
            
            # 6. 从每个细胞类型采样细胞
            selected_cells = []
            for ct_idx, n_ct in zip(selected_ct_indices, celltype_counts):
                if n_ct > 0:
                    ct_name = self.celltypes[ct_idx]
                    ct_indices = self.celltype_indices[ct_name]
                    sampled = np.random.choice(ct_indices, size=n_ct, replace=True)
                    selected_cells.extend(sampled)
            
            # 7. 混合表达（求和）
            x_mixed = X_sc[selected_cells].sum(axis=0)
            
            # 8. 计算实际比例（全局，包含未参与的类型）
            actual_proportions = np.zeros(self.n_celltypes, dtype=np.float32)
            for ct_idx, n_ct in zip(selected_ct_indices, celltype_counts):
                actual_proportions[ct_idx] = n_ct / len(selected_cells)
            
            X_pseudo.append(x_mixed)
            proportions_pseudo.append(actual_proportions)
            n_cells_pseudo.append(len(selected_cells))
        
        X_pseudo = np.array(X_pseudo, dtype=np.float32)
        proportions_pseudo = np.array(proportions_pseudo, dtype=np.float32)
        n_cells_pseudo = np.array(n_cells_pseudo, dtype=np.int32)
        
        print(f"✅ 生成 {len(X_pseudo)} 个伪 spots")
        print(f"📊 平均非零细胞类型数: {(proportions_pseudo > 0).sum(axis=1).mean():.1f}")
        
        return X_pseudo, proportions_pseudo, n_cells_pseudo
    
"""
伪点生成模块 - 修复 Label 顺序问题
"""
import numpy as np
import torch
from typing import Tuple, Optional


class PseudoSpotGenerator:
    """
    从单细胞数据生成伪 spot
    
    🔥 关键修复：确保细胞类型顺序与 SingleCellDataset 一致
    """
    def __init__(
        self,
        adata_sc,
        celltype_key: str = "cell_type",
        n_cells_per_spot: Tuple[int, int] = (5, 20),
        n_types_per_spot: Tuple[int, int] = (3, 5),
        celltype_order: Optional[np.ndarray] = None,  # 🔥 新增：强制指定细胞类型顺序
    ):
        self.adata_sc = adata_sc
        self.celltype_key = celltype_key
        self.n_cells_per_spot = n_cells_per_spot
        self.n_types_per_spot = n_types_per_spot
        
        # 🔥 关键修复：使用外部传入的细胞类型顺序
        if celltype_order is not None:
            self.celltypes = celltype_order
            print(f"✅ 使用外部指定的细胞类型顺序（长度={len(celltype_order)}）")
        else:
            # 回退到直接读取（可能不一致）
            self.celltypes = np.array(adata_sc.obs[celltype_key].cat.categories)
            print(f"⚠️ 警告：未指定细胞类型顺序，使用 adata 默认顺序")
        
        self.n_celltypes = len(self.celltypes)
        self.celltype_to_idx = {ct: i for i, ct in enumerate(self.celltypes)}
        
        # 🔥 验证：打印细胞类型顺序
        print(f"📊 伪点生成器的细胞类型顺序:")
        for i, ct in enumerate(self. celltypes):
            print(f"   [{i}] {ct}")
        
        # 按细胞类型分组索引
        self.celltype_indices = {}
        self.celltype_counts = {}
        for ct in self.celltypes:
            indices = np.where(adata_sc.obs[celltype_key] == ct)[0]
            self.celltype_indices[ct] = indices
            self.celltype_counts[ct] = len(indices)
        
        # 🔥 计算全局细胞类型频率 P_global（按指定顺序）
        total_cells = sum(self.celltype_counts. values())
        self.P_global = np.array([
            self.celltype_counts[ct] / total_cells 
            for ct in self.celltypes
        ])
        
        print(f"📊 全局细胞类型频率 P_global:")
        for ct, prob in zip(self.celltypes, self.P_global):
            print(f"   {ct}: {prob:.4f} ({self.celltype_counts[ct]} 个细胞)")
    
    def generate_pseudo_spots(
        self,
        n_pseudo: int = 1000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成伪 spot
        
        Returns:
            X_pseudo: [n_pseudo, n_genes] 混合后的表达矩阵（raw counts）
            proportions_pseudo: [n_pseudo, n_celltypes] 真实比例（🔥 按 self.celltypes 顺序）
            n_cells_pseudo: [n_pseudo] 每个伪点包含的细胞数
        """
        np.random.seed(random_state)
        
        X_sc = self.adata_sc.X
        if hasattr(X_sc, 'toarray'):
            X_sc = X_sc.toarray()
        X_sc = np.asarray(X_sc, dtype=np.float32)
        
        n_genes = X_sc.shape[1]
        X_pseudo = []
        proportions_pseudo = []
        n_cells_pseudo = []
        
        for i in range(n_pseudo):
            # 1. 采样细胞数
            n_cells = np.random.randint(
                self.n_cells_per_spot[0],
                self.n_cells_per_spot[1] + 1
            )
            
            # 2. 采样细胞类型数量（3-5 种）
            n_types = np.random.randint(
                self.n_types_per_spot[0],
                self. n_types_per_spot[1] + 1
            )
            n_types = min(n_types, self.n_celltypes)
            n_types = min(n_types, n_cells)
            
            # 3. 🔥 均匀采样细胞类型（所有类型平等对待）
            selected_ct_indices = np.random.choice(
                self.n_celltypes,
                size=n_types,
                replace=False,
                p=None  # 均匀采样
            )
            
            # # 4. 为选中的细胞类型分配比例（Dirichlet 分布）
            # alpha = np.ones(n_types) * 2.0
            # type_proportions = np.random. dirichlet(alpha)
            
            # # 5.  根据比例确定每个细胞类型的细胞数
            # celltype_counts = np.random.multinomial(n_cells, type_proportions)
            # 4. 🔥 强制保底分配
            # 先给每个选中的类型分配 1 个细胞
            current_counts = np.ones(n_types, dtype=int)
            remaining_cells = n_cells - n_types
            
            if remaining_cells > 0:
                # 剩下的名额再随机分配
                # 可以均匀分，也可以按 Dirichlet 分
                # 这里为了简单且保持多样性，使用均匀分布再分配剩余名额
                extra_counts = np.random.multinomial(remaining_cells, np.ones(n_types)/n_types)
                celltype_counts = current_counts + extra_counts
            else:
                # 如果 n_cells 比 n_types 还少（比如 n_cells=2, n_types=3），
                # 理论上应该不会发生，因为 n_types = min(n_types, n_cells) 更好
                # 或者在这里做截断。但根据你的参数设置 n_cells_min=2, n_types_min=1，可能会有 n_cells < n_types 的情况
                # 建议在前面加一句：
                # n_types = min(n_types, n_cells) 
                
                # 如果没加，就截断
                celltype_counts = current_counts[:n_cells]
                # 补齐长度
                if len(celltype_counts) < n_types:
                     # 这种情况选中的类型有些就被丢弃了，但这比 multinomial 随机丢弃要好控制
                     pass 
            
            # 5.  从每个细胞类型采样细胞 (逻辑稍微调整)
            selected_cells = []
            # 注意：celltype_counts 的长度是 n_types，对应 selected_ct_indices
            
            for i, ct_idx in enumerate(selected_ct_indices):
                # 如果上面截断了，这里要防止越界
                if i >= len(celltype_counts): break
                
                n_ct = celltype_counts[i]
                if n_ct > 0:
                    ct_name = self.celltypes[ct_idx]
                    ct_indices = self.celltype_indices[ct_name]
                    if len(ct_indices) == 0:
                        continue
                    sampled = np.random.choice(ct_indices, size=n_ct, replace=True)
                    selected_cells.extend(sampled)
            
            # 6.  从每个细胞类型采样细胞
            selected_cells = []
            for ct_idx, n_ct in zip(selected_ct_indices, celltype_counts):
                if n_ct > 0:
                    ct_name = self.celltypes[ct_idx]
                    ct_indices = self.celltype_indices[ct_name]
                    if len(ct_indices) == 0:
                        continue
                    sampled = np.random.choice(ct_indices, size=n_ct, replace=True)
                    selected_cells. extend(sampled)
            
            if len(selected_cells) == 0:
                continue
            
            # 7. 混合表达（求和）- Raw Counts
            x_mixed = X_sc[selected_cells]. sum(axis=0)
            
            # 8. 🔥 计算实际比例（全局，包含未参与的类型，按 self.celltypes 顺序）
            actual_proportions = np.zeros(self.n_celltypes, dtype=np.float32)
            for ct_idx, n_ct in zip(selected_ct_indices, celltype_counts):
                actual_proportions[ct_idx] = n_ct / len(selected_cells)
            
            X_pseudo.append(x_mixed)
            proportions_pseudo. append(actual_proportions)
            n_cells_pseudo.append(len(selected_cells))
        
        X_pseudo = np.array(X_pseudo, dtype=np.float32)
        proportions_pseudo = np. array(proportions_pseudo, dtype=np.float32)
        n_cells_pseudo = np. array(n_cells_pseudo, dtype=np.int32)
        
        print(f"✅ 生成 {len(X_pseudo)} 个伪 spots")
        print(f"📊 平均细胞数: {n_cells_pseudo.mean():.1f} ± {n_cells_pseudo. std():.1f}")
        print(f"📊 平均非零细胞类型数: {(proportions_pseudo > 0).sum(axis=1).mean():.1f}")
        
        # 🔥 验证比例维度
        print(f"📊 伪点比例矩阵形状: {proportions_pseudo.shape} (应该是 [n_pseudo, {self.n_celltypes}])")
        
        return X_pseudo, proportions_pseudo, n_cells_pseudo