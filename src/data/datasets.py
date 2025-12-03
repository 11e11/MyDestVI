# """数据集定义"""
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from scipy.sparse import issparse


# class SingleCellDataset(Dataset):
#     """单细胞RNA-seq数据集"""
#     def __init__(self, adata, label_key: str, use_layer: str | None = None):
#         """
#         Args:
#             adata: AnnData对象
#             label_key: 细胞类型标签的键
#             use_layer: 使用的layer，None则使用X
#         """
#         # 提取表达矩阵
#         if use_layer and use_layer in adata.layers:
#             X = adata.layers[use_layer]
#         else:
#             X = adata.X
        
#         self.X = X.toarray() if issparse(X) else np.array(X)
#         self.X = torch.FloatTensor(self.X)
        
#         # 提取标签
#         if label_key not in adata.obs.columns:
#             raise ValueError(f"Label key '{label_key}' not found in adata.obs")
        
#         labels = adata.obs[label_key]
#         if hasattr(labels, 'cat'):
#             self.labels = torch.LongTensor(labels.cat.codes.values)
#             self.label_names = labels.cat.categories.values
#             self.n_labels = len(self.label_names)
#         else:
#             raise ValueError(f"Column '{label_key}' must be categorical")
        
#         # 批次信息（如果有）
#         self.batch = None
#         if 'batch' in adata.obs.columns:
#             batch = adata.obs['batch']
#             if hasattr(batch, 'cat'):
#                 self.batch = torch.LongTensor(batch.cat.codes.values)
#                 self.n_batch = len(batch.cat.categories)
#             else:
#                 self.batch = torch.LongTensor(batch.values)
#                 self.n_batch = len(np.unique(batch))
#         else:
#             self.n_batch = 1
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         item = {
#             'X': self.X[idx],
#             'labels': self.labels[idx]
#         }
#         if self.batch is not None:
#             item['batch'] = self.batch[idx]
#         return item


# class SpatialDataset(Dataset):
#     """空间转录组数据集"""
#     def __init__(self, adata, use_layer: str | None = None, spatial_key: str = 'spatial'):
#         """
#         Args:
#             adata: AnnData对象
#             use_layer: 使用的layer
#             spatial_key: 空间坐标的键
#         """
#         # 表达矩阵
#         if use_layer and use_layer in adata.layers:
#             X = adata.layers[use_layer]
#         else:
#             X = adata.X
        
#         self.X = X.toarray() if issparse(X) else np.array(X)
#         self.X = torch.FloatTensor(self.X)
        
#         # 索引
#         self.indices = torch.arange(len(self.X), dtype=torch.long)
        
#         # 空间坐标
#         self.spatial = None
#         if spatial_key in adata.obsm:
#             self.spatial = torch.FloatTensor(adata.obsm[spatial_key])
        
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         item = {
#             'X': self.X[idx],
#             'ind_x': self.indices[idx]
#         }
#         if self.spatial is not None:
#             item['spatial'] = self.spatial[idx]
#         return item
"""数据集定义（修复 NumPy 只读数组导致的 PyTorch 警告）"""
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse


def _to_writable_float32_ndarray(X) -> np.ndarray:
    """
    将输入矩阵转换为可写、C 连续、float32 的 numpy 数组。
    - 稀疏 -> toarray()
    - 非稀疏 -> 强制 copy=True, order='C'
    """
    if issparse(X):
        arr = X.toarray()
    else:
        # 有些 anndata.X/layers 返回只读视图或内存映射视图
        arr = np.array(X, dtype=np.float32, copy=True, order="C")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    # 确保可写
    if not arr.flags.writeable:
        arr = arr.copy()
    # 确保 C 连续
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _to_writable_int64_ndarray(x) -> np.ndarray:
    """
    将标签/批次一维数据转换为可写 int64 ndarray。
    """
    arr = np.asarray(x, dtype=np.int64)
    if not arr.flags.writeable:
        arr = arr.copy()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


class SingleCellDataset(Dataset):
    """单细胞RNA-seq数据集"""
    def __init__(self, adata, label_key: str, use_layer: str | None = None):
        """
        Args:
            adata: AnnData对象
            label_key: 细胞类型标签的键（必须是分类类型 categorical）
            use_layer: 使用的layer，None则使用X
        """
        # 表达矩阵 -> 可写 float32 ndarray -> torch.tensor（强制复制，避免只读共享）
        if use_layer and (hasattr(adata, "layers") and use_layer in adata.layers):
            X = adata.layers[use_layer]
        else:
            X = adata.X
        X_np = _to_writable_float32_ndarray(X)
        self.X = torch.tensor(X_np, dtype=torch.float32)

        # 提取标签（要求 categorical），保持与 pandas 的编码一致
        if label_key not in adata.obs.columns:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs")

        labels = adata.obs[label_key]
        if hasattr(labels, "cat"):
            codes = _to_writable_int64_ndarray(labels.cat.codes.values)
            self.labels = torch.from_numpy(codes)
            self.label_names = labels.cat.categories.astype(str).values
            self.n_labels = len(self.label_names)
        else:
            raise ValueError(f"Column '{label_key}' must be categorical (please make it .astype('category'))")

        # 批次信息（可选）
        self.batch = None
        if "batch" in adata.obs.columns:
            batch = adata.obs["batch"]
            if hasattr(batch, "cat"):
                bcodes = _to_writable_int64_ndarray(batch.cat.codes.values)
                self.batch = torch.from_numpy(bcodes)
                self.n_batch = len(batch.cat.categories)
            else:
                bvals = _to_writable_int64_ndarray(batch.values)
                self.batch = torch.from_numpy(bvals)
                self.n_batch = int(np.unique(bvals).size)
        else:
            self.n_batch = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        item = {
            "X": self.X[idx],
            "labels": self.labels[idx],
        }
        if self.batch is not None:
            item["batch"] = self.batch[idx]
        return item


class SpatialDataset(Dataset):
    """空间转录组数据集（支持双输入：raw counts + preprocessed features）"""
    def __init__(self, adata, 
                 use_layer: str | None = None,
                 encoder_layer: str | None = None,  # 🔥 新增：用于 encoder 的预处理 layer
                 spatial_key: str = "spatial"):
        """
        Args:
            adata: AnnData对象
            use_layer: 用于 likelihood 重构的 layer (raw counts)
            encoder_layer: 用于 encoder 的预处理 layer (normalized + log1p)
            spatial_key: 空间坐标的键
        """
        # 1. Raw counts for likelihood (用于 NB 重构)
        if use_layer and (hasattr(adata, "layers") and use_layer in adata.layers):
            X_raw = adata.layers[use_layer]
        else:
            X_raw = adata.X
        X_raw_np = _to_writable_float32_ndarray(X_raw)
        self.X_raw = torch.tensor(X_raw_np, dtype=torch.float32)
        
        # 2. Preprocessed features for encoder (🔥 新增)
        if encoder_layer and (hasattr(adata, "layers") and encoder_layer in adata.layers):
            X_encoder = adata.layers[encoder_layer]
        else:
            # 默认：如果没指定 encoder_layer，就用 raw counts（退化到原始行为）
            X_encoder = X_raw
        
        X_encoder_np = _to_writable_float32_ndarray(X_encoder)
        self.X_encoder = torch.tensor(X_encoder_np, dtype=torch.float32)
        
        # 全局索引
        self.indices = torch.arange(self.X_raw.shape[0], dtype=torch.long)
        
        # 空间坐标（可选）
        self.spatial = None
        if hasattr(adata, "obsm") and spatial_key in adata.obsm:
            spatial_np = np.array(adata.obsm[spatial_key], dtype=np.float32, copy=True, order="C")
            self.spatial = torch.tensor(spatial_np, dtype=torch.float32)
    
    def __len__(self):
        return self.X_raw.shape[0]
    
    def __getitem__(self, idx):
        item = {
            "X": self.X_raw[idx],              # 🔥 raw counts for likelihood
            "X_encoder": self.X_encoder[idx],  # 🔥 preprocessed for encoder
            "ind_x": self.indices[idx],
        }
        if self.spatial is not None:
            item["spatial"] = self.spatial[idx]
        return item
    
class STDatasetWithPseudo(Dataset):
    """混合数据集"""
    def __init__(
        self,
        adata_st,
        X_pseudo: Optional[np.ndarray] = None,
        proportions_pseudo: Optional[np.ndarray] = None,
        pseudo_pass_filter: Optional[np.ndarray] = None,  # 🔥 新增
        layer: str = None,
    ):
        from scipy.sparse import issparse
        
        # 真实数据
        if layer is not None and layer in adata_st.layers:
            X_real = adata_st.layers[layer]
        else:
            X_real = adata_st.X
        if issparse(X_real):
            X_real = X_real.toarray()
        X_real = np.asarray(X_real, dtype=np.float32)
        
        # 真实数据 encoder 特征
        if 'normalized_log' in adata_st.layers:
            X_encoder_real = adata_st.layers['normalized_log']
            if issparse(X_encoder_real):
                X_encoder_real = X_encoder_real.toarray()
            X_encoder_real = np.asarray(X_encoder_real, dtype=np.float32)
        else:
            X_encoder_real = None
        
        self.n_real = X_real.shape[0]
        
        # 伪点数据
        self.use_pseudo = (X_pseudo is not None and proportions_pseudo is not None)
        if self.use_pseudo:
            X_pseudo = X_pseudo.astype(np.float32)
            proportions_pseudo = proportions_pseudo.astype(np. float32)
            self.n_pseudo = len(X_pseudo)
            
            # 归一化
            library_pseudo = X_pseudo.sum(axis=1, keepdims=True)
            X_encoder_pseudo = np.log1p(X_pseudo / (library_pseudo + 1e-6) * 10000)
            
            # 拼接
            self.X_all = np.vstack([X_real, X_pseudo])
            if X_encoder_real is not None:
                self.X_encoder_all = np.vstack([X_encoder_real, X_encoder_pseudo])
            else:
                self.X_encoder_all = None
            
            self. proportions_pseudo = proportions_pseudo
            
            # 🔥 保存过滤信息
            if pseudo_pass_filter is not None:
                self.pseudo_pass_filter = pseudo_pass_filter
                n_pass = pseudo_pass_filter.sum()
                print(f"✅ STDatasetWithPseudo：{self.n_real} 真实 + {self.n_pseudo} 伪点（{n_pass} 个通过过滤）")
            else:
                # 如果没有提供过滤信息，默认全部通过
                self.pseudo_pass_filter = np.ones(self.n_pseudo, dtype=bool)
                print(f"⚠️ 未提供过滤信息，默认所有伪点通过")
        else:
            self. X_all = X_real
            self.X_encoder_all = X_encoder_real
            self.n_pseudo = 0
            self.proportions_pseudo = None
            self.pseudo_pass_filter = None
        
        self.n_total = len(self.X_all)
    
    def __len__(self):
        return self.n_total
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X_all[idx])
        ind_x = torch.tensor(idx, dtype=torch.long)
        
        if self.X_encoder_all is not None:
            X_encoder = torch.from_numpy(self.X_encoder_all[idx])
        else:
            X_encoder = X
        
        item = {
            "X": X,
            "X_encoder": X_encoder,
            "ind_x": ind_x,
        }
        
        if idx < self.n_real:
            item["is_pseudo"] = torch.tensor(0, dtype=torch.long)
        else:
            pseudo_idx = idx - self.n_real
            item["is_pseudo"] = torch.tensor(1, dtype=torch.long)
            item["pseudo_proportion"] = torch.from_numpy(self.proportions_pseudo[pseudo_idx])
            item["pseudo_pass_filter"] = torch.tensor(self.pseudo_pass_filter[pseudo_idx], dtype=torch.bool)  # 🔥 新增
        
        return item


def collate_fn_with_pseudo(batch):
    """自定义 collate 函数"""
    X = torch.stack([item["X"] for item in batch])
    X_encoder = torch.stack([item["X_encoder"] for item in batch]) if "X_encoder" in batch[0] else None
    ind_x = torch.stack([item["ind_x"] for item in batch])
    is_pseudo = torch.stack([item["is_pseudo"] for item in batch])
    
    has_pseudo = any(item. get("is_pseudo", 0) == 1 for item in batch)
    
    if has_pseudo:
        n_celltypes = None
        for item in batch:
            if item.get("is_pseudo", 0) == 1 and "pseudo_proportion" in item:
                n_celltypes = item["pseudo_proportion"].size(0)
                break
        
        if n_celltypes is None:
            raise ValueError("Found is_pseudo=1 but no pseudo_proportion")
        
        pseudo_proportions = []
        pseudo_pass_filters = []  # 🔥 新增
        for item in batch:
            if item. get("is_pseudo", 0) == 1 and "pseudo_proportion" in item:
                pseudo_proportions.append(item["pseudo_proportion"])
                pseudo_pass_filters.append(item["pseudo_pass_filter"])  # 🔥 新增
            else:
                pseudo_proportions. append(torch.zeros(n_celltypes))
                pseudo_pass_filters. append(torch.tensor(False))  # 真实点标记为False
        
        pseudo_proportions = torch.stack(pseudo_proportions)
        pseudo_pass_filters = torch.stack(pseudo_pass_filters)  # 🔥 新增
    else:
        pseudo_proportions = None
        pseudo_pass_filters = None
    
    result = {
        "X": X,
        "ind_x": ind_x,
        "is_pseudo": is_pseudo,
    }
    
    if X_encoder is not None:
        result["X_encoder"] = X_encoder
    
    if pseudo_proportions is not None:
        result["pseudo_proportion"] = pseudo_proportions
        result["pseudo_pass_filter"] = pseudo_pass_filters  # 🔥 新增
    
    return result
    