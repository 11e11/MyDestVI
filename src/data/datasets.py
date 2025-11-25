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