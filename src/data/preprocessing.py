"""数据预处理工具"""
import random
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
import torch


def filter_genes(adata, min_cells=5, min_counts=10):
    """基因过滤"""
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    return adata


def select_hvgs(sc_adata, st_adata, n_top_genes=4000, batch_key=None):
    """选择高变基因"""
    # 计算高变基因
    sc.pp.highly_variable_genes(
        sc_adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        layer="counts",
        batch_key=batch_key,
        subset=False
    )
    
    # 获取共有基因
    hvgs_sc = sc_adata.var_names[sc_adata.var['highly_variable'].fillna(False)]
    genes_st = st_adata.var_names
    shared_genes = hvgs_sc.intersection(genes_st)
    
    print(f"HVGs from sc: {len(hvgs_sc)}")
    print(f"Total genes in st: {len(genes_st)}")
    print(f"Shared genes: {len(shared_genes)}")
    
    # 子集化
    sc_adata = sc_adata[:, shared_genes].copy()
    st_adata = st_adata[:, shared_genes].copy()
    
    return sc_adata, st_adata, shared_genes


def prepare_anndata(adata):
    """准备AnnData对象"""
    # 确保基因名唯一
    adata.var_names_make_unique()
    
    # 保存原始计数
    if 'counts' not in adata.layers:
        if issparse(adata.X):
            adata.layers['counts'] = adata.X.copy()
        else:
            adata.layers['counts'] = adata.X.copy()
    
    return adata

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 PyTorch 操作的算法为确定性（PyTorch>=1.12）
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception as e:
    #     print("Warning: deterministic_algorithms not set:", e)

def preprocess_sc_st(adata_sc, adata_st, sc_n_top_genes=12000, st_n_top_genes=15000, mt_prefixes=("MT.", "MT-")):
    """对单细胞和空间数据AnnData对象进行标准预处理
    返回：预处理后的 adata_sc, adata_st（会原地修改），以及二者高度变基因交集
    """
    # ST 预处理
    adata_st.var['MT_gene'] = [any(gene.startswith(p) for p in mt_prefixes) for gene in adata_st.var_names]
    adata_st = adata_st[:, ~adata_st.var['MT_gene'].values]
    sc.pp.filter_cells(adata_st, min_genes=10)
    sc.pp.filter_genes(adata_st, min_cells=2)
    adata_st.layers['counts'] = adata_st.X.copy()
    sc.pp.normalize_total(adata_st)
    sc.pp.log1p(adata_st)
    adata_st.raw = adata_st
    set_seed(0)
    sc.pp.highly_variable_genes(
        adata_st,
        n_top_genes=st_n_top_genes,
        layer="counts",
        flavor="seurat_v3",
        span=1
    )
    adata_st = adata_st[:, adata_st.var.highly_variable]

    # SC 预处理
    adata_sc.var['MT_gene'] = [any(gene.startswith(p) for p in mt_prefixes) for gene in adata_sc.var_names]
    adata_sc = adata_sc[:, ~adata_sc.var['MT_gene'].values]
    sc.pp.filter_cells(adata_sc, min_genes=1)
    adata_sc.layers['counts'] = adata_sc.X.copy()
    sc.pp.normalize_total(adata_sc)
    sc.pp.log1p(adata_sc)
    adata_sc.raw = adata_sc
    set_seed(42)
    sc.pp.highly_variable_genes(
        adata_sc,
        n_top_genes=sc_n_top_genes
    )
    adata_sc = adata_sc[:, adata_sc.var.highly_variable]

    # 计算共同基因（可选，可用于后续对齐）
    common_genes = np.intersect1d(adata_sc.var.index, adata_st.var.index)
    print(f"sc和st表达矩阵交集基因数: {len(common_genes)}")
    return adata_sc, adata_st, common_genes