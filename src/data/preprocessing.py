"""数据预处理工具"""
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


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