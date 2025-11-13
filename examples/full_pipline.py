"""完整训练流程示例"""
import scanpy as sc
import torch
from torch.utils.data import DataLoader
import pandas as pd

from standalone.data.datasets import SingleCellDataset, SpatialDataset
from standalone.data.preprocessing import prepare_anndata, select_hvgs, filter_genes
from standalone.models.sc.condscvi import CondSCVI
from standalone.models.st.destvi import DestVI
from standalone.training.trainer import Trainer
from standalone.training.inference import InferenceEngine


def main():
    # ========== 1. 加载和预处理数据 ==========
    print("Loading data...")
    sc_adata = sc.read_h5ad("data/DLPFC/sc/DLPFC_snRNAseq.h5ad")
    st_adata = sc.read_h5ad("data/DLPFC/st/151510/filtered_feature_bc_matrix.h5ad")
    
    print("Preprocessing...")
    sc_adata = prepare_anndata(sc_adata)
    st_adata = prepare_anndata(st_adata)
    
    # 基因过滤
    sc_adata = filter_genes(sc_adata, min_cells=5, min_counts=10)
    st_adata = filter_genes(st_adata, min_cells=5, min_counts=10)
    
    # 高变基因选择
    sc_adata, st_adata, shared_genes = select_hvgs(
        sc_adata, st_adata, n_top_genes=4000
    )
    
    # 保存基因列表
    pd.DataFrame(list(shared_genes)).to_csv(
        "checkpoints/training_genes.csv", 
        index=False, header=False
    )
    
    # ========== 2. 创建数据集 ==========
    print("Creating datasets...")
    sc_dataset = SingleCellDataset(
        sc_adata, 
        label_key="cellType_broad_k",
        use_layer="counts"
    )
    
    st_dataset = SpatialDataset(
        st_adata,
        use_layer=None  # 使用X
    )
    
    sc_loader = DataLoader(sc_dataset, batch_size=256, shuffle=True, num_workers=4)
    st_loader = DataLoader(st_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # ========== 3. 训练CondSCVI ==========
    print("\n" + "="*50)
    print("Training CondSCVI...")
    print("="*50)
    
    # 计算细胞类型计数
    ct_counts = sc_adata.obs["cellType_broad_k"].value_counts().sort_index().values
    
    sc_model = CondSCVI(
        n_input=sc_adata.n_vars,
        n_labels=sc_dataset.n_labels,
        n_batch=sc_dataset.n_batch,
        n_hidden=128,
        n_latent=5,
        n_layers=2,
        weight_obs=False,  # 设为True可以对低频细胞类型重加权
        dropout_rate=0.05,
        ct_counts=ct_counts,
        encode_covariates=False
    )
    
    sc_optimizer = torch.optim.Adam(sc_model.parameters(), lr=0.001)
    sc_trainer = Trainer(sc_model, sc_optimizer, device='cuda')
    
    sc_trainer.train(
        sc_loader,
        max_epochs=300,
        n_epochs_kl_warmup=50,
        print_every=50
    )
    
    # 保存CondSCVI
    sc_trainer.save("checkpoints/condscvi.pth")
    print("CondSCVI training completed!")
    
    # ========== 4. 计算VampPrior ==========
    print("\nComputing VampPrior...")
    # 准备全量数据用于计算VampPrior
    from scipy.sparse import issparse
    X_full = sc_adata.X.toarray() if issparse(sc_adata.X) else sc_adata.X
    X_full = torch.FloatTensor(X_full).to('cuda')
    labels_full = torch.LongTensor(sc_adata.obs["cellType_broad_k"].cat.codes.values).to('cuda')
    
    mean_vprior, var_vprior, mp_vprior = sc_model.get_vamp_prior(
        X_full, labels_full, p=15
    )
    
    # ========== 5. 训练DestVI ==========
    print("\n" + "="*50)
    print("Training DestVI...")
    print("="*50)
    
    st_model = DestVI(
        n_spots=st_adata.n_obs,
        n_labels=sc_dataset.n_labels,
        n_genes=st_adata.n_vars,
        n_latent=5,
        n_hidden=128,
        n_layers=2,
        condscvi_model=sc_model,
        vamp_prior_p=15,
        l1_reg=10.0,
        dirichlet_alpha=0.4,
        dirichlet_mmd_reg=2.0,
        contrastive_reg=0.0,  # 不使用对比学习
        use_gat=True,
        gat_hidden=64,
        gat_heads=4,
        mean_vprior=mean_vprior,
        var_vprior=var_vprior,
        mp_vprior=mp_vprior
    )
    
    # 附加全量X和图结构（如果使用GAT）
    print("Attaching graph and full X...")
    st_model.attach_full_X(st_adata)
    st_model.attach_graph(st_adata, k=6, spatial_key='spatial')
    
    st_optimizer = torch.optim.Adam(st_model.parameters(), lr=0.003)
    st_trainer = Trainer(st_model, st_optimizer, device='cuda')
    
    st_trainer.train(
        st_loader,
        max_epochs=2000,
        n_epochs_kl_warmup=200,
        print_every=100
    )
    
    # 保存DestVI
    st_trainer.save("checkpoints/destvi.pth")
    print("DestVI training completed!")
    
    # ========== 6. 推理 ==========
    print("\n" + "="*50)
    print("Running inference...")
    print("="*50)
    
    engine = InferenceEngine(st_model, device='cuda')
    
    # 获取比例
    proportions = engine.get_proportions(st_loader, keep_noise=False)
    proportions_df = pd.DataFrame(
        proportions,
        index=st_adata.obs.index,
        columns=sc_dataset.label_names
    )
    proportions_df.to_csv("results/destvi_proportions.csv")
    print(f"Proportions saved! Shape: {proportions_df.shape}")
    
    # 获取gamma
    gamma = engine.get_gamma(st_loader)
    print(f"Gamma shape: {gamma.shape}")
    
    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    main()