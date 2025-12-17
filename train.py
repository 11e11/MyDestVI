"""空间域识别训练脚本 - PCA 输入版本"""
from sklearn.discriminant_analysis import StandardScaler
import torch
import numpy as np
import scanpy as sc
from pathlib import Path
import argparse
import time

from src.modules.spatial_domain_model import SpatialDomainModel
from src.utils.dual_graph_builder import build_dual_graphs
from src.utils.Contrasive_sampler import MultiScaleContrastiveSampler

import logging
from pathlib import Path

# 配置日志
log_path = Path('./results/spatial_domain/training.log')
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

log = logging.info


def train_one_epoch_fullgraph(model, X_input, X_target, contrast_samples, optimizer, epoch, device):
    """
    全图训练一个 epoch
    
    Args:  
        X_input: 编码器输入 (PCA) [N, n_pca]
        X_target:  解码器目标 (PCA) [N, n_pca]  🔥 与 X_input 相同
    """
    model.train()
    
    # 前向传播
    z, decoder_outputs, gate_info = model.forward_all()
    
    # 1.重构损失（MSE）
    recon_loss = model.compute_reconstruction_loss(
        decoder_outputs=decoder_outputs,
        x_target=X_target  # 🔥 目标是 PCA
    ).mean()
    
    # 检查 NaN
    if torch.isnan(recon_loss):
        log("⚠️ 警告：重构损失为 NaN")
        return {
            'total_loss': float('inf'),
            'recon_loss':  float('inf'),
            'contrast_loss': 0.0,
            'local_contrast': 0.0,
            'global_contrast': 0.0,
            **gate_info
        }
    
    # 2.对比损失
    contrast_loss, local_loss, global_loss = model.contrastive_loss_fast(z, contrast_samples)
    contrast_loss = contrast_loss * model.contrast_weight
    
    # 总损失
    total_loss = recon_loss + contrast_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 统计
    metrics = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'contrast_loss': contrast_loss.item(),
        'local_contrast': local_loss.item() if isinstance(local_loss, torch.Tensor) else local_loss,
        'global_contrast': global_loss.item() if isinstance(global_loss, torch.Tensor) else global_loss,
        **gate_info
    }
    
    return metrics

def main(args):
    setup_seed(args.seed)
    log("=" * 80)
    log("🚀 空间域识别训练 - PCA->PCA + MSE + Euclidean + RBF")
    log("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"📱 Device: {device}")
    
    # 1.加载数据
    log("\n📂 加载数据...")
    adata = sc.read_visium(args.adata_path)
    log(f"  原始数据:   {adata.n_obs} spots × {adata.n_vars} genes")
    
    # 保存原始计数
    log("  保存原始计数到 adata.layers['counts']")
    adata.layers['counts'] = adata.X.copy()
    
    # 2.预处理
    log("\n📊 数据预处理...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['normalized_log'] = adata.X.copy()
    
    # 3.筛选高变基因
    if 'highly_variable' not in adata.var.columns:
        log(f"  筛选高变基因 (n_top_genes={args.n_hvg})...")
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor='seurat_v3')
    
    n_hvg = adata.var['highly_variable'].sum()
    log(f"  高变基因数量: {n_hvg}")
    
    # 4.只保留高变基因
    log(f"  只保留高变基因用于训练...")
    adata_hvg = adata[: , adata.var['highly_variable']].copy()
    log(f"  筛选后数据: {adata_hvg.n_obs} spots × {adata_hvg.n_vars} genes")
    
    # 5.PCA 降维（在高变基因上做）
    log(f"\n🔬 在高变基因上执行 PCA 降维到 {args.n_pca} 维...")
    sc.tl.pca(adata_hvg, n_comps=args.n_pca)
    variance_ratio = adata_hvg.uns['pca']['variance_ratio'].sum()
    log(f"  PCA 解释方差比: {variance_ratio:.2%}")
    
    # 🔥 6.标准化 PCA（关键步骤！）
    log(f"\n📐 标准化 PCA...")
    from sklearn.preprocessing import StandardScaler
    
    X_pca = adata_hvg.obsm['X_pca'][: , :args.n_pca].copy()
    
    log(f"  标准化前:")
    log(f"    - 均值: {X_pca.mean(axis=0)[:5]}")
    log(f"    - 标准差:  {X_pca.std(axis=0)[:5]}")
    
    scaler = StandardScaler()
    X_pca_scaled = scaler.fit_transform(X_pca)
    
    log(f"  标准化后:")
    log(f"    - 均值: {X_pca_scaled.mean(axis=0)[:5]}")
    log(f"    - 标准差: {X_pca_scaled.std(axis=0)[:5]}")
    
    # 🔥🔥🔥 关键：将标准化后的 PCA 写回 adata，供构图使用 🔥🔥🔥
    adata_hvg.obsm['X_pca'] = X_pca_scaled.copy()
    adata.obsm['X_pca'] = X_pca_scaled.copy()  # 也写回原始 adata
    
    log(f"  ✅ 标准化 PCA 已写回 adata.obsm['X_pca']")
    
    # 🔥 7.构建双图（使用标准化后的 PCA + Euclidean + RBF）
    log("\n📊 构建双图（基于标准化 PCA + Euclidean + RBF）...")
    graphs = build_dual_graphs(
        adata_hvg,  # 🔥 此时 adata_hvg.obsm['X_pca'] 已经是标准化的
        k_spatial=args.k_spatial,
        k_expr=args.k_expr,
        spatial_key=args.spatial_key,
        use_pca_for_expr=True,  # 🔥 使用 PCA 构建表达图
        n_pcs=args.n_pca
    )
    
    # 8.初始化对比学习采样器
    log("\n🎯 初始化对比学习采样器...")
    sampler = MultiScaleContrastiveSampler(
        adata_hvg,
        spatial_key=args.spatial_key,
        expr_layer='normalized_log',
        k_spatial=args.k_spatial,
        local_pos=args.local_pos,
        local_neg=args.local_neg,
        global_pos=args.global_pos,
        global_neg1=args.global_neg1,
        global_neg2=args.global_neg2,
        local_sim_percentile=args.local_sim_percentile
    )
    
    contrast_samples = sampler.get_samples()
    sampler.get_statistics()
    
    # 9.准备训练数据
    log("\n📦 准备训练数据...")
    
    # 🔥 使用标准化后的 PCA
    X_pca_tensor = torch.FloatTensor(X_pca_scaled).to(device)
    
    log(f"  X_pca 形状: {X_pca_tensor.shape}")
    log(f"    - 范围: [{X_pca_tensor.min():.2f}, {X_pca_tensor.max():.2f}]")
    log(f"    - 均值: {X_pca_tensor.mean():.4f}")
    log(f"    - 标准差: {X_pca_tensor.std():.4f}")
    log(f"  🎯 训练目标: 重构标准化的 PCA（MSE 损失）")
    log(f"  🎯 图结构: 基于标准化 PCA + Euclidean + RBF")
    
    # 10.初始化模型
    log("\n🏗️ 初始化模型...")
    model = SpatialDomainModel(
        n_input=args.n_pca,
        n_output=args.n_pca,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gnn_type=args.gnn_type,
        n_heads=args.n_heads,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        reconstruction_loss='gaussian',
        contrast_weight=args.contrast_weight,
        local_contrast_weight=args.local_contrast_weight,
        global_contrast_weight=args.global_contrast_weight,
        temperature=args.temperature,
        contrast_sample_rate=args.contrast_sample_rate
    ).to(device)
    
    # 注册数据和图
    model.attach_full_X(X_pca=X_pca_tensor)
    model.attach_dual_graphs(
        graphs['spatial_edge_index'].to(device),
        graphs['spatial_edge_weight'].to(device),
        graphs['expr_edge_index'].to(device),
        graphs['expr_edge_weight'].to(device)
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  参数量: {n_params:,}")
    
    # 11.优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.lr * 0.01
    )
    
    # 12.训练循环
    log("\n🏋️ 开始训练...")
    log("="*80)
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        
        metrics = train_one_epoch_fullgraph(
            model,
            X_pca_tensor,   # 🔥 输入
            X_pca_tensor,   # 🔥 目标（同一个数据）
            contrast_samples,
            optimizer,
            epoch,
            device
        )
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        metrics['epoch'] = epoch + 1
        metrics['lr'] = scheduler.get_last_lr()[0]
        metrics['time'] = epoch_time
        history.append(metrics)
        
        # 打印日志
        if (epoch + 1) % args.log_every == 0:
            log(f"\n{'='*80}")
            log(f"Epoch {epoch+1}/{args.max_epochs} ({epoch_time:.2f}s)")
            log(f"{'='*80}")
            log(f"  Total Loss:         {metrics['total_loss']:.4f}")
            log(f"  Recon Loss (MSE):  {metrics['recon_loss']:.4f}")
            log(f"  Contrast Loss:    {metrics['contrast_loss']:.4f}")
            log(f"    - Local:          {metrics['local_contrast']:.4f}")
            log(f"    - Global:      {metrics['global_contrast']:.4f}")
            
            if 'gate_spatial_mean' in metrics:
                log(f"  Gate (Independent):")
                log(f"    - Spatial:      {metrics['gate_spatial_mean']:.3f}")
                log(f"    - Expression:  {metrics['gate_expr_mean']:.3f}")
            
            log(f"  LR:                {metrics['lr']:.6f}")
        
        # 保存最佳模型
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            save_path = Path(args.save_dir) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'metrics': metrics,
                'args': vars(args)
            }, save_path)
            
            if (epoch + 1) % args.log_every == 0:
                log(f"  ✅ 保存最佳模型 (loss={best_loss:.4f})")
    
        # 13.提取 latent 并聚类
    log("\n🔍 提取 latent 表示并聚类...")
    model.eval()
    z = model.get_latent_representation()

    adata.obsm['X_spatial_domain'] = z

    # 使用 SEDR 风格的 mclust
    from src.utils.mcluster_clustering import mclust_with_spatial_refinement

    log(f"\n🎯 使用指定的聚类数: {args.n_clusters}, 模型:  {args.mclust_model}")

    try:
        adata = mclust_with_spatial_refinement(
            adata,
            n_clusters=int(args.n_clusters),
            use_rep='X_spatial_domain',
            spatial_key=args.spatial_key,
            key_added='mclust',
            modelNames=args.mclust_model,
            refine_boundary=args.refine_boundary,
            random_seed=args.seed
        )
        
        n_domains = adata.obs['mclust'].nunique()
        log(f"  ✅ 发现 {n_domains} 个空间域")
        
        # 聚类统计
        cluster_counts = adata.obs['mclust'].value_counts().sort_index()
        log(f"  聚类分布:")
        for cluster_id, count in cluster_counts.items():
            log(f"    - 域 {cluster_id}: {count} 个样本")
        
        # 如果有 BIC 信息
        if 'mclust_bic' in adata.uns: 
            log(f"  BIC: {adata.uns['mclust_bic']:.2f}")
        
        # 如果有 uncertainty 信息
        if 'mclust_uncertainty' in adata.obs.columns:
            uncertainty = adata.obs['mclust_uncertainty'].values
            log(f"  平均 uncertainty: {uncertainty.mean():.4f}")
            log(f"  低置信度样本 (>0.1): {(uncertainty > 0.1).sum()} / {len(uncertainty)}")

    except Exception as e:
        log(f"\n❌ mclust 聚类失败: {e}")
        log("\n⚠️ 回退到 Python GMM...")
        
        # 回退方案：使用 sklearn 的 GMM
        from sklearn.mixture import GaussianMixture
        
        gmm = GaussianMixture(n_components=int(args.n_clusters), random_state=args.seed, n_init=10)
        labels = gmm.fit_predict(z)
        probas = gmm.predict_proba(z)
        
        adata.obs['mclust'] = labels.astype(str)
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        adata.obs['mclust_probability'] = probas.max(axis=1)
        adata.uns['mclust_bic'] = -gmm.bic(z)
        
        n_domains = len(np.unique(labels))
        log(f"  ✅ GMM 发现 {n_domains} 个空间域")
        log(f"  BIC: {adata.uns['mclust_bic']:.2f}")
        log(f"  平均后验概率: {probas.max(axis=1).mean():.3f}")

    # 可选：同时运行 Leiden 作为对比
    if hasattr(args, 'also_run_leiden') and args.also_run_leiden:
        log("\n📊 同时运行 Leiden 聚类作为对比...")
        sc.pp.neighbors(adata, use_rep='X_spatial_domain', n_neighbors=15)
        sc.tl.leiden(adata, resolution=args.leiden_resolution)
        n_leiden = adata.obs['leiden'].nunique()
        log(f"  Leiden 发现 {n_leiden} 个域")

    # 保存结果
    output_path = Path(args.save_dir) / 'adata_with_domains.h5ad'
    adata.write_h5ad(output_path)
    log(f"  ✅ 保存结果到 {output_path}")

    # 保存训练历史
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_path = Path(args.save_dir) / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    log(f"  ✅ 保存训练历史到 {history_path}")

    # 保存聚类信息摘要
    summary_path = Path(args.save_dir) / 'clustering_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("聚类结果摘要\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"聚类方法: mclust ({args.mclust_model})\n")
        f.write(f"聚类数量: {n_domains}\n")
        f.write(f"总样本数: {adata.n_obs}\n\n")
        
        f.write("聚类分布:\n")
        cluster_counts = adata.obs['mclust'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = count / adata.n_obs * 100
            f.write(f"  域 {cluster_id}: {count: 4d} 样本 ({percentage: 5.2f}%)\n")
        
        if 'mclust_bic' in adata.uns:
            f.write(f"\nBIC: {adata.uns['mclust_bic']:.2f}\n")
        
        if 'mclust_uncertainty' in adata.obs.columns:
            uncertainty = adata.obs['mclust_uncertainty'].values
            f.write(f"\n不确定性统计:\n")
            f.write(f"  平均:  {uncertainty.mean():.4f}\n")
            f.write(f"  标准差:  {uncertainty.std():.4f}\n")
            f.write(f"  中位数: {np.median(uncertainty):.4f}\n")
            f.write(f"  最大值: {uncertainty.max():.4f}\n")
            f.write(f"  低置信度 (>0.1): {(uncertainty > 0.1).sum()} / {len(uncertainty)}\n")
        
        if 'mclust_probability' in adata.obs.columns:
            proba = adata.obs['mclust_probability'].values
            f.write(f"\n后验概率统计:\n")
            f.write(f"  平均: {proba.mean():.4f}\n")
            f.write(f"  中位数: {np.median(proba):.4f}\n")
            f.write(f"  最小值: {proba.min():.4f}\n")

    log(f"  ✅ 保存聚类摘要到 {summary_path}")

    # 14.可视化
    if args.spatial_key in adata.obsm:
        plot_results(adata, history_df, args, n_domains)

    # 15.计算评估指标（从 metadata.tsv 加载 ground truth）
    log("\n📊 计算评估指标...")

    # 尝试从 metadata.tsv 加载 ground truth
    metadata_path = Path(args.adata_path) / 'metadata.tsv'
    gt_column = None
    has_ground_truth = False

    if metadata_path.exists():
        log(f"  找到 metadata 文件: {metadata_path}")
        
        try:
            # 读取 metadata
            import pandas as pd
            meta = pd.read_csv(metadata_path, sep="\t")
            
            # 设置索引为 barcode
            if 'barcode' in meta.columns:
                meta = meta.set_index("barcode")
                log(f"  metadata 形状: {meta.shape}")
                log(f"  metadata 列:  {list(meta.columns)}")
                
                # 查找可能的 ground truth 列
                gt_candidates = ['layer_guess_reordered', 'layer_guess', 'layer_annotation', 
                                'ground_truth', 'cell_type', 'region', 'manual_annotation']
                
                for candidate in gt_candidates:
                    if candidate in meta.columns:
                        gt_column = candidate
                        log(f"  使用 ground truth 列: '{gt_column}'")
                        
                        # 合并到 adata
                        common_barcodes = adata.obs_names.intersection(meta.index)
                        log(f"  共同 barcode 数: {len(common_barcodes)} / {adata.n_obs}")
                        
                        if len(common_barcodes) > 0:
                            # 将 ground truth 添加到 adata.obs
                            adata.obs[gt_column] = pd.Series(dtype=str)
                            adata.obs.loc[common_barcodes, gt_column] = meta.loc[common_barcodes, gt_column]
                            
                            # 处理 NaN 和空值
                            adata.obs[gt_column] = adata.obs[gt_column].astype(str).replace({
                                "nan": "NA", 
                                "":  "NA", 
                                "None": "NA"
                            })
                            
                            # 统计分布
                            gt_counts = adata.obs[gt_column].value_counts(dropna=False)
                            log(f"  Ground Truth 分布:")
                            for label, count in gt_counts.items():
                                log(f"    - {label}: {count}")
                            
                            has_ground_truth = True
                            break
                        else:
                            log(f"  ⚠️ 警告: barcode 不匹配")
                
                if not has_ground_truth:
                    log(f"  ⚠️ 未找到 ground truth 列")
                    log(f"     尝试的列名: {gt_candidates}")
                    log(f"     可用的列:  {list(meta.columns)}")
            else:
                log(f"  ⚠️ metadata 中没有 'barcode' 列")
                log(f"     可用的列: {list(meta.columns)}")
        
        except Exception as e:
            log(f"  ❌ 读取 metadata 失败: {e}")
            import traceback
            log(traceback.format_exc())

    else:
        log(f"  ℹ️ 未找到 metadata.tsv:  {metadata_path}")
        
        # 回退：检查 adata.obs 中是否已有 ground truth
        gt_candidates = ['layer_guess_reordered', 'layer_guess', 'layer_annotation', 
                        'ground_truth', 'cell_type', 'region']
        
        for candidate in gt_candidates:
            if candidate in adata.obs.columns:
                gt_column = candidate
                has_ground_truth = True
                log(f"  在 adata.obs 中找到 ground truth: '{gt_column}'")
                break

    # 计算评估指标
    if has_ground_truth and gt_column is not None:
        # 过滤有效样本（排除 NA 等无效标签）
        valid_mask = (
            ~adata.obs[gt_column].isna() & 
            (adata.obs[gt_column] != "NA") & 
            (adata.obs[gt_column] != "Unknown") &
            (adata.obs[gt_column] != "")
        )
        
        n_valid = valid_mask.sum()
        n_total = len(adata)
        
        log(f"\n  有效样本:  {n_valid} / {n_total} ({n_valid/n_total*100:.1f}%)")
        
        if n_valid > 0:
            from sklearn.metrics import (
                adjusted_rand_score, 
                normalized_mutual_info_score,
                adjusted_mutual_info_score,
                fowlkes_mallows_score,
                homogeneity_completeness_v_measure,
                confusion_matrix
            )
            
            # 获取预测和真实标签
            y_true = adata.obs.loc[valid_mask, gt_column].astype(str).values
            y_pred = adata.obs.loc[valid_mask, 'mclust'].astype(str).values
            
            # 调试信息
            log(f"\n  [DEBUG] 标签统计:")
            log(f"    - y_true 唯一值: {len(set(y_true))} -> {sorted(set(y_true))}")
            log(f"    - y_pred 唯一值: {len(set(y_pred))} -> {sorted(set(y_pred))}")
            
            # 计算指标
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)
            fmi = fowlkes_mallows_score(y_true, y_pred)
            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
            
            log(f"\n  ✅ 评估结果:")
            log(f"  {'='*60}")
            log(f"  ARI (Adjusted Rand Index):           {ari:.4f}")
            log(f"  NMI (Normalized Mutual Info):        {nmi:.4f}")
            log(f"  AMI (Adjusted Mutual Info):          {ami:.4f}")
            log(f"  FMI (Fowlkes-Mallows Index):         {fmi:.4f}")
            log(f"  Homogeneity:                          {homogeneity:.4f}")
            log(f"  Completeness:                         {completeness:.4f}")
            log(f"  V-measure:                           {v_measure:.4f}")
            log(f"  {'='*60}")
            
            # 聚类分布分析
            log(f"\n  聚类 vs Ground Truth 交叉表:")
            cross_tab = pd.crosstab(
                pd.Series(y_pred, name='Predicted'),
                pd.Series(y_true, name='True')
            )
            log(f"\n{cross_tab}")
            
            # 保存指标到文件
            metrics_path = Path(args.save_dir) / 'evaluation_metrics.txt'
            with open(metrics_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("评估指标\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Ground Truth 列: {gt_column}\n")
                f.write(f"数据路径: {args.adata_path}\n")
                f.write(f"有效样本数: {n_valid} / {n_total} ({n_valid/n_total*100:.1f}%)\n\n")
                
                f.write("Ground Truth 分布:\n")
                gt_counts = pd.Series(y_true).value_counts().sort_index()
                for label, count in gt_counts.items():
                    f.write(f"  {label}: {count}\n")
                
                f.write("\n预测聚类分布:\n")
                pred_counts = pd.Series(y_pred).value_counts().sort_index()
                for label, count in pred_counts.items():
                    f.write(f"  Cluster {label}: {count}\n")
                
                f.write("\n聚类评估指标:\n")
                f.write(f"  ARI (Adjusted Rand Index):           {ari:.4f}\n")
                f.write(f"  NMI (Normalized Mutual Info):        {nmi:.4f}\n")
                f.write(f"  AMI (Adjusted Mutual Info):          {ami:.4f}\n")
                f.write(f"  FMI (Fowlkes-Mallows Index):         {fmi:.4f}\n")
                f.write(f"  Homogeneity:                         {homogeneity:.4f}\n")
                f.write(f"  Completeness:                          {completeness:.4f}\n")
                f.write(f"  V-measure:                           {v_measure:.4f}\n\n")
                
                f.write("\n交叉表:\n")
                f.write(str(cross_tab))
                f.write("\n\n")
                
                # 添加解释
                f.write("\n指标说明:\n")
                f.write("  - ARI: 衡量聚类与真实标签的一致性 (范围: -1 to 1, 越高越好)\n")
                f.write("  - NMI: 衡量聚类与真实标签的互信息 (范围: 0 to 1, 越高越好)\n")
                f.write("  - AMI: 调整后的互信息 (范围: -1 to 1, 越高越好)\n")
                f.write("  - Homogeneity: 每个聚类只包含一个类别 (0 to 1, 越高越好)\n")
                f.write("  - Completeness: 同一类别分配到同一聚类 (0 to 1, 越高越好)\n")
            
            log(f"  ✅ 保存评估指标到 {metrics_path}")
            
            # 保存混淆矩阵
            try:
                # 获取所有可能的标签
                all_true_labels = sorted(set(y_true))
                all_pred_labels = sorted(set(y_pred))
                
                log(f"\n  计算混淆矩阵:")
                log(f"    - True labels ({len(all_true_labels)}): {all_true_labels}")
                log(f"    - Pred labels ({len(all_pred_labels)}): {all_pred_labels}")
                
                # 使用 true labels 作为行，pred labels 作为列
                cm = confusion_matrix(y_true, y_pred, labels=all_true_labels)
                
                log(f"    - 混淆矩阵形状: {cm.shape}")
                
                # 创建 DataFrame
                cm_df = pd.DataFrame(
                    cm, 
                    index=[f'True_{label}' for label in all_true_labels],
                    columns=[f'Pred_{label}' for label in all_pred_labels]
                )
                
                cm_path = Path(args.save_dir) / 'confusion_matrix.csv'
                cm_df.to_csv(cm_path)
                log(f"  ✅ 保存混淆矩阵到 {cm_path}")

            except Exception as e:
                log(f"  ⚠️ 保存混淆矩阵失败: {e}")
                import traceback
                log(traceback.format_exc())
            
            # 可视化
            try:
                plot_evaluation(adata, y_true, y_pred, valid_mask, gt_column, args.save_dir, args.spatial_key)
            except Exception as e:
                log(f"  ⚠️ 评估可视化失败: {e}")
                import traceback
                log(traceback.format_exc())
            
            # 保存到 adata
            adata.uns['evaluation_metrics'] = {
                'ari': float(ari),
                'nmi': float(nmi),
                'ami': float(ami),
                'fmi': float(fmi),
                'homogeneity': float(homogeneity),
                'completeness':  float(completeness),
                'v_measure': float(v_measure),
                'n_valid_samples': int(n_valid),
                'n_total_samples': int(n_total),
                'ground_truth_column': gt_column
            }
            
            # 重新保存 adata
            adata.write_h5ad(output_path)
        
        else:
            log(f"  ⚠️ 警告: 没有有效的样本用于评估")

    else:
        log(f"  ℹ️ 未找到 ground truth，跳过评估")
        log(f"  💡 提示: 确保数据目录包含 metadata.tsv 文件")
        log(f"     期望路径: {Path(args.adata_path) / 'metadata.tsv'}")

    log("\n✨ 所有任务完成！")


# 辅助函数（在 main 函数外部定义）
def plot_evaluation(adata, y_true, y_pred, valid_mask, gt_column, save_dir, spatial_key='spatial'):
    """生成评估可视化图"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    
    log("  生成评估可视化...")
    
    # 创建包含有效样本的子集
    adata_valid = adata[valid_mask].copy()
    adata_valid.obs['ground_truth'] = y_true
    
    # 计算 ARI
    ari = adjusted_rand_score(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1.Ground Truth
    try:
        if spatial_key in adata_valid.obsm:
            sc.pl.spatial(adata_valid, color='ground_truth', ax=axes[0], show=False,
                         title=f'Ground Truth ({gt_column})', 
                         legend_loc='right margin', size=1.5, frameon=False)
        else:
            # 回退到 UMAP
            if 'X_umap' not in adata_valid.obsm:
                sc.pp.neighbors(adata_valid, use_rep='X_spatial_domain', n_neighbors=15)
                sc.tl.umap(adata_valid, min_dist=0.3)
            sc.pl.umap(adata_valid, color='ground_truth', ax=axes[0], show=False,
                      title=f'Ground Truth ({gt_column})', frameon=False)
    except Exception as e:
        log(f"    ⚠️ Ground Truth 可视化失败:  {e}")
        axes[0].text(0.5, 0.5, f'Visualization failed:\n{str(e)[:50]}', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # 2.Prediction
    try:
        if spatial_key in adata_valid.obsm:
            sc.pl.spatial(adata_valid, color='mclust', ax=axes[1], show=False,
                         title=f'Prediction (ARI={ari:.3f})', 
                         legend_loc='right margin', size=1.5, frameon=False)
        else:
            sc.pl.umap(adata_valid, color='mclust', ax=axes[1], show=False,
                      title=f'Prediction (ARI={ari:.3f})', frameon=False)
    except Exception as e:
        log(f"    ⚠️ Prediction 可视化失败: {e}")
        axes[1].text(0.5, 0.5, f'Visualization failed:\n{str(e)[:50]}', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    # 3.指标对比柱状图
    from sklearn.metrics import (
        adjusted_rand_score, 
        normalized_mutual_info_score,
        adjusted_mutual_info_score,
        fowlkes_mallows_score,
        homogeneity_completeness_v_measure
    )
    
    ari_val = adjusted_rand_score(y_true, y_pred)
    nmi_val = normalized_mutual_info_score(y_true, y_pred)
    ami_val = adjusted_mutual_info_score(y_true, y_pred)
    fmi_val = fowlkes_mallows_score(y_true, y_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
    
    metrics_names = ['ARI', 'NMI', 'AMI', 'FMI', 'Homog.', 'Compl.', 'V-meas.']
    metrics_values = [ari_val, nmi_val, ami_val, fmi_val, homogeneity, completeness, v_measure]
    
    colors = ['#1f77b4' if v >= 0.5 else '#ff7f0e' for v in metrics_values]
    bars = axes[2].bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='0.5 threshold')
    axes[2].set_ylim([0, 1.05])
    axes[2].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[2].set_title('Clustering Evaluation Metrics', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 添加数值标签
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    fig_path = Path(save_dir) / 'evaluation_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    log(f"  ✅ 保存评估对比图到 {fig_path}")
    plt.close()
    
    # 绘制混淆矩阵热图
    plot_confusion_matrix_heatmap(y_true, y_pred, save_dir)


def plot_confusion_matrix_heatmap(y_true, y_pred, save_dir):
    """绘制混淆矩阵热图"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import numpy as np
    
    log("  生成混淆矩阵热图...")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 获取标签
    true_labels = sorted(set(y_true))
    pred_labels = sorted(set(y_pred))
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 1.原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pred_labels, yticklabels=true_labels,
                ax=axes[0], cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # 2.归一化
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=pred_labels, yticklabels=true_labels,
                ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'Proportion'}, 
                linewidths=0.5, linecolor='gray')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = Path(save_dir) / 'confusion_matrix_heatmap.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    log(f"  ✅ 保存混淆矩阵热图到 {fig_path}")
    plt.close()


def plot_results(adata, history_df, args, n_domains):
    """生成可视化"""
    import matplotlib.pyplot as plt
    
    log("\n📊 生成可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1.空间域分布
    try:
        sc.pl.spatial(adata, color='mclust', ax=axes[0, 0], show=False, 
                     title=f'Spatial Domains (mclust, n={n_domains})', 
                     legend_loc='right margin', size=1.5)
    except Exception as e: 
        log(f"  ⚠️ 空间可视化失败: {e}")
        axes[0, 0].text(0.5, 0.5, 'Spatial plot not available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # 2.UMAP
    try:
        sc.pp.neighbors(adata, use_rep='X_spatial_domain', n_neighbors=15)
        sc.tl.umap(adata, min_dist=0.3)
        sc.pl.umap(adata, color='mclust', ax=axes[0, 1], show=False,
                  title='UMAP (Latent Space)', legend_loc='right margin')
    except Exception as e: 
        log(f"  ⚠️ UMAP 失败: {e}")
        axes[0, 1].text(0.5, 0.5, f'UMAP failed:  {str(e)[:50]}', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3.训练曲线
    axes[1, 0].plot(history_df['epoch'], history_df['total_loss'], 
                   label='Total', linewidth=2, color='black')
    axes[1, 0].plot(history_df['epoch'], history_df['recon_loss'], 
                   label='Recon', alpha=0.7, color='blue')
    axes[1, 0].plot(history_df['epoch'], history_df['contrast_loss'], 
                   label='Contrast', alpha=0.7, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'Training Curves ({args.reconstruction_loss.upper()})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 4.门控权重
    if 'gate_spatial_mean' in history_df.columns:
        axes[1, 1].plot(history_df['epoch'], history_df['gate_spatial_mean'], 
                       label='Spatial Gate', marker='o', markersize=2, linewidth=2, color='blue')
        axes[1, 1].plot(history_df['epoch'], history_df['gate_expr_mean'], 
                       label='Expression Gate', marker='s', markersize=2, linewidth=2, color='orange')
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balance')
        axes[1, 1].set_ylabel('Gate Value (Sigmoid)')
        axes[1, 1].set_title('Independent Gated Fusion')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = Path(args.save_dir) / 'results_summary.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    log(f"  ✅ 保存可视化到 {fig_path}")
    plt.close()
    
    # 聚类不确定性图
    if 'mclust_uncertainty' in adata.obs.columns:
        plot_uncertainty(adata, args.save_dir, args.spatial_key)


def plot_uncertainty(adata, save_dir, spatial_key='spatial'):
    """绘制聚类不确定性"""
    import matplotlib.pyplot as plt
    
    log("  生成不确定性可视化...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1.空间分布
    try:
        sc.pl.spatial(adata, color='mclust_uncertainty', ax=axes[0], show=False,
                     title='Clustering Uncertainty', cmap='YlOrRd', 
                     vmin=0, vmax=adata.obs['mclust_uncertainty'].quantile(0.95), size=1.5)
    except: 
        pass
    
    # 2.后验概率
    if 'mclust_probability' in adata.obs.columns:
        try:
            sc.pl.spatial(adata, color='mclust_probability', ax=axes[1], show=False,
                         title='Max Posterior Probability', cmap='RdYlGn',
                         vmin=0.5, vmax=1.0, size=1.5)
        except:
            pass
    
    # 3.不确定性分布
    uncertainty = adata.obs['mclust_uncertainty'].values
    axes[2].hist(uncertainty, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[2].axvline(x=0.05, color='green', linestyle='--', label='Low (0.05)')
    axes[2].axvline(x=0.2, color='orange', linestyle='--', label='High (0.2)')
    axes[2].set_xlabel('Uncertainty')
    axes[2].set_ylabel('Number of Spots')
    axes[2].set_title('Distribution of Uncertainty')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = Path(save_dir) / 'mclust_uncertainty.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    log(f"  ✅ 保存不确定性图到 {fig_path}")
    plt.close()
    log("所有任务完成！")

import random
import os

def setup_seed(seed=2020):
    # 1.Python 自身
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2.Numpy
    np.random.seed(seed)
    
    # 3.Pytorch CPU/GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果有多张卡
    
    # 4.🔥 强制 CuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5.🔥 (可选) PyTorch 某些算子的确定性设置
    # torch.use_deterministic_algorithms(True) # 这行可能会报错，如果某些算子不支持的话先注释掉
    
    # 6.🔥 R 语言种子
    try:
        import rpy2.robjects as ro
        ro.r(f'set.seed({seed})')
    except ImportError:
        pass

    print(f"🔒 全局随机种子已固定: {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='空间域识别 - PCA输入 + NB损失')
        
    # 数据
    parser.add_argument('--adata_path', type=str, required=True)
    parser.add_argument('--spatial_key', type=str, default='spatial')
    parser.add_argument('--save_dir', type=str, default='./results/spatial_domain')
    
    # 高变基因和 PCA
    parser.add_argument('--n_hvg', type=int, default=3000)
    parser.add_argument('--n_pca', type=int, default=50)
    
    # 图构建
    parser.add_argument('--k_spatial', type=int, default=10) 
    parser.add_argument('--k_expr', type=int, default=15)
    
    # 对比学习采样
    parser.add_argument('--local_pos', type=int, default=5)
    parser.add_argument('--local_neg', type=int, default=0)
    parser.add_argument('--global_pos', type=int, default=5)
    parser.add_argument('--global_neg1', type=int, default=5)
    parser.add_argument('--global_neg2', type=int, default=5)
    parser.add_argument('--local_sim_percentile', type=int, default=70)
    
    # 模型架构
    parser.add_argument('--gnn_type', type=str, default='gine')
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=16) 
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fusion_type', type=str, default='independent')
    
    # 重构损失
    parser.add_argument('--reconstruction_loss', type=str, default='gaussian')  
    
    # 训练
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--log_every', type=int, default=10)
    
    # 损失权重
    parser.add_argument('--contrast_weight', type=float, default=5) 
    parser.add_argument('--local_contrast_weight', type=float, default=2.0)
    parser.add_argument('--global_contrast_weight', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.3)  
    parser.add_argument('--contrast_sample_rate', type=float, default=0.3)
    
    # 聚类
    parser.add_argument('--n_clusters', type=str, default='7')
    parser.add_argument('--mclust_model', type=str, default='EEE')
    parser.add_argument('--refine_boundary', action='store_true')
    parser.add_argument('--seed', type=int, default=2020)
    
    args = parser.parse_args()
    
    main(args)