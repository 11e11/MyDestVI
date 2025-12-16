"""空间域识别训练脚本 - 全图训练版本 + 多种重构损失"""
import torch
import numpy as np
import scanpy as sc
from pathlib import Path
import argparse
import time

from src.modules.spatial_domain_model import SpatialDomainModel
from src.utils.dual_graph_builder import build_dual_graphs
from src.utils.Contrasive_sampler import MultiScaleContrastiveSampler
# from src.utils.mcluster_clustering import (
#     # mclust_clustering, 
#     # select_best_mclust,
#     # mclust_with_refinement,
#     # check_rpy2_available
# )


import logging
from pathlib import Path

# 配置日志
log_path = Path('./results/spatial_domain/training.log')
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),  # mode='a' 为追加模式
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 创建快捷函数
log = logging.info



def train_one_epoch_fullgraph(model, X_log, X_raw, contrast_samples, optimizer, epoch, device):
    """
    全图训练一个 epoch
    
    Args:
        X_log: log变换数据 (编码器输入)
        X_raw: 原始计数数据 (重构损失)
    """
    model.train()
    
    # 前向传播
    z, decoder_outputs, gate_info = model.forward_all()
    
    # 1. 重构损失 🔥 传入两份数据
    recon_loss = model.compute_reconstruction_loss(
        x_log=X_log, 
        decoder_outputs=decoder_outputs,
        x_raw=X_raw  # 🔥 原始数据
    ).mean()
    
    # 检查 NaN
    if torch.isnan(recon_loss):
        log("⚠️ 警告：重构损失为 NaN")
        return {
            'total_loss': float('inf'),
            'recon_loss': float('inf'),
            # 'kl_loss': 0.0,
            'contrast_loss': 0.0,
            'local_contrast':  0.0,
            'global_contrast': 0.0,
            **gate_info  # 🔥 直接合并，不加前缀
        }
    
    # 2. KL 损失
    # kl_loss = torch.mean(z ** 2) * model.kl_weight
    # kl_loss = -0.5 * torch.mean(1 + logvar - mu. pow(2) - logvar.exp()) * model.kl_weight
    
    # 3. 对比损失
    contrast_loss, local_loss, global_loss = model.contrastive_loss_fast(z, contrast_samples)
    contrast_loss = contrast_loss * model.contrast_weight
    
    # 总损失
    # total_loss = recon_loss + kl_loss + contrast_loss
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
        # 'kl_loss': kl_loss.item(),
        'contrast_loss': contrast_loss.item(),
        'local_contrast': local_loss.item(),
        'global_contrast': global_loss.item(),
        **gate_info  # 🔥 直接合并，不加前缀
    }
    
    # 分布特定统计
    if model.reconstruction_loss in ['nb', 'zinb']:
        px_r = torch.exp(model.decoder. px_r)
        metrics['px_r_mean'] = px_r.mean().item()
        metrics['px_r_std'] = px_r.std().item()
    
    if model.reconstruction_loss == 'zinb':
        px_dropout_prob = torch.sigmoid(decoder_outputs['px_dropout'])
        metrics['dropout_mean'] = px_dropout_prob. mean().item()
    
    if model.reconstruction_loss == 'gaussian':
        var = decoder_outputs['var']
        metrics['var_mean'] = var.mean().item()
    
    return metrics

def main(args):
    log("=" * 80)
    log("🚀 空间域识别训练 - 全图模式（门控融合）")
    log("=" * 80)

    # 检查 mclust 是否可用
    # available, error = check_rpy2_available()
    # if not available:
    #     print(f"\n⚠️ 警告:  {error}")
    #     print("   将使用 Python 的 GMM 作为替代")
    #     use_true_mclust = False
    # else:
    #     use_true_mclust = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"📱 Device: {device}")
    
    # 1. 加载数据
    log("\n📂 加载数据...")
    adata = sc.read_visium(args.adata_path)

    # 🔥 保存原始计数(在任何预处理之前)
    if 'counts' not in adata.layers:
        log("  保存原始计数到 adata.layers['counts']")
        adata.layers['counts'] = adata.X.copy()
    
    # 预处理
    if 'normalized_log' not in adata.layers:
        log("  执行预处理...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.layers['normalized_log'] = adata.X.copy()
    
    # 修改后 (建议方案)
    if 'highly_variable' not in adata.var.columns:
        # 指定 layer='counts'，让它去算原始数据
        sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor='seurat_v3', layer='counts')

    # 方案A：直接取子集（最简单，推荐）
    adata = adata[:, adata.var['highly_variable']].copy()
    
    log(f"  样本数:  {adata.n_obs}, 基因数: {adata.n_vars}")
    
    # 2. 构建双图
    log("\n📊 构建双图...")
    graphs = build_dual_graphs(
        adata,
        k_spatial=args.k_spatial,
        k_expr=args.k_expr,
        spatial_key=args.spatial_key,
        encoder_layer='normalized_log'
    )
    
    # 3. 初始化对比学习采样器
    log("\n🎯 初始化对比学习采样器...")
    sampler = MultiScaleContrastiveSampler(
        adata,
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
    
    # 4. 准备数据
    log("\n📦 准备训练数据...")
    from scipy.sparse import issparse
    
    # 🔥 1. 提取原始计数数据
    if 'counts' in adata.layers:
        X_raw = adata.layers['counts']
        log("  ✅ 使用 adata.layers['counts'] 作为原始数据")
    else:
        # 回退方案:如果没有保存counts,尝试从X恢复
        log("  ⚠️ 未找到 adata.layers['counts']")
        if 'normalized_log' in adata.layers:
            # 如果有log数据,假设原始数据在某处
            if issparse(adata.X):
                X_raw = adata.X.toarray()
            else:
                X_raw = adata.X.copy()
            log("  ⚠️ 使用 adata.X 作为原始数据(可能已归一化,请检查)")
        else:
            raise ValueError("无法找到原始计数数据!请确保 adata.layers['counts'] 存在")
    
    # 转换为tensor
    if issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw_tensor = torch.FloatTensor(np.asarray(X_raw, dtype=np.float32)).to(device)
    
    # 🔥 2. log变换数据(编码器输入)
    X_log = adata.layers['normalized_log']
    if issparse(X_log):
        X_log = X_log.toarray()
    X_log_tensor = torch.FloatTensor(np.asarray(X_log, dtype=np.float32)).to(device)
    
    log(f"  X_raw 形状:  {X_raw_tensor.shape}, 范围: [{X_raw_tensor.min():.2f}, {X_raw_tensor.max():.2f}]")
    log(f"  X_log 形状:  {X_log_tensor.shape}, 范围: [{X_log_tensor.min():.2f}, {X_log_tensor.max():.2f}]")
    
    # 🔥 验证数据类型
    if args.reconstruction_loss in ['nb', 'zinb', 'poisson']:
        # 检查是否为整数
        is_integer = torch.all(X_raw_tensor == X_raw_tensor.floor())
        if not is_integer:
            log(f"  ⚠️ 警告: 原始数据不是整数! 前10个值: {X_raw_tensor[0, :10]}")
            log(f"     {args.reconstruction_loss.upper()} 损失需要整数计数数据")
    
    # 5. 初始化模型
    log("\n🏗️ 初始化模型...")
    model = SpatialDomainModel(
        n_genes=adata.n_vars,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gnn_type=args.gnn_type,
        n_heads=args.n_heads,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        reconstruction_loss=args.reconstruction_loss,
        kl_weight=args.kl_weight,
        contrast_weight=args.contrast_weight,
        local_contrast_weight=args.local_contrast_weight,
        global_contrast_weight=args.global_contrast_weight,
        temperature=args.temperature,
        contrast_sample_rate=args.contrast_sample_rate
    ).to(device)
    
    # 注册图和数据
    model.attach_full_X(X=X_log_tensor, X_raw=X_raw_tensor)
    model.attach_dual_graphs(
        graphs['spatial_edge_index']. to(device),
        graphs['spatial_edge_weight'].to(device),
        graphs['expr_edge_index'].to(device),
        graphs['expr_edge_weight'].to(device)
    )
    # 🔥 添加检查
    log(f"\n🔍 模型架构检查:")
    log(f"  - Fusion type: {args.fusion_type}")
    log(f"  - Encoder fusion type: {model.encoder.gated_fusion. fusion_type}")

    # 检查是否有 residual_weight
    if hasattr(model. encoder. gated_fusion, 'residual_weight'):
        log(f"  - Has residual_weight: ✅ (初始值: {model.encoder.gated_fusion.residual_weight. item():.3f})")
    else:
        log(f"  - Has residual_weight: ❌")


    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  参数量: {n_params:,}")
    
    # 6. 优化器
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
    
    # 7. 训练循环
    log("\n🏋️ 开始训练...")
    log("="*80)
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        
        metrics = train_one_epoch_fullgraph(
            model, 
            X_log_tensor,   # log数据
            X_raw_tensor,   # 原始数据
            contrast_samples, 
            optimizer, 
            epoch, 
            device
        )
        
        scheduler.step()
        epoch_time = time. time() - epoch_start
        
        # 记录
        metrics['epoch'] = epoch + 1
        metrics['lr'] = scheduler.get_last_lr()[0]
        metrics['time'] = epoch_time
        history. append(metrics)
        
        # 在训练循环中打印
        if (epoch + 1) % args.log_every == 0:
            log(f"\n{'='*80}")
            log(f"Epoch {epoch+1}/{args. max_epochs} ({epoch_time:.2f}s)")
            log(f"{'='*80}")
            log(f"  Total Loss:       {metrics['total_loss']:.4f}")
            log(f"  Recon Loss:      {metrics['recon_loss']:.4f}")
            # log(f"  KL Loss:         {metrics['kl_loss']:.6f}")
            log(f"  Contrast Loss:   {metrics['contrast_loss']:.4f}")
            log(f"    - Local:        {metrics['local_contrast']:.4f}")
            log(f"    - Global:      {metrics['global_contrast']:.4f}")
            
            # 🔥 调试：先打印所有 gate_ 相关的键
            gate_keys = {k: v for k, v in metrics. items() if k.startswith('gate_') or 'weight' in k}
            if gate_keys:
                log(f"  [DEBUG] Gate keys found: {list(gate_keys.keys())}")
                for k, v in gate_keys.items():
                    log(f"    {k}: {v}")
            
            # 🔥 修复：根据 fusion_type 直接判断，而不是依赖键名
            # 优先级：Independent > Adaptive > Simple
            
            if 'gate_spatial_mean' in metrics:
                # Independent 模式（两个独立门控）
                log(f"  Gate (Independent):")
                log(f"    - Spatial:      {metrics['gate_spatial_mean']:.3f}")
                log(f"    - Expression:  {metrics['gate_expr_mean']:.3f}")
            
            elif 'gate_mean' in metrics and 'gate_residual_weight' in metrics: 
                # 🔥 Adaptive 模式（门控 + 残差）
                log(f"  Gate (Adaptive):")
                log(f"    - Gate Mean:   {metrics['gate_mean']:.3f}")
                log(f"    - Gate Std:    {metrics. get('gate_std', 0.0):.3f}")
                log(f"    - Residual:     {metrics['residual_weight']:.3f}")
                if 'spatial_weight' in metrics:
                    log(f"    - Spatial:     {metrics['spatial_weight']:.3f}")
                    log(f"    - Expression:   {metrics['expr_weight']:.3f}")
            
            elif 'gate_mean' in metrics: 
                # Simple 模式（单一门控）
                log(f"  Gate (Simple):")
                log(f"    - Mean:        {metrics['gate_mean']:.3f}")
                if 'gate_std' in metrics:
                    log(f"    - Std:         {metrics['gate_std']:.3f}")
                if 'spatial_weight' in metrics:
                    log(f"    - Spatial:     {metrics['spatial_weight']:.3f}")
                    log(f"    - Expression:  {metrics['expr_weight']:.3f}")
            
            else:
                log(f"  ⚠️ 警告：未找到门控信息")
            
            # 分布特定统计
            if 'px_r_mean' in metrics:
                log(f"  Dispersion (θ):  {metrics['px_r_mean']:.3f} ± {metrics['px_r_std']:.3f}")
            if 'dropout_mean' in metrics:
                log(f"  Dropout Prob:    {metrics['dropout_mean']:.3f}")
            if 'var_mean' in metrics:
                log(f"  Variance:        {metrics['var_mean']:.3f}")
            
            log(f"  LR:               {metrics['lr']:.6f}")
        
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
    
    log("\n" + "="*80)
    log("🎉 训练完成！")
    log("="*80)
    
    # 8. 提取 latent 并聚类
    log("\n🔍 提取 latent 表示并聚类...")
    model.eval()
    z = model.get_latent_representation()

    adata. obsm['X_spatial_domain'] = z

    # 🔥 使用 SEDR 风格的 mclust
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
            log(f"  BIC: {adata.uns['mclust_bic']:. 2f}")
        
        # 如果有 uncertainty 信息
        if 'mclust_uncertainty' in adata.obs. columns:
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
        
        adata.obs['mclust'] = labels. astype(str)
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        adata.obs['mclust_probability'] = probas. max(axis=1)
        adata.uns['mclust_bic'] = -gmm.bic(z)
        
        n_domains = len(np.unique(labels))
        log(f"  ✅ GMM 发现 {n_domains} 个空间域")
        log(f"  BIC: {adata.uns['mclust_bic']:.2f}")
        log(f"  平均后验概率: {probas. max(axis=1).mean():.3f}")

    # 可选：同时运行 Leiden 作为对比
    if args.also_run_leiden:
        log("\n📊 同时运行 Leiden 聚类作为对比...")
        sc.pp.neighbors(adata, use_rep='X_spatial_domain', n_neighbors=15)
        sc.tl.leiden(adata, resolution=args.leiden_resolution)
        n_leiden = adata.obs['leiden']. nunique()
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

    # 🔥 保存聚类信息摘要
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
            f.write(f"\nBIC: {adata. uns['mclust_bic']:.2f}\n")
        
        if 'mclust_uncertainty' in adata.obs.columns:
            uncertainty = adata.obs['mclust_uncertainty'].values
            f.write(f"\n不确定性统计:\n")
            f.write(f"  平均:  {uncertainty.mean():.4f}\n")
            f.write(f"  标准差:  {uncertainty.std():.4f}\n")
            f.write(f"  中位数: {np.median(uncertainty):.4f}\n")
            f.write(f"  最大值:  {uncertainty.max():.4f}\n")
            f.write(f"  低置信度 (>0.1): {(uncertainty > 0.1).sum()} / {len(uncertainty)}\n")
        
        if 'mclust_probability' in adata.obs.columns:
            proba = adata.obs['mclust_probability'].values
            f.write(f"\n后验概率统计:\n")
            f.write(f"  平均: {proba.mean():.4f}\n")
            f.write(f"  中位数: {np.median(proba):.4f}\n")
            f.write(f"  最小值: {proba.min():.4f}\n")

    log(f"  ✅ 保存聚类摘要到 {summary_path}")

    # 9. 可视化
    if args.spatial_key in adata.obsm:
        plot_results(adata, history_df, args, n_domains)

    # 10. 计算评估指标（从 metadata.tsv 加载 ground truth）
    log("\n📊 计算评估指标...")

    # 🔥 尝试从 metadata.tsv 加载 ground truth
    metadata_path = Path(args. adata_path) / 'metadata.tsv'
    gt_column = None
    has_ground_truth = False

    if metadata_path.exists():
        log(f"  找到 metadata 文件: {metadata_path}")
        
        try:
            # 读取 metadata
            import pandas as pd
            meta = pd.read_csv(metadata_path, sep="\t")
            
            # 设置索引为 barcode
            if 'barcode' in meta. columns:
                meta = meta. set_index("barcode")
                log(f"  metadata 形状: {meta.shape}")
                log(f"  metadata 列: {list(meta.  columns)}")
                
                # 🔥 查找可能的 ground truth 列
                gt_candidates = ['layer_guess_reordered', 'layer_guess', 'layer_annotation', 
                            'ground_truth', 'cell_type', 'region', 'manual_annotation']
                
                for candidate in gt_candidates:
                    if candidate in meta.columns:
                        gt_column = candidate
                        log(f"  使用 ground truth 列: '{gt_column}'")
                        
                        # 🔥 合并到 adata
                        # 确保索引对齐
                        common_barcodes = adata.obs_names.intersection(meta.index)
                        log(f"  共同 barcode 数: {len(common_barcodes)} / {adata.n_obs}")
                        
                        if len(common_barcodes) > 0:
                            # 将 ground truth 添加到 adata. obs
                            adata.obs[gt_column] = pd. Series(dtype=str)
                            adata. obs. loc[common_barcodes, gt_column] = meta.loc[common_barcodes, gt_column]
                            
                            # 🔥 处理 NaN 和空值
                            adata.obs[gt_column] = adata.obs[gt_column].astype(str).replace({
                                "nan": "NA", 
                                "": "NA", 
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
                    log(f"     可用的列: {list(meta.columns)}")
            else:
                log(f"  ⚠️ metadata 中没有 'barcode' 列")
                log(f"     可用的列: {list(meta.columns)}")
        
        except Exception as e:
            log(f"  ❌ 读取 metadata 失败: {e}")
            import traceback
            log(traceback.format_exc())

    else:
        log(f"  ℹ️ 未找到 metadata. tsv:  {metadata_path}")
        
        # 🔥 回退：检查 adata. obs 中是否已有 ground truth
        gt_candidates = ['layer_guess_reordered', 'layer_guess', 'layer_annotation', 
                    'ground_truth', 'cell_type', 'region']
        
        for candidate in gt_candidates:
            if candidate in adata.obs.columns:
                gt_column = candidate
                has_ground_truth = True
                log(f"  在 adata.obs 中找到 ground truth: '{gt_column}'")
                break

    # 🔥 计算评估指标
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
        
        # 在评估指标计算部分，修复混淆矩阵
        if n_valid > 0:
            from sklearn.metrics import (
                adjusted_rand_score, 
                normalized_mutual_info_score,
                adjusted_mutual_info_score,
                fowlkes_mallows_score,
                homogeneity_completeness_v_measure,
                confusion_matrix  # 🔥 在这里导入
            )
            
            # 获取预测和真实标签
            y_true = adata.obs. loc[valid_mask, gt_column].astype(str).values
            y_pred = adata.obs.loc[valid_mask, 'mclust']. astype(str).values
            
            # 🔥 调试信息
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
            log(f"  Completeness:                          {completeness:.4f}")
            log(f"  V-measure:                           {v_measure:.4f}")
            log(f"  {'='*60}")
            
            # 🔥 改进的聚类分布分析
            log(f"\n  聚类 vs Ground Truth 交叉表:")
            cross_tab = pd.crosstab(
                pd.Series(y_pred, name='Predicted'),
                pd.Series(y_true, name='True')
            )
            log(f"\n{cross_tab}")
            
            # 保存指标到文件
            metrics_path = Path(args.save_dir) / 'evaluation_metrics. txt'
            with open(metrics_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("评估指标\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Ground Truth 列: {gt_column}\n")
                f.write(f"数据路径: {args. adata_path}\n")
                f.write(f"有效样本数: {n_valid} / {n_total} ({n_valid/n_total*100:.1f}%)\n\n")
                
                f.write("Ground Truth 分布:\n")
                gt_counts = pd.Series(y_true).value_counts().sort_index()
                for label, count in gt_counts.items():
                    f.write(f"  {label}: {count}\n")
                
                f. write("\n预测聚类分布:\n")
                pred_counts = pd.Series(y_pred).value_counts().sort_index()
                for label, count in pred_counts. items():
                    f.write(f"  Cluster {label}: {count}\n")
                
                f.write("\n聚类评估指标:\n")
                f.write(f"  ARI (Adjusted Rand Index):           {ari:.4f}\n")
                f.write(f"  NMI (Normalized Mutual Info):        {nmi:.4f}\n")
                f.write(f"  AMI (Adjusted Mutual Info):          {ami:.4f}\n")
                f.write(f"  FMI (Fowlkes-Mallows Index):         {fmi:.4f}\n")
                f.write(f"  Homogeneity:                         {homogeneity:.4f}\n")
                f.write(f"  Completeness:                          {completeness:.4f}\n")
                f.write(f"  V-measure:                           {v_measure:.4f}\n\n")
                
                f. write("\n交叉表:\n")
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
            
            # 🔥 修复混淆矩阵：确保标签对齐
            # 在混淆矩阵部分，修复列名
            try:
                # 获取所有可能的标签
                all_true_labels = sorted(set(y_true))
                all_pred_labels = sorted(set(y_pred))
                
                log(f"\n  计算混淆矩阵:")
                log(f"    - True labels ({len(all_true_labels)}): {all_true_labels}")
                log(f"    - Pred labels ({len(all_pred_labels)}): {all_pred_labels}")
                
                # 🔥 使用 true labels 作为行和列（确保对齐）
                cm = confusion_matrix(y_true, y_pred, labels=all_true_labels)
                
                log(f"    - 混淆矩阵形状: {cm.shape}")
                
                # 🔥 创建 DataFrame：行和列都使用 true labels
                cm_df = pd. DataFrame(
                    cm, 
                    index=[f'True_{label}' for label in all_true_labels],
                    columns=[f'Pred_{label}' for label in all_true_labels]  # 🔥 修复：都用 true labels
                )
                
                cm_path = Path(args.save_dir) / 'confusion_matrix. csv'
                cm_df. to_csv(cm_path)
                log(f"  ✅ 保存混淆矩阵到 {cm_path}")

            except Exception as e: 
                log(f"  ⚠️ 保存混淆矩阵失败: {e}")
                import traceback
                log(traceback.format_exc())
            
            # 可视化
            try:
                plot_evaluation(adata, y_true, y_pred, valid_mask, gt_column, args. save_dir, args.spatial_key)
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
                'completeness':   float(completeness),
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

def plot_evaluation(adata, y_true, y_pred, valid_mask, gt_column, save_dir, spatial_key='spatial'):
    """生成评估可视化图"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    
    log("  生成评估可视化...")
    
    # 创建包含有效样本的子集
    adata_valid = adata[valid_mask]. copy()
    adata_valid.obs['ground_truth'] = y_true
    
    # 计算 ARI
    ari = adjusted_rand_score(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Ground Truth
    try:
        if spatial_key in adata_valid. obsm:
            sc.pl.spatial(adata_valid, color='ground_truth', ax=axes[0], show=False,
                         title=f'Ground Truth ({gt_column})', 
                         legend_loc='right margin', size=1.5, frameon=False)
        else:
            # 回退到 UMAP
            if 'X_umap' not in adata_valid.obsm:
                sc.tl.umap(adata_valid, min_dist=0.3)
            sc.pl.umap(adata_valid, color='ground_truth', ax=axes[0], show=False,
                      title=f'Ground Truth ({gt_column})', frameon=False)
    except Exception as e:
        log(f"    ⚠️ Ground Truth 可视化失败:  {e}")
        axes[0].text(0.5, 0.5, f'Visualization failed:\n{str(e)[: 50]}', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # 2. Prediction
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
    
    # 3. 指标对比柱状图
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
    
    metrics_names = ['ARI', 'NMI', 'AMI', 'FMI', 'Homog. ', 'Compl.', 'V-meas. ']
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
    
    # 1. 原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pred_labels, yticklabels=true_labels,
                ax=axes[0], cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # 2. 归一化
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
    
    # 1. 空间域分布
    try:
        sc.pl.spatial(adata, color='mclust', ax=axes[0, 0], show=False, 
                     title=f'Spatial Domains (mclust, n={n_domains})', 
                     legend_loc='right margin', size=1.5)
    except Exception as e: 
        log(f"  ⚠️ 空间可视化失败: {e}")
        axes[0, 0].text(0.5, 0.5, 'Spatial plot not available', 
                       ha='center', va='center', transform=axes[0, 0]. transAxes)
    
    # 2. UMAP (修复)
    try:
        # 🔥 先运行 neighbors
        sc.pp.neighbors(adata, use_rep='X_spatial_domain', n_neighbors=15)
        sc.tl.umap(adata, min_dist=0.3)
        sc.pl.umap(adata, color='mclust', ax=axes[0, 1], show=False,
                  title='UMAP (Latent Space)', legend_loc='right margin')
    except Exception as e:  
        log(f"  ⚠️ UMAP 失败: {e}")
        axes[0, 1].text(0.5, 0.5, f'UMAP failed: {str(e)[:50]}', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. 训练曲线
    axes[1, 0].plot(history_df['epoch'], history_df['total_loss'], 
                   label='Total', linewidth=2, color='black')
    axes[1, 0].plot(history_df['epoch'], history_df['recon_loss'], 
                   label='Recon', alpha=0.7, color='blue')
    axes[1, 0].plot(history_df['epoch'], history_df['contrast_loss'], 
                   label='Contrast', alpha=0.7, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'Training Curves ({args.reconstruction_loss. upper()})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # 使用对数刻度
    
    # 4. 门控权重（根据融合类型）
    if 'gate_spatial_mean' in history_df.columns:
        # Independent 模式
        axes[1, 1].plot(history_df['epoch'], history_df['gate_spatial_mean'], 
                       label='Spatial Gate', marker='o', markersize=2, linewidth=2, color='blue')
        axes[1, 1].plot(history_df['epoch'], history_df['gate_expr_mean'], 
                       label='Expression Gate', marker='s', markersize=2, linewidth=2, color='orange')
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balance')
        axes[1, 1].set_ylabel('Gate Value (Sigmoid)')
        axes[1, 1].set_title('Independent Gated Fusion')
        axes[1, 1].set_ylim([0, 1])
    
    elif 'gate_mean' in history_df.columns and 'residual_weight' in history_df.columns:
        # 🔥 Adaptive 模式（双Y轴）
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        # 左Y轴：门控值
        line1 = ax1.plot(history_df['epoch'], history_df['gate_mean'], 
                        label='Gate Mean', marker='o', markersize=2, 
                        linewidth=2, color='blue')
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Gate Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim([0, 1])
        
        # 右Y轴：残差权重
        line2 = ax2.plot(history_df['epoch'], history_df['residual_weight'], 
                        label='Residual Weight', marker='s', markersize=2, 
                        linewidth=2, color='red')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Residual Weight', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 1])
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        ax1.set_xlabel('Epoch')
        ax1.set_title('Adaptive Gated Fusion')
        ax1.grid(True, alpha=0.3)
    
    elif 'gate_mean' in history_df. columns and 'gate_std' in history_df. columns:
        # Simple 模式
        axes[1, 1].plot(history_df['epoch'], history_df['gate_mean'], 
                       label='Gate Mean', marker='o', markersize=2, linewidth=2, color='blue')
        axes[1, 1].fill_between(
            history_df['epoch'], 
            history_df['gate_mean'] - history_df['gate_std'],
            history_df['gate_mean'] + history_df['gate_std'],
            alpha=0.3, label='±1 Std', color='blue'
        )
        axes[1, 1]. axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balance')
        axes[1, 1].set_ylabel('Gate Value')
        axes[1, 1].set_title('Simple Gated Fusion')
        axes[1, 1].set_ylim([0, 1])
    
    else:
        # 没有门控信息
        axes[1, 1].text(0.5, 0.5, 'No gating information available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # 统一设置（如果不是 adaptive 已经设置过的）
    if 'residual_weight' not in history_df.columns:
        axes[1, 1].set_xlabel('Epoch')
        if 'gate_spatial_mean' in history_df. columns or 'gate_mean' in history_df.columns:
            axes[1, 1].legend(loc='best')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = Path(args.save_dir) / 'results_summary.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    log(f"  ✅ 保存可视化到 {fig_path}")
    plt.close()
    
    # 额外：聚类不确定性图（如果有）
    if 'mclust_uncertainty' in adata.obs.columns:
        plot_uncertainty(adata, args.save_dir, args.spatial_key)


def plot_uncertainty(adata, save_dir, spatial_key='spatial'):
    """绘制聚类不确定性"""
    import matplotlib.pyplot as plt
    
    log("  生成不确定性可视化...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 空间分布
    try:
        sc.pl.spatial(adata, color='mclust_uncertainty', ax=axes[0], show=False,
                     title='Clustering Uncertainty', cmap='YlOrRd', 
                     vmin=0, vmax=adata.obs['mclust_uncertainty'].quantile(0.95), size=1.5)
    except: 
        pass
    
    # 2. 后验概率（如果有）
    if 'mclust_probability' in adata.obs.columns:
        try:
            sc.pl.spatial(adata, color='mclust_probability', ax=axes[1], show=False,
                         title='Max Posterior Probability', cmap='RdYlGn',
                         vmin=0.5, vmax=1.0, size=1.5)
        except:
            pass
    
    # 3. 不确定性分布
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
    fig_path = Path(save_dir) / 'mclust_uncertainty. png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    log(f"  ✅ 保存不确定性图到 {fig_path}")
    plt.close()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='空间域识别 - 全图训练（门控融合）')
    
    # 数据
    parser.add_argument('--adata_path', type=str, required=True, help='AnnData 文件路径')
    parser.add_argument('--spatial_key', type=str, default='spatial', help='空间坐标 key')
    parser.add_argument('--save_dir', type=str, default='./results/spatial_domain', help='保存目录')
    
    # 图构建
    parser.add_argument('--k_spatial', type=int, default=6, help='空间图邻居数')
    parser.add_argument('--k_expr', type=int, default=10, help='表达图邻居数')
    
    # 对比学习采样
    parser.add_argument('--local_pos', type=int, default=3, help='局部正样本数')
    parser.add_argument('--local_neg', type=int, default=1, help='局部负样本数')
    parser.add_argument('--global_pos', type=int, default=3, help='全局正样本数')
    parser.add_argument('--global_neg1', type=int, default=5, help='全局最不相似负样本数')
    parser.add_argument('--global_neg2', type=int, default=2, help='全局随机负样本数')
    parser.add_argument('--local_sim_percentile', type=int, default=60, 
                       help='局部相似度阈值（百分位数）')
    
    # 模型架构
    parser.add_argument('--gnn_type', type=str, default='gine', 
                       choices=['gine', 'gat', 'gcn'],
                       help='GNN 类型')
    parser.add_argument('--n_heads', type=int, default=4, help='GAT heads (仅GAT有效)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent 维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')
    
    # 🔥 门控融合类型
    parser.add_argument('--fusion_type', type=str, default='independent',
                       choices=['simple', 'independent', 'adaptive'],
                       help='门控融合类型:  simple(单门控), independent(独立门控), adaptive(自适应门控)')
    
    # 重构损失
    parser.add_argument('--reconstruction_loss', type=str, default='nb',
                       choices=['nb', 'zinb', 'poisson', 'gaussian'],
                       help='重构损失分布类型')
    
    # 训练
    parser.add_argument('--max_epochs', type=int, default=400, help='最大 epoch 数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--log_every', type=int, default=1, help='打印频率')
    
    # 损失权重
    parser.add_argument('--kl_weight', type=float, default=1e-4, help='KL 损失权重')
    parser.add_argument('--contrast_weight', type=float, default=20, help='对比损失权重')
    parser.add_argument('--local_contrast_weight', type=float, default=1.0, help='局部对比权重')
    parser.add_argument('--global_contrast_weight', type=float, default=0.5, help='全局对比权重')
    parser.add_argument('--temperature', type=float, default=0.5, help='对比学习温度')
    parser.add_argument('--contrast_sample_rate', type=float, default=0.5, 
                       help='对比学习采样率（每次随机采样的anchor比例）')
    
    
    # 🔥 mclust 相关参数
    parser.add_argument('--n_clusters', type=str, default='7',
                       help='聚类数量:  数字 或 "auto"')
    parser.add_argument('--min_clusters', type=int, default=3)
    parser.add_argument('--max_clusters', type=int, default=15)
    parser.add_argument('--mclust_model', type=str, default='EEE',
                       choices=['EII', 'VII', 'EEI', 'VVI', 'EEE', 'VVV', 'EEV', 'VEV'],
                       help='mclust 模型类型')
    parser.add_argument('--refine_boundary', action='store_true',
                       help='使用空间信息细化聚类边界')
    parser.add_argument('--also_run_leiden', action='store_true')
    parser.add_argument('--leiden_resolution', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2020)
    
    args = parser.parse_args()
    
    main(args)