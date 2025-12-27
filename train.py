"""空间域识别训练 - V2版本（read_visium + metadata.tsv）"""
from matplotlib import pyplot as plt
import torch
import numpy as np
import scanpy as sc
from pathlib import Path
import argparse
import time
import logging
from scipy.sparse import issparse
import pandas as pd
import random
import os

# 导入模型
# # from src.modules.spatial_domain_model import SpatialDomainModelV2
# from src.utils.dual_graph_builder import build_dual_graphs
# from src.utils.mclust_clustering import mclust_with_spatial_refinement


# ========================================
# 随机种子固定函数
# ========================================
def set_seed(seed:  int):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import scanpy as sc
        sc.settings.seed = seed
    except: 
        pass
    
    print(f"🎲 随机种子已设置为: {seed}")


# ========================================
# 日志配置
# ========================================
def setup_logger(save_dir):
    """配置日志（覆盖模式）"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(save_dir) / 'training.log'
    
    if log_file.exists():
        log_file.unlink()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    return logging.getLogger(__name__)


def log(message):
    """便捷的日志函数"""
    logging.info(message)


# ========================================
# 训练函数
# ========================================
from src.modules.spatial_domain_model import SpatialDomainModelGIB  # 新模型


def train_one_epoch_gib(model, X_pca, X_raw, optimizer, epoch, device, warmup_epochs=20):
    """训练一个epoch - GIB版本"""
    model.train()
    
    # 前向传播
    z_anchor, z_learner, decoder_outputs, info = model.forward_all()
    
    # 1.重构损失
    recon_loss = model.compute_reconstruction_loss(decoder_outputs, X_raw).mean()
    
    if torch.isnan(recon_loss):
        log("⚠️ NaN detected in recon_loss")
        return {
            'total_loss': float('inf'),
            'recon_loss': float('inf'),
            'contrastive_loss': 0.0,
            'prototype_loss': 0.0,
            'gib_sparsity_loss': 0.0,  # 🔥 新增
            'in_warmup': True,
            **info
        }
    
    in_warmup = epoch < warmup_epochs
    
    # 🔥 2.GIB稀疏性损失（从第一个epoch就启动）
    gib_sparsity_loss = info['_gib_sparsity_loss_tensor'] * model.gib_sparsity_weight
    
    # 3.对比学习损失（warmup后启动）
    if not in_warmup:
        contrastive_loss = model.compute_contrastive_loss(z_anchor, z_learner)
        contrastive_loss = contrastive_loss * model.contrastive_weight
    else:
        contrastive_loss = torch.tensor(0.0, device=device)
    
    # 4.原型损失（warmup后）
    if in_warmup:
        prototype_loss = torch.tensor(0.0, device=device)
        proto_loss_dict = {'proto_s2p': 0.0, 'proto_compact': 0.0, 'proto_disperse': 0.0}
        # 🔥 warmup期：重构 + GIB稀疏性
        total_loss = recon_loss + gib_sparsity_loss
    
    elif epoch == warmup_epochs:
        # K-Means初始化原型
        log("\n" + "="*80)
        log(f"🎯 Warmup完成！在Epoch {epoch+1}使用K-Means初始化原型...")
        log("="*80)
        
        with torch.no_grad():
            z_init, _, _, _ = model.forward_all()
            z_proto = model.latent_to_prototype(z_init)
            model.prototypes.initialize_with_kmeans(z_proto)
        
        prototype_loss = torch.tensor(0.0, device=device)
        proto_loss_dict = {'proto_s2p': 0.0, 'proto_compact': 0.0, 'proto_disperse': 0.0}
        # 🔥 初始化epoch：重构 + 对比 + GIB
        total_loss = recon_loss + contrastive_loss + gib_sparsity_loss
    
    else:
        prototype_loss, proto_loss_dict, Q = model.compute_prototype_loss(z_anchor)
        prototype_loss = prototype_loss * model.prototype_weight
        # 🔥 完整训练：重构 + 对比 + 原型 + GIB
        total_loss = recon_loss + contrastive_loss + prototype_loss + gib_sparsity_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 🔥 返回完整指标
    metrics = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'contrastive_loss': contrastive_loss.item() if not in_warmup else 0.0,
        'prototype_loss': prototype_loss.item(),
        'gib_sparsity_loss': gib_sparsity_loss.item(),  # 🔥 新增
        'in_warmup':  in_warmup,
        **proto_loss_dict,
        **{k: v for k, v in info.items() if not k.startswith('_')}  # 过滤内部tensor
    }
    
    # 🔥 添加dispersion参数统计
    if model.reconstruction_loss in ['nb', 'zinb']: 
        if hasattr(model, 'px_r'):
            px_r = torch.exp(model.px_r)
            metrics['px_r_mean'] = px_r.mean().item()
            metrics['px_r_std'] = px_r.std().item()
    
    return metrics

# ========================================
# 🎨 可视化函数
# ========================================
def plot_training_curves_gib(history_df, save_dir):
    """绘制训练曲线 - GIB版本"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1.总损失
    ax = axes[0, 0]
    ax.plot(history_df['epoch'], history_df['total_loss'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # 2.重构损失
    ax = axes[0, 1]
    ax.plot(history_df['epoch'], history_df['recon_loss'], linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.grid(True, alpha=0.3)
    
    # 🔥 3.对比学习损失
    ax = axes[0, 2]
    if 'contrastive_loss' in history_df.columns:
        non_warmup = history_df[~history_df.get('in_warmup', False)]
        if len(non_warmup) > 0:
            ax.plot(non_warmup['epoch'], non_warmup['contrastive_loss'], 
                   linewidth=2, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Contrastive Loss')
            ax.set_title('Contrastive Loss (after warmup)')
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No contrastive loss', ha='center', va='center')
        ax.axis('off')
    
    # 4.原型损失
    ax = axes[1, 0]
    if 'prototype_loss' in history_df.columns:
        non_warmup = history_df[~history_df.get('in_warmup', False)]
        if len(non_warmup) > 0 and (non_warmup['prototype_loss'] > 0).any():
            ax.plot(non_warmup['epoch'], non_warmup['prototype_loss'], 
                   linewidth=2, color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Prototype Loss')
            ax.set_title('Prototype Loss')
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No prototype loss', ha='center', va='center')
        ax.axis('off')
    
    # 🔥 5.GIB Keep Rate
    ax = axes[1, 1]
    if 'gib_keep_rate' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['gib_keep_rate'], 
               linewidth=2, color='red')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Keep Rate')
        ax.set_title('GIB Edge Keep Rate')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No GIB stats', ha='center', va='center')
        ax.axis('off')
    
    # 6.学习率
    ax = axes[1, 2]
    ax.plot(history_df['epoch'], history_df['lr'], linewidth=2, color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"  ✅ 保存训练曲线到 {save_path}")


def plot_gib_statistics(history_df, save_dir):
    """🔥 绘制GIB剪枝统计"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1.Keep Rate演化
    ax = axes[0]
    ax.plot(history_df['epoch'], history_df['gib_keep_rate'], 
           linewidth=2, label='Keep Rate')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Target (50%)')
    ax.fill_between(history_df['epoch'], 0.4, 0.6, alpha=0.2, color='green', label='Ideal Range')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Edge Keep Rate')
    ax.set_title('GIB Pruning Ratio Evolution')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2.平均保留概率
    ax = axes[1]
    if 'gib_avg_prob' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['gib_avg_prob'], 
               linewidth=2, color='orange', label='Avg Keep Prob')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Keep Probability')
        ax.set_title('GIB Average Edge Score')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'gib_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"  ✅ 保存GIB统计图到 {save_path}")

def plot_spatial_comparison(adata, save_dir, spatial_key='spatial'):
    """
    绘制空间分布对比图
    
    包括：
    1.Ground Truth空间分布
    2.预测聚类空间分布
    3.两者的混淆矩阵
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import rcParams
    
    # 设置字体
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 10
    
    # 获取空间坐标
    spatial_coords = adata.obsm[spatial_key]
    
    # 检查是否有ground truth
    gt_columns = ['layer_guess_reordered', 'layer_guess', 'ground_truth']
    gt_column = None
    for col in gt_columns:
        if col in adata.obs.columns:
            gt_column = col
            break
    
    if gt_column is None: 
        log("  ⚠️ 没有ground truth，跳过对比可视化")
        return
    
    # 过滤有效样本
    valid_mask = (
        ~adata.obs[gt_column].isna() & 
        (adata.obs[gt_column] != "NA") & 
        (adata.obs[gt_column] != "Unknown")
    )
    
    # 创建图形
    fig = plt.figure(figsize=(20, 6))
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 子图1: Ground Truth
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax1 = plt.subplot(1, 3, 1)
    
    # 定义DLPFC层的标准颜色
    layer_colors = {
        'Layer1': '#F8766D',
        'Layer2': '#B79F00',
        'Layer3': '#00BA38',
        'Layer4': '#00BFC4',
        'Layer5': '#619CFF',
        'Layer6': '#F564E3',
        'WM': '#999999'
    }
    
    # 绘制ground truth
    for layer, color in layer_colors.items():
        mask = valid_mask & (adata.obs[gt_column] == layer)
        if mask.sum() > 0:
            ax1.scatter(
                spatial_coords[mask, 0],
                spatial_coords[mask, 1],
                c=color,
                s=30,
                label=layer,
                alpha=0.8,
                edgecolors='none'
            )
    
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Spatial X')
    ax1.set_ylabel('Spatial Y')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()  # Visium数据通常需要翻转Y轴
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 子图2: 预测聚类
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax2 = plt.subplot(1, 3, 2)
    
    # 获取预测标签
    pred_labels = adata.obs['mclust'].astype(str)
    unique_clusters = sorted(pred_labels.unique())
    
    # 使用tab10配色
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster in enumerate(unique_clusters):
        mask = (pred_labels == cluster)
        if mask.sum() > 0:
            ax2.scatter(
                spatial_coords[mask, 0],
                spatial_coords[mask, 1],
                c=[colors[i]],
                s=30,
                label=f'Cluster {cluster}',
                alpha=0.8,
                edgecolors='none'
            )
    
    ax2.set_title('Predicted Clusters (mclust)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Spatial X')
    ax2.set_ylabel('Spatial Y')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 子图3: 混淆矩阵热图
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax3 = plt.subplot(1, 3, 3)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    
    y_true = adata.obs.loc[valid_mask, gt_column].values
    y_pred = pred_labels[valid_mask].values
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化（按行，显示每个真实类别的分布）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热图
    sns.heatmap(
        cm_normalized,
        annot=cm,  # 显示原始计数
        fmt='d',
        cmap='YlOrRd',
        xticklabels=[f'C{c}' for c in sorted(set(y_pred))],
        yticklabels=sorted(set(y_true)),
        cbar_kws={'label': 'Proportion'},
        ax=ax3
    )
    
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted Cluster')
    ax3.set_ylabel('True Layer')
    
    plt.tight_layout()
    
    # 保存图形
    save_path = Path(save_dir) / 'spatial_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"  ✅ 保存空间对比图到 {save_path}")


def plot_training_curves(history_df, save_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 总损失
    ax = axes[0, 0]
    ax.plot(history_df['epoch'], history_df['total_loss'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # 重构损失
    ax = axes[0, 1]
    ax.plot(history_df['epoch'], history_df['recon_loss'], linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.grid(True, alpha=0.3)
    
    # 原型损失（如果有）
    ax = axes[1, 0]
    if 'prototype_loss' in history_df.columns:
        # 只绘制非warmup期
        non_warmup = history_df[~history_df.get('in_warmup', False)]
        if len(non_warmup) > 0:
            ax.plot(non_warmup['epoch'], non_warmup['prototype_loss'], 
                   linewidth=2, color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Prototype Loss')
            ax.set_title('Prototype Loss (after warmup)')
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No prototype loss', ha='center', va='center')
        ax.axis('off')
    
    # 学习率
    ax = axes[1, 1]
    ax.plot(history_df['epoch'], history_df['lr'], linewidth=2, color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"  ✅ 保存训练曲线到 {save_path}")


def plot_gate_weights(history_df, save_dir):
    """绘制门控权重变化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'gate_spatial' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['gate_spatial'], 
               label='Spatial', linewidth=2)
    
    if 'gate_learned' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['gate_learned'], 
               label='Learned Graph', linewidth=2)
    
    if 'gate_expr' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['gate_expr'], 
               label='Expression', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gate Weight', fontsize=12)
    ax.set_title('Gating Weights Evolution', fontsize=14, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'gate_weights.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"  ✅ 保存门控权重图到 {save_path}")


def plot_cluster_distribution(adata, save_dir):
    """绘制聚类分布条形图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ground truth分布
    ax = axes[0]
    gt_columns = ['layer_guess_reordered', 'layer_guess', 'ground_truth']
    gt_column = None
    for col in gt_columns:
        if col in adata.obs.columns:
            gt_column = col
            break
    
    if gt_column: 
        gt_counts = adata.obs[gt_column].value_counts().sort_index()
        gt_counts = gt_counts[gt_counts.index != 'NA']
        
        ax.bar(range(len(gt_counts)), gt_counts.values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(gt_counts)))
        ax.set_xticklabels(gt_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Ground Truth Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(gt_counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom')
    
    # 预测聚类分布
    ax = axes[1]
    pred_counts = adata.obs['mclust'].value_counts().sort_index()
    
    ax.bar(range(len(pred_counts)), pred_counts.values, color='coral', alpha=0.7)
    ax.set_xticks(range(len(pred_counts)))
    ax.set_xticklabels([f'C{c}' for c in pred_counts.index], rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Cluster Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(pred_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'cluster_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"  ✅ 保存聚类分布图到 {save_path}")

# ========================================
# 主函数
# ========================================
def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logger(args.save_dir)
    
    log("=" * 80)
    log("🚀 空间域识别训练 - V2版本（原型学习 + Warmup）")
    log("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"📱 Device: {device}")
    
    # ========================================
    # 1.🔥 加载数据（使用 read_visium）
    # ========================================
    log(f"\n📂 加载数据:  {args.adata_path}")
    
    # 🔥 使用 read_visium 读取 10x Visium 数据
    adata = sc.read_visium(args.adata_path)
    
    log(f"  数据形状: {adata.shape}")
    log(f"  样本数: {adata.n_obs}")
    log(f"  基因数: {adata.n_vars}")
    
    # 2.预处理
    log("\n🔧 数据预处理...")
    
    # 保存原始计数
    if 'counts' not in adata.layers:
        if issparse(adata.X):
            adata.layers['counts'] = adata.X.copy()
        else:
            adata.layers['counts'] = adata.X.copy()
        log("  保存原始计数到 adata.layers['counts']")
    
    # 归一化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['normalized_log'] = adata.X.copy()
    log("  ✅ 归一化完成")
    
    # 高变基因
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_hvg,
        subset=False,
        flavor='seurat_v3',
        layer='counts'
    )
    
    adata_hvg = adata[: , adata.var['highly_variable']].copy()
    log(f"  ✅ 选择 {args.n_hvg} 个高变基因")
    log(f"  HVG数据形状: {adata_hvg.shape}")
    
    # PCA
    sc.pp.scale(adata_hvg)
    sc.tl.pca(adata_hvg, n_comps=args.n_pca, svd_solver='arpack')
    log(f"  ✅ PCA降维到 {args.n_pca} 维")
    
    log("\n📊 构建空间KNN图...")
    
    from src.utils.dual_graph_builder import build_spatial_graph_only  # 需要新写这个函数
    
    spatial_edge_index, spatial_edge_weight = build_spatial_graph_only(
        adata_hvg,
        k=args.k_spatial,
        spatial_key=args.spatial_key
    )
    
    log(f"  ✅ 空间图:  {spatial_edge_index.size(1)} 条边")
    
    # 4.准备张量
    log("\n📦 准备训练数据...")
    
    X_pca = adata_hvg.obsm['X_pca'][: , :args.n_pca]
    X_pca_tensor = torch.FloatTensor(X_pca).to(device)
    log(f"  X_pca:  {X_pca_tensor.shape}")
    
    if 'counts' in adata_hvg.layers:
        X_raw = adata_hvg.layers['counts']
        if issparse(X_raw):
            X_raw = X_raw.toarray()
    else:
        X_raw = adata_hvg.X
        if issparse(X_raw):
            X_raw = X_raw.toarray()
    
    X_raw_tensor = torch.FloatTensor(X_raw).to(device)
    log(f"  X_raw: {X_raw_tensor.shape}")
    
    # ========================================
    # 初始化GIB模型
    # ========================================
    model = SpatialDomainModelGIB(
        n_input=args.n_pca,
        n_output=adata_hvg.n_vars,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_domains=args.n_domains,
        gib_hidden=args.gib_hidden,
        gib_min_keep=args.gib_min_keep,
        gib_max_keep=args.gib_max_keep,
        contrastive_temperature=args.contrastive_temp,
        contrastive_weight=args.contrastive_weight,
        prototype_dim=args.prototype_dim,
        prototype_temperature=args.prototype_temp,
        prototype_weight=args.prototype_weight,
        gnn_num_layers=args.gnn_num_layers,
        dropout=args.dropout,
        reconstruction_loss=args.reconstruction_loss
    ).to(device)
    
    model.attach_spatial_graph(
        spatial_edge_index.to(device),
        spatial_edge_weight.to(device)
    )
    
    model.attach_full_X(X_pca=X_pca_tensor, X_raw=X_raw_tensor)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  参数量: {n_params:,}")
    
        # ========================================
    # 优化器和调度器
    # ========================================
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
    
    # ========================================
    # 训练循环
    # ========================================
    log("\n🏋️ 开始训练...")
    log("=" * 80)
    log(f"🔥 Warmup期:   Epoch 1-{args.warmup_epochs} (只训练重构)")
    log(f"🔥 对比学习启动:  Epoch {args.warmup_epochs + 1}")
    log(f"🔥 初始化原型:   Epoch {args.warmup_epochs + 1} (K-Means)")
    log(f"🔥 完整训练:  Epoch {args.warmup_epochs + 2}-{args.max_epochs}")
    log("=" * 80)
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        
        # 训练一个epoch
        metrics = train_one_epoch_gib(
            model,
            X_pca_tensor,
            X_raw_tensor,
            optimizer,
            epoch,
            device,
            warmup_epochs=args.warmup_epochs
        )
        
        # 更新学习率
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # 记录指标
        metrics['epoch'] = epoch + 1
        metrics['lr'] = scheduler.get_last_lr()[0]
        metrics['time'] = epoch_time
        history.append(metrics)
        
        # ========================================
        # 打印日志
        # ========================================
        if (epoch + 1) % args.log_every == 0:
            log(f"\n{'='*80}")
            
            if metrics.get('in_warmup', False):
                log(f"Epoch {epoch+1}/{args.max_epochs} [WARMUP] ({epoch_time:.2f}s)")
            else:
                log(f"Epoch {epoch+1}/{args.max_epochs} ({epoch_time:.2f}s)")
            
            log(f"{'='*80}")
            log(f"  Total Loss:               {metrics['total_loss']:.4f}")
            log(f"  Recon Loss:              {metrics['recon_loss']:.4f}")
            
            # 🔥 GIB稀疏性损失
            if 'gib_sparsity_loss' in metrics:
                log(f"  🔥 GIB Sparsity Loss:    {metrics['gib_sparsity_loss']:.4f}")
            
            # 🔥 对比学习损失
            if not metrics.get('in_warmup', False) and 'contrastive_loss' in metrics:
                log(f"  🔥 Contrastive Loss:     {metrics['contrastive_loss']:.4f}")
            
            # 🔥 原型学习损失
            if not metrics.get('in_warmup', False) and metrics.get('prototype_loss', 0) > 0:
                log(f"  🔥 Prototype Loss:       {metrics['prototype_loss']:.4f}")
                
                if 'proto_s2p' in metrics and metrics['proto_s2p'] > 0:
                    log(f"    - Spot-to-Prototype:     {metrics['proto_s2p']:.4f}")
                    log(f"    - Compactness:         {metrics['proto_compact']:.4f}")
                    log(f"    - Dispersion:          {metrics['proto_disperse']:.4f}")
            
            # 🔥 GIB统计
            if 'gib_keep_rate' in metrics:  
                log(f"  GIB Keep Rate:           {metrics['gib_keep_rate']:.3f}")
                log(f"  GIB Avg Prob:            {metrics['gib_avg_prob']:.3f}")
            
            # Dispersion参数（如果使用NB/ZINB）
            if 'px_r_mean' in metrics:  
                log(f"  Dispersion (θ):          {metrics['px_r_mean']:.3f} ± {metrics['px_r_std']:.3f}")
            
            log(f"  Learning Rate:           {metrics['lr']:.6f}")
        
        # ========================================
        # 保存最佳模型
        # ========================================
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            best_model_path = Path(args.save_dir) / 'best_model.pt'
            
            torch.save({
                'epoch':  epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 🔥 也保存scheduler
                'loss': best_loss,
                'args': vars(args),
                'seed': args.seed,
                'history': history  # 🔥 可选：保存训练历史
            }, best_model_path)
    
    log("\n" + "="*80)
    log("🎉 训练完成！")
    log(f"   最佳损失: {best_loss:.4f}")
    log("="*80)
    
    # ========================================
    # 保存训练历史
    # ========================================
    history_df = pd.DataFrame(history)
    history_path = Path(args.save_dir) / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    log(f"  ✅ 保存训练历史到 {history_path}")
    
    # ========================================
    # 提取latent表示并聚类
    # ========================================
    log("\n🔍 提取 latent 表示并聚类...")
    
    model.eval()
    with torch.no_grad():
        z_latent = model.get_latent_representation()
    
    adata.obsm['X_spatial_domain'] = z_latent
    log(f"  ✅ Latent 表示:   {z_latent.shape}")
    
    # 🔥 使用mclust聚类（同原版）
    log(f"\n🎯 使用指定的聚类数:  {args.n_domains}, 模型:  {args.mclust_model}")
    
    try:
        from src.utils.mclust_clustering import mclust_with_spatial_refinement
        
        adata = mclust_with_spatial_refinement(
            adata,
            n_clusters=int(args.n_domains),
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
        
        if 'mclust_bic' in adata.uns:
            log(f"  BIC: {adata.uns['mclust_bic']:.2f}")
        
        if 'mclust_uncertainty' in adata.obs.columns:
            uncertainty = adata.obs['mclust_uncertainty'].values
            log(f"  平均 uncertainty: {uncertainty.mean():.4f}")
            log(f"  低置信度样本 (>0.1): {(uncertainty > 0.1).sum()} / {len(uncertainty)}")
    
    except Exception as e: 
        log(f"\n❌ mclust 聚类失败: {e}")
        import traceback
        log(traceback.format_exc())
        
        log("\n⚠️ 回退到 Python GMM...")
        from sklearn.mixture import GaussianMixture
        
        gmm = GaussianMixture(n_components=int(args.n_domains), random_state=args.seed, n_init=10)
        labels = gmm.fit_predict(z_latent)
        probas = gmm.predict_proba(z_latent)
        
        adata.obs['mclust'] = labels.astype(str)
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        adata.obs['mclust_probability'] = probas.max(axis=1)
        adata.uns['mclust_bic'] = -gmm.bic(z_latent)
        
        n_domains = len(np.unique(labels))
        log(f"  ✅ GMM 发现 {n_domains} 个空间域")
    
    # ========================================
    # 保存结果
    # ========================================
    output_path = Path(args.save_dir) / 'adata_with_domains.h5ad'
    adata.write_h5ad(output_path)
    log(f"\n💾 保存结果到 {output_path}")

    # ========================================
    # 11.🔥 计算评估指标（从 metadata.tsv 加载 ground truth）
    # ========================================
    log("\n📊 计算评估指标...")
    
    # 🔥 尝试从 metadata.tsv 加载 ground truth
    metadata_path = Path(args.adata_path) / 'metadata.tsv'
    gt_column = None
    has_ground_truth = False
    
    if metadata_path.exists():
        log(f"  找到 metadata 文件: {metadata_path}")
        
        try:
            # 读取 metadata
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
            log(f"  V-measure:                            {v_measure:.4f}")
            log(f"  {'='*60}")
            
            # 聚类分布分析
            log(f"\n  聚类 vs Ground Truth 交叉表:")
            cross_tab = pd.crosstab(
                pd.Series(y_pred, name='Predicted'),
                pd.Series(y_true, name='True')
            )
            log(f"\n{cross_tab}")
            
            # 保存指标到文件
            eval_results = {
                'ARI': ari,
                'NMI': nmi,
                'AMI': ami,
                'FMI': fmi,
                'Homogeneity': homogeneity,
                'Completeness': completeness,
                'V-measure':  v_measure,
                'n_valid_samples': int(n_valid),
                'n_total_samples': int(n_total),
                'ground_truth_column': gt_column,
                'seed': args.seed
            }
            
            eval_df = pd.DataFrame([eval_results])
            eval_path = Path(args.save_dir) / 'evaluation.csv'
            eval_df.to_csv(eval_path, index=False)
            log(f"\n  ✅ 保存评估结果到 {eval_path}")
            
            # 保存混淆矩阵
            try:
                all_true_labels = sorted(set(y_true))
                all_pred_labels = sorted(set(y_pred))
                
                cm = confusion_matrix(y_true, y_pred, labels=all_true_labels)
                
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
            
            # 保存到 adata
            adata.uns['evaluation_metrics'] = eval_results
            
            # 重新保存 adata
            adata.write_h5ad(output_path)
        
        else:
            log(f"  ⚠️ 警告: 没有有效的样本用于评估")

    else:
        log(f"  ℹ️ 未找到 ground truth，跳过评估")
        log(f"  💡 提示: 确保数据目录包含 metadata.tsv 文件")
        log(f"     期望路径: {Path(args.adata_path) / 'metadata.tsv'}")


    # 在评估完成后，添加可视化
    # ========================================
    # 生成可视化
    # ========================================
    log("\n🎨 生成可视化...")
    
    try:
        # 1.空间分布对比图
        plot_spatial_comparison(adata, args.save_dir, args.spatial_key)
        
        # 2.训练曲线（需要适配新的metrics）
        plot_training_curves_gib(history_df, args.save_dir)
        
        # 3.GIB统计图（新增）
        if 'gib_keep_rate' in history_df.columns:
            plot_gib_statistics(history_df, args.save_dir)
        
        # 4.聚类分布
        plot_cluster_distribution(adata, args.save_dir)
        
        log("  ✅ 所有可视化完成")
    
    except Exception as e: 
        log(f"  ⚠️ 可视化失败: {e}")
        import traceback
        log(traceback.format_exc())
    
    log("\n✨ 所有任务完成！")
    

# ========================================
# 命令行参数
# ========================================
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='空间域识别 - V2版本（原型学习）')
    
    # 数据参数
    parser.add_argument('--adata_path', type=str, required=True,
                        help='AnnData文件路径')
    parser.add_argument('--spatial_key', type=str, default='spatial',
                        help='空间坐标的key')
    parser.add_argument('--save_dir', type=str, default='./results/spatial_domain_v2',
                        help='结果保存目录')
    
    # 预处理参数
    parser.add_argument('--n_hvg', type=int, default=3000,
                        help='高变基因数量')
    parser.add_argument('--n_pca', type=int, default=50,
                        help='PCA维度')
    
    # 图构建参数
    parser.add_argument('--k_spatial', type=int, default=6,
                        help='空间图的K邻居')

    
    # 模型架构参数
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层维度')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent维度')
    parser.add_argument('--gnn_num_layers', type=int, default=3,
                        help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')
    parser.add_argument('--reconstruction_loss', type=str, default='zinb',
                        choices=['nb', 'zinb', 'poisson'],
                        help='重构损失类型')
    
    
    # 🔥 原型学习参数
    parser.add_argument('--use_prototype', action='store_true', default=False,
                        help='是否使用原型学习')
    parser.add_argument('--n_domains', type=int, default=7,
                        help='域数量（原型数量）')
    parser.add_argument('--prototype_dim', type=int, default=128,
                        help='原型维度')
    parser.add_argument('--prototype_temperature', type=float, default=0.1,
                        help='原型对比温度')
    parser.add_argument('--prototype_momentum', type=float, default=0.99,
                        help='原型EMA动量')

    
    # 🔥 Warmup参数
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Warmup轮数')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--log_every', type=int, default=1,
                        help='日志打印频率')
    
    # 聚类参数
    # 🔥 mclust参数
    parser.add_argument('--mclust_model', type=str, default='EEE')
    parser.add_argument('--refine_boundary', action='store_true', default=False,
                       help='是否进行边界细化')
    
    # 🔥 新增GIB参数
    parser.add_argument('--gib_hidden', type=int, default=64)
    parser.add_argument('--gib_min_keep', type=float, default=0.05)  # 🔥 改为0.05
    parser.add_argument('--gib_sparsity_weight', type=float, default=0.5)  # 🔥 新增
    parser.add_argument('--gib_max_keep', type=float, default=0.8)
    
    # 🔥 对比学习参数
    parser.add_argument('--contrastive_temp', type=float, default=0.1)
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    
    # 原型学习参数
    parser.add_argument('--prototype_temp', type=float, default=0.3)
    parser.add_argument('--prototype_weight', type=float, default=20.0)

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    main(args)