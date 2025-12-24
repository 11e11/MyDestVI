"""GIB诊断工具：可视化和分析Mask学习情况"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd


class GIBDiagnostic:
    """GIB诊断器"""
    
    def __init__(self, model, adata, save_dir):
        """
        Args:
            model: 训练好的SpatialDomainModel
            adata: AnnData对象（包含ground truth）
            save_dir: 保存路径
        """
        self. model = model
        self. adata = adata
        self. save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def extract_similarity_matrix(self):
        """
        提取完整的相似度矩阵 S [N, N]
        """
        print("\n🔍 提取相似度矩阵...")
        self.model.eval()
        
        # 获取输入特征
        X_pca = self.model._X_pca  # [N, n_input]
        
        # 通过GIB模块计算相似度矩阵
        if not self.model.use_gib:
            print("  ⚠️ 模型未启用GIB，无法提取相似度矩阵")
            return None
        
        S = self.model.encoder. gib_module.compute_similarity_matrix(X_pca)
        S_np = S.cpu().numpy()
        
        print(f"  ✅ 相似度矩阵形状: {S_np.shape}")
        print(f"  - 范围: [{S_np.min():.3f}, {S_np.max():.3f}]")
        print(f"  - 均值: {S_np.mean():.3f}")
        print(f"  - 标准差: {S_np. std():.3f}")
        
        return S_np
    
    @torch.no_grad()
    def extract_edge_masks(self):
        """
        提取边掩码（针对实际存在的边）
        """
        print("\n🔍 提取边掩码...")
        self.model.eval()
        
        X_pca = self.model._X_pca
        spatial_edge_index = self.model._spatial_edge_index
        expr_edge_index = self.model._expr_edge_index
        
        # 计算mask
        spatial_mask = self.model.encoder.gib_module(X_pca, spatial_edge_index)
        expr_mask = self. model.encoder.gib_module(X_pca, expr_edge_index)
        
        spatial_mask_np = spatial_mask.cpu().numpy()
        expr_mask_np = expr_mask.cpu().numpy()
        
        print(f"  ✅ 空间边掩码:  {spatial_mask_np.shape}")
        print(f"  - 范围: [{spatial_mask_np.min():.3f}, {spatial_mask_np.max():.3f}]")
        print(f"  - 均值: {spatial_mask_np.mean():.3f}")
        print(f"  - 稀疏度(<0.1): {(spatial_mask_np < 0.1).mean():.2%}")
        
        print(f"  ✅ 表达边掩码: {expr_mask_np.shape}")
        print(f"  - 范围: [{expr_mask_np.min():.3f}, {expr_mask_np.max():.3f}]")
        print(f"  - 均值: {expr_mask_np.mean():.3f}")
        print(f"  - 稀疏度(<0.1): {(expr_mask_np < 0.1).mean():.2%}")
        
        return {
            'spatial_mask': spatial_mask_np,
            'expr_mask': expr_mask_np,
            'spatial_edge_index': spatial_edge_index. cpu().numpy(),
            'expr_edge_index': expr_edge_index.cpu().numpy()
        }
    
    def plot_similarity_matrix_heatmap(self, S, max_spots=500):
        """
        🔥 核心诊断：绘制相似度矩阵热图
        """
        print(f"\n🎨 绘制相似度矩阵热图...")
        
        n_spots = S.shape[0]
        
        # 如果样本太多，随机采样
        if n_spots > max_spots:
            print(f"  样本数 {n_spots} 过多，随机采样 {max_spots} 个")
            indices = np.random.choice(n_spots, max_spots, replace=False)
            indices = np.sort(indices)
            S_plot = S[indices][: , indices]
        else:
            S_plot = S
            indices = np.arange(n_spots)
        
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        
        # 1. 原始相似度矩阵
        im1 = axes[0].imshow(S_plot, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[0].set_title('Raw Similarity Matrix S\n(Cosine Similarity)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Spot Index')
        axes[0].set_ylabel('Spot Index')
        plt.colorbar(im1, ax=axes[0], label='Similarity')
        
        # 2.  Sigmoid后的Mask
        M = 1 / (1 + np.exp(-S_plot))  # sigmoid
        im2 = axes[1].imshow(M, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[1].set_title('Mask after Sigmoid(S)\n(Actual Edge Weights)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Spot Index')
        axes[1].set_ylabel('Spot Index')
        plt.colorbar(im2, ax=axes[1], label='Mask Value')
        
        # 3. 二值化Mask（阈值0.5）
        M_binary = (M > 0.5).astype(float)
        im3 = axes[2].imshow(M_binary, cmap='Greys', aspect='auto', vmin=0, vmax=1)
        axes[2].set_title('Binary Mask (Threshold=0.5)\n(Active Connections)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Spot Index')
        axes[2].set_ylabel('Spot Index')
        plt.colorbar(im3, ax=axes[2], label='Active (1) / Inactive (0)')
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'similarity_matrix_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 保存到:  {save_path}")
        plt.close()
        
        return M
    
    def plot_similarity_with_ground_truth(self, S, gt_column='mclust', max_spots=500):
        """
        🔥 高级诊断：按ground truth排序后的相似度矩阵
        """
        print(f"\n🎨 绘制按域排序的相似度矩阵...")
        
        if gt_column not in self. adata.obs.columns:
            print(f"  ⚠️ 未找到列 '{gt_column}'，跳过")
            return
        
        # 获取标签
        labels = self.adata.obs[gt_column].values
        
        # 过滤无效标签
        valid_mask = (labels != 'nan') & (labels != 'NA') & (labels != '')
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            print("  ⚠️ 没有有效的ground truth标签")
            return
        
        labels_valid = labels[valid_indices]
        S_valid = S[valid_indices][: , valid_indices]
        
        # 按标签排序
        sorted_indices = np.argsort(labels_valid)
        S_sorted = S_valid[sorted_indices][:, sorted_indices]
        labels_sorted = labels_valid[sorted_indices]
        
        # 采样
        n_valid = len(labels_sorted)
        if n_valid > max_spots: 
            step = n_valid // max_spots
            sample_indices = np.arange(0, n_valid, step)
            S_sorted = S_sorted[sample_indices][: , sample_indices]
            labels_sorted = labels_sorted[sample_indices]
        
        # 绘制
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. 原始相似度（排序后）
        im1 = axes[0].imshow(S_sorted, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[0].set_title(f'Similarity Matrix (Sorted by {gt_column})\nShould Show Block-Diagonal Structure', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Spot Index (Sorted)')
        axes[0].set_ylabel('Spot Index (Sorted)')
        plt.colorbar(im1, ax=axes[0], label='Similarity')
        
        # 添加域分割线
        unique_labels = np.unique(labels_sorted)
        boundaries = []
        for label in unique_labels[:-1]:
            boundary = np.where(labels_sorted == label)[0][-1]
            boundaries.append(boundary)
            axes[0].axhline(boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
            axes[0].axvline(boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
        
        # 2.  Mask（排序后）
        M_sorted = 1 / (1 + np.exp(-S_sorted))
        im2 = axes[1].imshow(M_sorted, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[1].set_title(f'Mask after Sigmoid (Sorted by {gt_column})', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Spot Index (Sorted)')
        axes[1].set_ylabel('Spot Index (Sorted)')
        plt.colorbar(im2, ax=axes[1], label='Mask Value')
        
        # 添加域分割线
        for boundary in boundaries:
            axes[1].axhline(boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
            axes[1].axvline(boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
        
        plt.tight_layout()
        
        save_path = self.save_dir / f'similarity_matrix_sorted_by_{gt_column}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 保存到: {save_path}")
        plt.close()
    
    def analyze_mask_distribution(self, edge_masks):
        """
        分析边掩码的分布特征
        """
        print(f"\n📊 分析掩码分布...")
        
        spatial_mask = edge_masks['spatial_mask']
        expr_mask = edge_masks['expr_mask']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 空间掩码直方图
        axes[0, 0].hist(spatial_mask, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
        axes[0, 0].set_xlabel('Mask Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Spatial Edge Mask Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 表达掩码直方图
        axes[0, 1].hist(expr_mask, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[0, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
        axes[0, 1].set_xlabel('Mask Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Expression Edge Mask Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. CDF（累积分布）
        spatial_sorted = np.sort(spatial_mask)
        expr_sorted = np.sort(expr_mask)
        
        axes[1, 0].plot(spatial_sorted, np.linspace(0, 1, len(spatial_sorted)), 
                       label='Spatial', linewidth=2)
        axes[1, 0].plot(expr_sorted, np.linspace(0, 1, len(expr_sorted)), 
                       label='Expression', linewidth=2)
        axes[1, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1, 0].set_xlabel('Mask Value')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution Function (CDF)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 统计摘要
        stats_text = f"""
        Spatial Mask Statistics:
        - Mean: {spatial_mask.mean():.3f}
        - Std:   {spatial_mask.std():.3f}
        - Median: {np.median(spatial_mask):.3f}
        - Q25: {np.percentile(spatial_mask, 25):.3f}
        - Q75: {np.percentile(spatial_mask, 75):.3f}
        - Sparsity (<0.1): {(spatial_mask < 0.1).mean():.2%}
        - Active (>0.5): {(spatial_mask > 0.5).mean():.2%}
        
        Expression Mask Statistics:
        - Mean: {expr_mask. mean():.3f}
        - Std:  {expr_mask.std():.3f}
        - Median: {np.median(expr_mask):.3f}
        - Q25: {np.percentile(expr_mask, 25):.3f}
        - Q75: {np.percentile(expr_mask, 75):.3f}
        - Sparsity (<0.1): {(expr_mask < 0.1).mean():.2%}
        - Active (>0.5): {(expr_mask > 0.5).mean():.2%}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'mask_distribution_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 保存到:  {save_path}")
        plt.close()
    
    def compute_mask_quality_metrics(self, S, gt_column='mclust'):
        """
        🔥 量化诊断：计算掩码质量指标
        """
        print(f"\n📈 计算掩码质量指标...")
        
        if gt_column not in self. adata.obs.columns:
            print(f"  ⚠️ 未找到列 '{gt_column}'")
            return None
        
        labels = self.adata.obs[gt_column].values
        valid_mask = (labels != 'nan') & (labels != 'NA') & (labels != '')
        
        if valid_mask.sum() == 0:
            print("  ⚠️ 没有有效标签")
            return None
        
        labels_valid = labels[valid_mask]
        S_valid = S[valid_mask][:, valid_mask]
        M_valid = 1 / (1 + np.exp(-S_valid))
        
        # 指标1: 域内相似度 vs 域间相似度
        n_valid = len(labels_valid)
        within_domain_sim = []
        between_domain_sim = []
        
        for i in range(n_valid):
            for j in range(i+1, n_valid):
                if labels_valid[i] == labels_valid[j]:
                    within_domain_sim.append(M_valid[i, j])
                else:
                    between_domain_sim.append(M_valid[i, j])
        
        within_mean = np.mean(within_domain_sim) if within_domain_sim else 0
        between_mean = np.mean(between_domain_sim) if between_domain_sim else 0
        separation_ratio = within_mean / (between_mean + 1e-8)
        
        # 指标2: Silhouette Score（使用相似度作为距离的倒数）
        distance_matrix = 1 / (M_valid + 1e-8)
        try:
            silhouette = silhouette_score(distance_matrix, labels_valid, metric='precomputed')
        except:
            silhouette = np.nan
        
        # 指标3: Block-Diagonal 结构强度
        # 按标签排序后，计算对角块的平均值 vs 非对角块的平均值
        sorted_indices = np.argsort(labels_valid)
        M_sorted = M_valid[sorted_indices][:, sorted_indices]
        labels_sorted = labels_valid[sorted_indices]
        
        diagonal_values = []
        off_diagonal_values = []
        
        unique_labels = np.unique(labels_sorted)
        for label in unique_labels: 
            mask_label = (labels_sorted == label)
            indices_label = np.where(mask_label)[0]
            
            # 域内（对角块）
            for i in indices_label:
                for j in indices_label: 
                    if i < j:
                        diagonal_values.append(M_sorted[i, j])
            
            # 域间（非对角块）
            for i in indices_label:
                for j in range(len(labels_sorted)):
                    if labels_sorted[j] != label: 
                        off_diagonal_values.append(M_sorted[i, j])
        
        diagonal_mean = np.mean(diagonal_values) if diagonal_values else 0
        off_diagonal_mean = np.mean(off_diagonal_values) if off_diagonal_values else 0
        block_strength = diagonal_mean - off_diagonal_mean
        
        metrics = {
            'within_domain_similarity': within_mean,
            'between_domain_similarity': between_mean,
            'separation_ratio': separation_ratio,
            'silhouette_score': silhouette,
            'block_diagonal_strength': block_strength,
            'diagonal_mean': diagonal_mean,
            'off_diagonal_mean': off_diagonal_mean
        }
        
        print(f"\n  ✅ 掩码质量指标:")
        print(f"  - 域内相似度:     {within_mean:.3f}")
        print(f"  - 域间相似度:     {between_mean:.3f}")
        print(f"  - 分离比率:      {separation_ratio:.3f} (越大越好)")
        print(f"  - Silhouette:     {silhouette:.3f} (>0表示有结构)")
        print(f"  - 块对角强度:    {block_strength:.3f} (>0表示有块结构)")
        
        # 保存到文件
        report_path = self.save_dir / 'mask_quality_metrics.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("GIB Mask 质量诊断报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("【量化指标】\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n【诊断结论】\n")
            
            # 情况A:  GIB有用但没调好
            if separation_ratio > 1.2 and block_strength > 0.1:
                f.write("  ✅ 情况A: GIB正在学习有意义的结构\n")
                f.write("     - Mask显示出域内高相似、域间低相似的模式\n")
                f.write("     - 建议:  可以尝试增大GIB权重，进一步稀疏化\n")
            
            # 情况B:  GIB彻底没用
            elif separation_ratio < 1.05 and abs(block_strength) < 0.05:
                f.write("  ❌ 情况B: GIB没有学到有用的结构\n")
                f.write("     - Mask几乎是随机的或完全均匀的\n")
                f.write("     - 可能原因:\n")
                f.write("       1.  Metric Network太弱，无法从PCA特征学习相似度\n")
                f.write("       2. GIB权重过大，过早稀疏化\n")
                f.write("       3. 输入特征质量不足（PCA维度太低？）\n")
                f.write("     - 建议:\n")
                f.write("       1. 增加Metric Network的深度/宽度\n")
                f.write("       2. 延长GIB预热期\n")
                f.write("       3. 增加PCA维度或使用原始归一化数据\n")
            
            # 中间状态
            else:
                f.write("  ⚠️ 中间状态: GIB学到了部分结构但不够强\n")
                f.write("     - 建议: 调整GIB超参数\n")
                if separation_ratio < 1.2:
                    f.write("       → 增大GIB权重，强化稀疏性\n")
                if block_strength < 0.1:
                    f.write("       → 增加Metric Network容量\n")
        
        print(f"  ✅ 保存诊断报告到: {report_path}")
        
        return metrics
    
    def run_full_diagnostic(self, gt_column='mclust', max_spots=500):
        """
        🔥 运行完整诊断流程
        """
        print("\n" + "=" * 80)
        print("🔬 开始GIB完整诊断")
        print("=" * 80)
        
        # 1. 提取相似度矩阵
        S = self.extract_similarity_matrix()
        if S is None:
            return
        
        # 2. 提取边掩码
        edge_masks = self.extract_edge_masks()
        
        # 3. 绘制热图（无序）
        self.plot_similarity_matrix_heatmap(S, max_spots=max_spots)
        
        # 4. 绘制热图（按GT排序）
        self.plot_similarity_with_ground_truth(S, gt_column=gt_column, max_spots=max_spots)
        
        # 5. 分析掩码分布
        self.analyze_mask_distribution(edge_masks)
        
        # 6. 计算质量指标
        metrics = self.compute_mask_quality_metrics(S, gt_column=gt_column)
        
        print("\n" + "=" * 80)
        print("✅ 诊断完成！请查看以下文件:")
        print(f"  1. {self.save_dir / 'similarity_matrix_heatmap.png'}")
        print(f"  2. {self.save_dir / f'similarity_matrix_sorted_by_{gt_column}.png'}")
        print(f"  3. {self.save_dir / 'mask_distribution_analysis.png'}")
        print(f"  4. {self.save_dir / 'mask_quality_metrics.txt'}")
        print("=" * 80)
        
        return metrics