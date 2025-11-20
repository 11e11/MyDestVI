import matplotlib
# 强制使用非交互式后端，防止在服务器上报错
matplotlib.use('Agg') 
import scanpy as sc
import matplotlib.pyplot as plt
import os

# 0. 准备工作：确保输出目录存在
output_dir = './results'


# 1. 加载数据
file_path = '../data/simulation_Mouse_Kidney_MERFISH/MERFISH_kidney_object.h5ad'
print(f"正在加载数据: {file_path} ...")
adata = sc.read_h5ad(file_path)

# ==========================================
# 配置：在这里指定你想要可视化的多个细胞类型
# ==========================================
target_cell_types = [
    'immune cell', 
    'epithelial cell of proximal tubule',
    'kidney loop of Henle epithelial cell',
    'kidney distal convoluted tubule epithelial cell',
    'endothelial cell'
]

# 打印一下数据中实际存在的类型，方便核对
print("\n数据中包含的细胞类型:", adata.obs['free_annotation'].unique().tolist())
print(f"即将绘制以下类型: {target_cell_types}\n")

# =================================================
# 定义绘图函数（为了支持循环调用）
# =================================================

def plot_scanpy_style(adata, cell_type, save_dir):
    """方法 A: 使用 Scanpy 风格绘制"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 生成安全的文件名（把空格换成下划线）
    safe_name = cell_type.replace(" ", "_").replace("/", "-")
    filename = os.path.join(save_dir, f'{safe_name}_scanpy.png')

    try:
        sc.pl.spatial(
            adata,
            color='free_annotation',
            groups=[cell_type],
            spot_size=30,
            title=f'Spatial: {cell_type}',
            na_color='lightgrey',
            na_in_legend=False,
            frameon=False,
            show=False,
            ax=ax
        )
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[Scanpy风格] 已保存: {filename}")
    except Exception as e:
        print(f"绘制 {cell_type} (Scanpy) 时出错: {e}")
    finally:
        plt.close()

def plot_manual_style(adata, cell_type, save_dir):
    """方法 B: 手动绘制风格 (红点+灰底)"""
    coords = adata.obsm['spatial']
    
    # 分离背景和目标
    mask = adata.obs['free_annotation'] == cell_type
    
    # 如果该类型在数据中不存在，跳过
    if not mask.any():
        print(f"[警告] 数据中未找到细胞类型: {cell_type}，跳过绘制。")
        return

    bg_coords = coords[~mask]
    target_coords = coords[mask]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 画背景 (灰色)
    ax.scatter(bg_coords[:, 0], bg_coords[:, 1], c='lightgrey', s=5, alpha=0.3, label='Others')
    # 画目标 (红色)
    ax.scatter(target_coords[:, 0], target_coords[:, 1], c='red', s=10, alpha=0.8, label=cell_type)
    
    ax.set_title(f'Distribution: {cell_type}')
    ax.legend(loc='upper right') # 图例位置
    ax.set_aspect('equal')
    plt.axis('off')
    
    # 生成文件名
    safe_name = cell_type.replace(" ", "_").replace("/", "-")
    filename = os.path.join(save_dir, f'{safe_name}_manual.png')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[手动风格]   已保存: {filename}")
    plt.close()

# =================================================
# 主循环：遍历列表进行绘制
# =================================================

for cell_type in target_cell_types:
    print(f"--- 处理: {cell_type} ---")
    
    # 检查该类型是否真的在 adata 里，防止报错
    if cell_type not in adata.obs['free_annotation'].values:
        print(f"警告: '{cell_type}' 不在数据的 free_annotation 列中，已跳过。")
        continue
        
    plot_scanpy_style(adata, cell_type, output_dir)
    plot_manual_style(adata, cell_type, output_dir)

print("\n所有绘图任务完成！")