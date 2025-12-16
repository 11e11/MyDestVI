"""
mclust 聚类工具 - 修复 rpy2 3.6.4 兼容性
"""
import numpy as np
import warnings


def mclust_R(
    adata,
    n_clusters:  int,
    use_rep: str = 'X_spatial_domain',
    key_added: str = 'mclust',
    modelNames: str = 'EEE',
    random_seed: int = 2020
):
    """
    使用 mclust 进行聚类（兼容 rpy2 3.6.4）
    """
    print(f"\n🎯 使用 mclust 聚类:")
    print(f"   - 表示:  {use_rep}")
    print(f"   - 聚类数: {n_clusters}")
    print(f"   - 模型:  {modelNames}")
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 导入 rpy2
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError:
        raise ImportError("需要安装 rpy2: pip install rpy2")
    
    # 🔥 静默加载 mclust（避免打印信息）
    try:
        # 捕获 R 输出
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                robjects.r. library("mclust")
    except Exception as e:
        raise RuntimeError(f"无法加载 mclust R 包: {e}\n请在 R 中运行: install.packages('mclust')")
    
    # 🔥 使用新版 API（避免 activate() 弃用警告）
    try:
        # 创建 converter
        cv = robjects.default_converter + numpy2ri.converter
        use_converter = True
        print(f"   ✅ 使用 rpy2 新版 API")
    except:
        # 回退到旧版 API
        numpy2ri.activate()
        use_converter = False
        print(f"   ✅ 使用 rpy2 旧版 API")
    
    # 设置 R 随机种子
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    
    # 获取 mclust 函数
    rmclust = robjects.r['Mclust']
    
    # 获取数据
    if use_rep not in adata.obsm:
        raise ValueError(f"adata.obsm 中没有 '{use_rep}'")
    
    X = adata.obsm[use_rep]
    print(f"   - 数据形状: {X.shape}")
    
    # 数据验证
    if np.isnan(X).any():
        print(f"   ⚠️ 警告: 数据包含 NaN，将替换为 0")
        X = np.nan_to_num(X, nan=0.0)
    
    if np.isinf(X).any():
        print(f"   ⚠️ 警告: 数据包含 Inf，将替换为有限值")
        X = np.nan_to_num(X, posinf=X[~np.isinf(X)].max(), neginf=X[~np.isinf(X)].min())
    
    # 标准化数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✅ 数据已标准化")
    
    try:
        # 🔥 根据版本选择转换方式
        if use_converter: 
            # 新版 API
            with localconverter(cv):
                X_r = robjects.conversion.py2rpy(X_scaled)
                
                print(f"   ⏳ 运行 mclust...")
                res = rmclust(X_r, n_clusters, modelNames)
                
                # 提取结果
                try:
                    mclust_res = np.array(robjects.r('$')(res, 'classification'))
                except:
                    try:
                        mclust_res = np.array(res[-2])
                    except:
                        mclust_res = np. array(res[res.names.index('classification')])
        
        else:
            # 旧版 API
            X_r = numpy2ri.numpy2rpy(X_scaled)
            
            print(f"   ⏳ 运行 mclust...")
            res = rmclust(X_r, n_clusters, modelNames)
            
            # 提取结果
            try:
                mclust_res = np.array(robjects.r('$')(res, 'classification'))
            except: 
                try:
                    mclust_res = np.array(res[-2])
                except:
                    mclust_res = np.array(res[res.names.index('classification')])
        
        # R 的索引从 1 开始，转换为从 0 开始
        mclust_res = mclust_res. astype(int) - 1
        
        print(f"   ✅ 聚类完成")
        print(f"      - 聚类分布: {np.bincount(mclust_res)}")
        
        # 保存结果
        adata.obs[key_added] = mclust_res
        adata.obs[key_added] = adata.obs[key_added]. astype('int')
        adata.obs[key_added] = adata.obs[key_added].astype('category')
        
        # 提取额外信息
        try:
            if use_converter:
                with localconverter(cv):
                    bic = np.array(robjects.r('$')(res, 'bic'))[0]
            else:
                bic = np.array(robjects.r('$')(res, 'bic'))[0]
            
            adata.uns[f'{key_added}_bic'] = float(bic)
            print(f"      - BIC:  {bic:.2f}")
        except:
            pass
        
        try:
            if use_converter: 
                with localconverter(cv):
                    uncertainty = np.array(robjects. r('$')(res, 'uncertainty'))
            else:
                uncertainty = np.array(robjects.r('$')(res, 'uncertainty'))
            
            adata.obs[f'{key_added}_uncertainty'] = uncertainty
            print(f"      - 平均 uncertainty: {uncertainty.mean():.4f}")
        except:
            pass
        
        return adata
    
    except Exception as e:
        print(f"\n❌ mclust 失败: {e}")
        
        # 回退
        if modelNames == 'VVV':
            print(f"\n⚠️ 尝试更稳定的模型 'EEE'...")
            return mclust_R(adata, n_clusters, use_rep, key_added, modelNames='EEE', random_seed=random_seed)
        else:
            raise


def mclust_with_spatial_refinement(
    adata,
    n_clusters: int,
    use_rep: str = 'X_spatial_domain',
    spatial_key: str = 'spatial',
    key_added: str = 'mclust',
    modelNames: str = 'EEE',
    refine_boundary: bool = True,
    random_seed: int = 2020
):
    """mclust 聚类 + 空间边界细化"""
    # 初始聚类
    adata = mclust_R(adata, n_clusters, use_rep, key_added, modelNames, random_seed)
    
    if refine_boundary and spatial_key in adata.obsm:
        print("\n🔧 细化聚类边界（基于空间邻域）...")
        
        from sklearn.neighbors import NearestNeighbors
        from scipy.stats import mode
        
        labels = adata.obs[key_added].values. astype(int)
        spatial_coords = adata.obsm[spatial_key]
        
        # 识别边界点
        if f'{key_added}_uncertainty' in adata.obs. columns:
            uncertainty = adata.obs[f'{key_added}_uncertainty'].values
            threshold = np.percentile(uncertainty, 75)
            low_conf_mask = uncertainty > threshold
        else:
            # 启发式：邻居标签不一致的点
            nbrs = NearestNeighbors(n_neighbors=7).fit(spatial_coords)
            _, indices = nbrs.kneighbors(spatial_coords)
            
            low_conf_mask = np.zeros(len(labels), dtype=bool)
            for i, neighbors in enumerate(indices):
                neighbor_labels = labels[neighbors[1: ]]
                most_common = mode(neighbor_labels, keepdims=True).mode[0]
                if labels[i] != most_common: 
                    low_conf_mask[i] = True
        
        if low_conf_mask.sum() > 0:
            print(f"   - 识别 {low_conf_mask.sum()} 个边界样本")
            
            nbrs = NearestNeighbors(n_neighbors=7).fit(spatial_coords)
            _, indices = nbrs.kneighbors(spatial_coords[low_conf_mask])
            
            refined_labels = labels. copy()
            n_refined = 0
            
            for i, (idx, neighbors) in enumerate(zip(np.where(low_conf_mask)[0], indices)):
                neighbor_labels = labels[neighbors[1:]]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                majority_label = unique[counts.argmax()]
                
                if majority_label != labels[idx]:
                    refined_labels[idx] = majority_label
                    n_refined += 1
            
            print(f"   - 重新分配 {n_refined} 个样本")
            
            adata.obs[key_added] = refined_labels
            adata.obs[key_added] = adata.obs[key_added].astype('int')
            adata.obs[key_added] = adata.obs[key_added].astype('category')
        else:
            print(f"   - 无需细化")
    
    return adata


def test_mclust_sedr_style():
    """测试"""
    print("=" * 60)
    print("测试 mclust")
    print("=" * 60)
    
    import scanpy as sc
    from sklearn.datasets import make_blobs
    
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)
    
    adata = sc.AnnData(X)
    adata.obsm['X_test'] = X
    
    print(f"\n生成测试数据:  {X.shape}")
    
    try:
        adata = mclust_R(adata, n_clusters=4, use_rep='X_test', key_added='mclust', modelNames='EEE')
        
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y_true, adata.obs['mclust'])
        print(f"\n   ARI: {ari:.3f}")
        
        print("\n✅ 测试通过！")
        return True
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_mclust_sedr_style()