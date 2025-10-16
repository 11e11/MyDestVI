import scanpy as sc
import scvi
from src import DestVI

# 完全按照原始DestVI的使用方式
sc_adata = sc.read_h5ad("path_to_scRNA_data.h5ad")
st_adata = sc.read_h5ad("path_to_spatial_data.h5ad")

# 训练CondSCVI
scvi.model.CondSCVI.setup_anndata(sc_adata, labels_key="cell_type")
sc_model = scvi.model.CondSCVI(sc_adata)
sc_model.train(max_epochs=400)

# 使用独立的DestVI
DestVI.setup_anndata(st_adata)
spatial_model = DestVI.from_rna_model(st_adata, sc_model)
spatial_model.train(max_epochs=2000)

# 获取结果
proportions = spatial_model.get_proportions()
gamma = spatial_model.get_gamma()

print("DestVI training completed!")
print(f"Proportions shape: {proportions.shape}")