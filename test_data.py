from __future__ import annotations

import argparse
from collections import defaultdict
import os
import random
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from src.data import SingleCellDataset, SpatialDataset
from src.data.preprocessing import preprocess_sc_st
from src.models.mycondscvi import CondSCVI
from src.models.mydestvi import DestVI


"""
python test_data.py \
  --sc-h5ad ../data/simulation_Mouse_Kidney_MERFISH/MERFISH_kidney_object.h5ad \
  --st-h5ad ../data/simulation_Mouse_Kidney_MERFISH/MK_simulated_ST.h5ad\
  --device cuda \
  --epochs-sc 300 \
  --epochs-st 1500 \
  --batch-size 256 \
  --use-dual-graph \
  --k-spatial 6 \
  --k-expr 10 \
  --log-file ./results/log_dual_graph.txt \
  --prop-csv ./results/mydestvi_predicted_proportions_dual_graph.csv

python test_data.py \
  --sc-h5ad ../data/PDAC/reference_example.h5ad \
  --st-h5ad ../data/PDAC/spatial_example.h5ad \
  --device cuda \
  --epochs-sc 300 \
  --epochs-st 1500 \
  --batch-size 256 \
  --use-dual-graph \
  --k-spatial 6 \
  --k-expr 10 \
  --log-file ./results/log_dual_graph_PDAC.txt \
  --prop-csv ./results/mydestvi_predicted_proportions_dual_graph_PDAC.csv \
  --labels-key "celltype"
"""

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

def move_item_to_device(item: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}


def linear_warmup(step: int, total_warmup: int) -> float:
    if total_warmup <= 0:
        return 1.0
    return float(min(1.0, step / max(1, total_warmup)))


def train_one_epoch_sc(
        module: torch.nn.Module,
        dataloader: DataLoader,
        optimizer,
        device: torch.device,
        start_step: int,
        warmup_steps_kl: int
    ) -> tuple[dict, int]:
    """
    返回 ({'loss':..., 'reconstruction_loss':..., 'kl_local':..., 'kl_library':...}, new_step)
    """
    module.train()
    epoch_losses = defaultdict(float)
    n_batches = 0
    step = start_step

    for item in dataloader:
        item = move_item_to_device(item, device)
        kl_w = linear_warmup(step, warmup_steps_kl)

        out = module.forward(item, kl_weight=kl_w)

        loss = out["loss"]
        loss_val = loss if loss.dim() == 0 else loss.mean()

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # 累积每个 loss 项
        for k, v in out.items():
            # 如果是 tensor，就取 mean 再取 item
            if torch.is_tensor(v):
                epoch_losses[k] += float(v.detach().cpu().item())
            else:
                epoch_losses[k] += float(v)

        n_batches += 1
        step += 1

    # 计算平均值
    avg_losses = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
    return avg_losses, step



def train_one_epoch_st(module: torch.nn.Module,
                       dataloader: DataLoader,
                       optimizer,
                       device: torch.device,
                       n_obs:int,
                       start_step: int,
                       warmup_steps_kl: int,
                       warmup_steps_reg: int) -> tuple[dict, int]:
    """
    返回 (avg_loss_dict, new_step)
    """
    module.train()
    epoch_losses = defaultdict(float)
    n_batches = 0
    step = start_step
    for item in dataloader:
        item = move_item_to_device(item, device)
        kl_w = linear_warmup(step, warmup_steps_kl)
        reg_w = linear_warmup(step, warmup_steps_reg)
        out = module.forward(item, kl_weight=kl_w, reg_warmup=reg_w, n_obs=n_obs)  # 关键：传入 kl_weight / reg_warmup
        loss = out["loss"]
        loss_val = loss if loss.dim() == 0 else loss.mean()
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        for k, v in out.items():
            v_scalar = v if (not torch.is_tensor(v) or v.dim() == 0) else v.mean()
            epoch_losses[k] += float(v_scalar.detach().cpu().item())
        n_batches += 1
        step += 1
    avg = {k: v / n_batches for k, v in epoch_losses.items()}  # 不用 max(1, n_batches)
    return avg, step


def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc-h5ad", required=True)
    parser.add_argument("--st-h5ad", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs-sc", type=int, default=None)
    parser.add_argument("--epochs-st", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer", default="counts")
    parser.add_argument("--labels-key", default="free_annotation")
    parser.add_argument("--log-file", default="log.txt")
    parser.add_argument("--prop-csv", default="destvi_predicted_proportions_mmdAndL1.csv")
    
    # 🔥 新增双图参数
    parser.add_argument("--use-dual-graph", action="store_true", help="启用双图模式（空间图+表达图）")
    parser.add_argument("--k-spatial", type=int, default=6, help="空间图的 k-NN 邻居数")
    parser.add_argument("--k-expr", type=int, default=10, help="表达图的 k-NN 邻居数")
    parser.add_argument("--spatial-key", default="spatial", help="st_adata.obsm 中的空间坐标 key")
    parser.add_argument("--gat-hidden", type=int, default=128, help="GAT 隐藏层维度")
    parser.add_argument("--gat-heads", type=int, default=4, help="GAT 注意力头数（暂未使用）")
    
    args = parser.parse_args()

    epochs_sc = args.epochs_sc if args.epochs_sc is not None else (args.epochs if args.epochs is not None else 300)
    epochs_st = args.epochs_st if args.epochs_st is not None else (args.epochs if args.epochs is not None else 1000)

    os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
    log_f = open(args.log_file, "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    device = torch.device(args.device)
    log(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"参数: {args}")
    log(f"实际使用 epochs_sc={epochs_sc}, epochs_st={epochs_st}")
    
    # 🔥 双图模式提示
    if args.use_dual_graph:
        log(f"🔥 启用双图模式：k_spatial={args.k_spatial}, k_expr={args.k_expr}")
    else:
        log("📊 使用标准 FC 模式（无图结构）")

    # 读取数据
    log(f"Loading sc AnnData: {args.sc_h5ad}")
    sc_adata = sc.read_h5ad(args.sc_h5ad)
    log(str(sc_adata))

    log(f"Loading st AnnData: {args.st_h5ad}")
    st_adata = sc.read_h5ad(args.st_h5ad)
    log(str(st_adata))

    # ---------- 预处理数据-------------
    log("预处理 sc 和 st AnnData ...")
    sc_adata, st_adata, common_genes = preprocess_sc_st(sc_adata, st_adata)   # 可传递具体基因数参数
    log(f"预处理完毕，交集基因数={len(common_genes)}")
    # ---------- 预处理 end -------------

    if 'counts' not in st_adata.layers:
        from scipy.sparse import issparse
        if issparse(st_adata.X):
            st_adata.layers['counts'] = st_adata.X.copy()
        else:
            st_adata.layers['counts'] = st_adata.X.copy()

    # 2. 创建归一化+log1p 的 layer (用于 encoder)
    from scipy.sparse import issparse
    X_raw = st_adata.layers['counts']
    if issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float32)

    # Library size normalization
    library_size = X_raw.sum(axis=1, keepdims=True)
    X_normalized = X_raw / (library_size + 1e-6) * 10000

    # Log1p
    X_log = np.log1p(X_normalized)

    # 保存为新 layer
    st_adata.layers['normalized_log'] = X_log
    log(f"✅ 已创建 'normalized_log' layer（预处理）")

    # 构建数据集
    sc_dataset = SingleCellDataset(sc_adata, label_key=args.labels_key, use_layer=args.layer)
    st_dataset = SpatialDataset(st_adata, use_layer=args.layer, encoder_layer='normalized_log', spatial_key=args.spatial_key)

    label_names_sc = list(map(str, sc_dataset.label_names))
    cell_type_mapping = np.array(label_names_sc)  # 🔥 改为 numpy 数组（mydestvi.py 需要）
    log(f"cell_type_mapping（与SC编码顺序一致）: {cell_type_mapping}")

    # DataLoader
    sc_loader = DataLoader(sc_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    st_loader = DataLoader(st_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # CondSCVI
    n_input = sc_dataset.X.shape[1]
    n_labels = int(sc_dataset.n_labels)
    log(f"Instantiating CondSCVI: n_input={n_input}, n_labels={n_labels}")
    # 单切片 => n_batch=0
    cond_model = CondSCVI(n_input=n_input, n_labels=n_labels, n_batch=0)
    cond_module = getattr(cond_model, "module", cond_model).to(device)

    cond_params = [p for p in cond_module.parameters() if p.requires_grad]
    cond_opt = torch.optim.Adam(cond_params, lr=args.lr)

    # SC 训练
    warmup_steps_sc = int(0.3 * epochs_sc * len(sc_loader))
    step_sc = 0
    log(f"Training CondSCVI for {epochs_sc} epochs on sc data")

    for ep in range(epochs_sc):
        t0 = time.time()
        avg_losses, step_sc = train_one_epoch_sc(
            cond_module, sc_loader, cond_opt, device, step_sc, warmup_steps_sc
        )
        t1 = time.time()

        loss_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
        log(f"[CondSCVI] Epoch {ep+1}/{epochs_sc} {loss_str} time={t1-t0:.1f}s")


    # 提取冻结解码器权重
    decoder_state = None
    if hasattr(cond_model, "export_decoder_state"):
        decoder_state = cond_model.export_decoder_state()
    if decoder_state is None and hasattr(cond_module, "decoder_backbone"):
        decoder_state = cond_module.decoder_backbone.state_dict()

    px_decoder_state = None
    if hasattr(cond_model, "export_px_decoder_state"):
        px_decoder_state = cond_model.export_px_decoder_state()
    if px_decoder_state is None and hasattr(cond_module, "px_decoder"):
        px_decoder_state = cond_module.px_decoder.state_dict()

    if not isinstance(decoder_state, dict):
        raise RuntimeError("decoder_state is None：请确认 CondSCVI 使用固定条件版 VAEC 并已训练，且存在 decoder_backbone")
    if not isinstance(px_decoder_state, dict):
        raise RuntimeError("px_decoder_state is None：请确认 CondSCVI.module.px_decoder 存在")

    # 计算 VampPrior（p=15）
    log("Computing VampPrior from CondSCVI ...")
    vamp_p = 15
    mean_vprior, var_vprior, mp_vprior = cond_model.get_vamp_prior(sc_loader, p=vamp_p, device=device)

    # DestVI 参数
    n_spots = int(st_dataset.X_raw.shape[0])
    n_genes = int(st_dataset.X_raw.shape[1])
    n_latent = int(getattr(cond_module, "n_latent", 5))
    n_hidden = int(cond_module.px_decoder[0].in_features)
    n_layers_dec = int(getattr(cond_module, "n_layers", 2))
    dropout_dec = float(getattr(cond_module, "dropout_rate", 0.05))

    destvi_kwargs = dict(
        n_spots=n_spots,
        n_genes=n_genes,
        n_labels=n_labels,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers_dec,
        decoder_state_dict=decoder_state,
        px_decoder_state_dict=px_decoder_state,
        px_r=(cond_module.px_r.detach().cpu().numpy()
              if hasattr(cond_module, "px_r") and cond_module.px_r.numel() == n_genes
              else np.ones(n_genes, dtype=np.float32)),
        cell_type_mapping=cell_type_mapping,
        dropout_decoder=dropout_dec,
        # 传入 VampPrior
        mean_vprior=mean_vprior,
        var_vprior=var_vprior,
        mp_vprior=mp_vprior,
        # 正则基值（用 reg_warmup 退火）
        l1_reg=10.0,
        dirichlet_alpha=0.4,
        dirichlet_mmd_reg=2.0,
        # 🔥 双图参数
        use_gat=args.use_dual_graph,           # 只有启用双图时才设为 True
        use_dual_graph=args.use_dual_graph,    # 双图开关
        dual_graph_fusion="concat",            # 融合方式
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
    )

    log("Instantiating DestVI for ST training")
    destvi_model = DestVI(**destvi_kwargs)
    dest_module = getattr(destvi_model, "module", destvi_model).to(device)

    # 🔥 如果启用双图，附加图结构
    if args.use_dual_graph:
        log("🔥 构建双图结构（空间图 + 表达图）...")
        try:
            # 1. 附加全量 X
            destvi_model.attach_full_X(st_adata, layer=args.layer, encoder_layer='normalized_log')
            
            # 2. 构建并附加双图
            destvi_model.attach_dual_graph(
                st_adata, 
                k_spatial=args.k_spatial, 
                k_expr=args.k_expr, 
                spatial_key=args.spatial_key
            )
            log("✅ 双图构建成功！")
        except Exception as e:
            log(f"❌ 双图构建失败: {e}")
            raise

    # 冻结解码器
    try:
        for p in dest_module.decoder.parameters():
            p.requires_grad = False
        for p in dest_module.px_decoder.parameters():
            p.requires_grad = False
    except Exception as e:
        log(f"冻结解码器时出现异常: {e}")

    st_params = [p for p in dest_module.parameters() if p.requires_grad]
    st_opt = torch.optim.Adam(st_params, lr=args.lr)

    # ST 训练（KL 退火 40%，L1/MMD 退火 40%）
    warmup_steps_kl_st = int(0.8 * epochs_st * len(st_loader))
    warmup_steps_reg_st = int(0.8 * epochs_st * len(st_loader))
    step_st = 0

    log(f"Training DestVI for {epochs_st} epochs on ST data")
    for ep in range(epochs_st):
        t0 = time.time()
        avg_loss_dict, step_st = train_one_epoch_st(
            dest_module,
            st_loader,
            st_opt,
            device,
            st_dataset.X_raw.shape[0],      # n_obs 正确传入
            step_st,                    # start_step
            warmup_steps_kl_st,
            warmup_steps_reg_st
        )
        t1 = time.time()
        main_loss = avg_loss_dict.get("loss", 0.0)
        others = ", ".join([f"{k}={v:.4f}" for k, v in avg_loss_dict.items() if k != "loss"])
        log(f"[DestVI] Epoch {ep+1}/{epochs_st} avg_loss={main_loss:.4f}, {others} time={t1-t0:.1f}s")

    # 预测比例
    log("Computing DestVI predicted proportions for full ST dataset...")
    with torch.no_grad():
        if args.use_dual_graph:
            # 双图模式：直接调用（已缓存全量 X）
            props = dest_module.get_proportions(x=None, keep_noise=False)
        else:
            # FC 模式：需要传入 X
            X_st_full = st_dataset.X_raw.to(device)
            props = dest_module.get_proportions(x=X_st_full, keep_noise=False)
        
        props_np = props.detach().cpu().numpy() if torch.is_tensor(props) else np.asarray(props)

    st_adata.obsm["proportions"] = props_np

    # 保存 CSV
    columns = label_names_sc[:props_np.shape[1]]
    prop_df = pd.DataFrame(props_np, index=st_adata.obs_names, columns=columns)
    prop_df.to_csv(args.prop_csv, index=True)
    log(f"Saved predicted proportions to {args.prop_csv}")

    # 基本统计
    col_means = prop_df.mean(axis=0)
    log("Predicted proportions mean per cell type:")
    for ct, val in col_means.items():
        log(f"  {ct}: {val:.4f}")
    row_sum_mean = prop_df.sum(axis=1).mean()
    log(f"Average row sum (should be ~1): {row_sum_mean:.4f}")

    # 保存权重
    out_dir = "test_model_states"
    os.makedirs(out_dir, exist_ok=True)
    try:
        torch.save(cond_module.state_dict(), os.path.join(out_dir, "condscvi_module.pt"))
        torch.save(dest_module.state_dict(), os.path.join(out_dir, "destvi_module.pt"))
        log(f"Saved model state_dicts to {out_dir}")
    except Exception as e:
        log(f"Warning: could not save state_dicts: {e}")

    # 可选：保存带比例的 ST AnnData
    try:
        st_adata.write(os.path.join(out_dir, "st_with_proportions.h5ad"))
        log("Saved updated ST AnnData with proportions.")
    except Exception as e:
        log(f"Warning: could not write updated ST AnnData: {e}")

    log("训练完成。")
    log_f.close()


if __name__ == "__main__":
    main()