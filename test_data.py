# from __future__ import annotations

# import argparse
# from collections import defaultdict
# import os
# import time
# import importlib
# import traceback
# from typing import Optional, Tuple

# import numpy as np
# import pandas as pd
# from pandas.api.types import CategoricalDtype
# import scanpy as sc
# import torch
# from torch.utils.data import DataLoader

# # 引入你写的 Dataset
# from src.data import SingleCellDataset, SpatialDataset
# from src.models.mycondscvi import CondSCVI
# from src.models.mydestvi import DestVI
# # from src.models import CondSCVI, DestVI


# def move_item_to_device(item: dict, device: torch.device) -> dict:
#     out = {}
#     for k, v in item.items():
#         if torch.is_tensor(v):
#             out[k] = v.to(device)
#         else:
#             out[k] = v
#     return out


# def train_one_epoch_sc(module: torch.nn.Module, dataloader: DataLoader, optimizer, device, verbose=True) -> float:
#     module.train()
#     total_loss = 0.0
#     n_batches = 0
#     for i, item in enumerate(dataloader, 1):
#         item = move_item_to_device(item, device)  # item = {'X': [B,G], 'labels': [B], optional 'batch':[B]}
#         out = module.forward(item)
#         loss = out["loss"]
#         loss_val = loss if loss.dim() == 0 else loss.mean()

#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()

#         total_loss += float(loss_val.detach().cpu().item())
#         n_batches += 1
#         # if verbose and i % 50 == 0:
#         #     print(f"  batch {i} loss={loss_val.detach().cpu().item():.4f}")
#     return total_loss / max(1, n_batches)


# # [--- M-2: 替换这个函数 ---]
# def train_one_epoch_st(module: torch.nn.Module, dataloader: DataLoader, optimizer, device, verbose=True) -> dict:
#     """
#     修改：此函数现在返回一个字典，包含所有损失项的平均值。
#     """
#     module.train()
#     # 使用 defaultdict 来自动累积所有损失项
#     epoch_losses = defaultdict(float)
#     n_batches = 0
    
#     for i, item in enumerate(dataloader, 1):
#         item = move_item_to_device(item, device)  # item = {'X':[B,G], 'ind_x':[B], optional 'spatial':[B,2]}
        
#         # out 是 mymrdeconv.forward() 返回的字典
#         # 包含 {'loss', 'reconstruction_loss', 'kl_local', 'kl_global', 'mmd'}
#         out = module.forward(item) 
        
#         loss = out["loss"]
#         loss_val = loss if loss.dim() == 0 else loss.mean()

#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()

#         # 累积所有返回的损失项
#         for key, value in out.items():
#             # 确保值是标量
#             if value.dim() > 0:
#                 value = value.mean()
#             epoch_losses[key] += float(value.detach().cpu().item())
            
#         n_batches += 1
#         # if verbose and i % 50 == 0:
#         #     # 仅打印主损失
#         #     print(f"  batch {i} loss={loss_val.detach().cpu().item():.4f}")

#     # 计算所有损失项的平均值
#     avg_epoch_losses = {key: total / max(1, n_batches) for key, total in epoch_losses.items()}
    
#     return avg_epoch_losses


# def main(argv=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sc-h5ad", required=True)
#     parser.add_argument("--st-h5ad", required=True)
#     parser.add_argument("--device", default="cpu")
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--batch-size", type=int, default=128)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--layer", default="counts")
#     parser.add_argument("--labels-key", default="free_annotation")
#     args = parser.parse_args(argv)

#     device = torch.device(args.device)

#     # CondSCVI = safe_import_condscvi()
#     # DestVI = safe_import_destvi()

#     print("Loading sc AnnData:", args.sc_h5ad)
#     sc_adata = sc.read_h5ad(args.sc_h5ad)
#     print(sc_adata)

#     print("Loading st AnnData:", args.st_h5ad)
#     st_adata = sc.read_h5ad(args.st_h5ad)
#     print(st_adata)

#     # 构建自定义数据集（注意：SingleCellDataset 要求 labels 必须是 categorical）
#     sc_dataset = SingleCellDataset(sc_adata, label_key=args.labels_key, use_layer=args.layer)
#     st_dataset = SpatialDataset(st_adata, use_layer=args.layer, spatial_key="spatial")

#     # 直接用 SC 的编码顺序构建 mapping（与 SingleCellDataset.label_names 完全一致）
#     label_names_sc = list(map(str, sc_dataset.label_names))
#     cell_type_mapping = {name: i for i, name in enumerate(label_names_sc)}
#     print("cell_type_mapping（与SC编码顺序一致）:", cell_type_mapping)

#     # DataLoader（默认 collate 会把 dict 的每个键堆叠为 batch tensor）
#     sc_loader = DataLoader(sc_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
#     st_loader = DataLoader(st_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

#     # 实例化 CondSCVI
#     n_input = sc_dataset.X.shape[1]
#     n_labels = int(sc_dataset.n_labels)
#     print(f"Instantiating CondSCVI: n_input={n_input}, n_labels={n_labels}")
#     cond_model = CondSCVI(n_input=n_input, n_labels=n_labels, n_batch=getattr(sc_dataset, "n_batch", 1))
#     cond_module = getattr(cond_model, "module", cond_model)
#     cond_module.to(device)

#     # CondSCVI 优化器
#     cond_params = [p for p in cond_module.parameters() if p.requires_grad]
#     cond_opt = torch.optim.Adam(cond_params, lr=args.lr)

#     # 训练 CondSCVI
#     print("Training CondSCVI for", args.epochs, "epochs on sc data")
#     for ep in range(args.epochs):
#         t0 = time.time()
#         avg_loss = train_one_epoch_sc(cond_module, sc_loader, cond_opt, device)
#         t1 = time.time()
#         print(f"[CondSCVI] Epoch {ep+1}/{args.epochs} avg_loss={avg_loss:.4f} time={t1-t0:.1f}s")

#     # 提取 decoder / px_decoder state_dict
#     decoder_state = cond_module.decoder.state_dict() if hasattr(cond_module, "decoder") else None
#     px_decoder_state = cond_module.px_decoder.state_dict() if hasattr(cond_module, "px_decoder") else None

#     # 组装 DestVI 参数（对齐 CondSCVI 结构）
#     n_spots = int(st_dataset.X.shape[0])
#     n_genes = int(st_dataset.X.shape[1])
#     n_latent = int(getattr(cond_module, "n_latent", 5))
#     n_hidden = int(cond_module.px_decoder[0].in_features) if hasattr(cond_module, "px_decoder") else 128
#     n_layers_dec = int(getattr(cond_module, "n_layers", 2))
#     dropout_dec = float(getattr(cond_module, "dropout_rate", 0.05))

#     destvi_kwargs = dict(
#         l1_reg= 10.0,
#         dirichlet_alpha= 0.4,
#         dirichlet_mmd_reg=2.0,
#         n_spots=n_spots,
#         n_genes=n_genes,
#         n_labels=n_labels,
#         n_latent=n_latent,
#         n_hidden=n_hidden,
#         n_layers=n_layers_dec,
#         decoder_state_dict=decoder_state,
#         px_decoder_state_dict=px_decoder_state,
#         px_r=(cond_module.px_r.detach().cpu().numpy() if hasattr(cond_module, "px_r") and cond_module.px_r.numel() == n_genes else np.ones(n_genes, dtype=np.float32)),
#         cell_type_mapping=cell_type_mapping,
#         dropout_decoder=dropout_dec
        
#     )

#     print("Instantiating DestVI for ST training")
#     destvi_model = DestVI(**destvi_kwargs)
#     dest_module = getattr(destvi_model, "module", destvi_model)
#     dest_module.to(device)

#     # 冻结解码器（若 DestVI 未默认冻结）
#     try:
#         if hasattr(dest_module, "decoder"):
#             for p in dest_module.decoder.parameters():
#                 p.requires_grad = False
#         if hasattr(dest_module, "px_decoder"):
#             for p in dest_module.px_decoder.parameters():
#                 p.requires_grad = False
#     except Exception:
#         pass

#     # ST 优化器
#     st_params = [p for p in dest_module.parameters() if p.requires_grad]
#     st_opt = torch.optim.Adam(st_params, lr=args.lr)

#     # 训练 DestVI
#     print("Training DestVI for", args.epochs, "epochs on ST data")
#     for ep in range(args.epochs):
#         t0 = time.time()
        
#         # [--- M-2: 修改 main 循环以处理字典 ---]
#         # 1. 捕获损失字典 (avg_loss_dict 现在是一个 dict)
#         avg_loss_dict = train_one_epoch_st(dest_module, st_loader, st_opt, device)
#         t1 = time.time()

#         # 2. 格式化打印
#         # (确保 'loss' 总是第一个被打印)
#         main_loss = avg_loss_dict.get('loss', 0.0)
#         loss_strings = [f"avg_loss={main_loss:.4f}"]
        
#         # 添加其他所有损失项
#         for key, value in avg_loss_dict.items():
#             if key != 'loss':
#                 # 格式化所有其他损失项
#                 loss_strings.append(f"{key}={value:.4f}")
        
#         print(f"[DestVI] Epoch {ep+1}/{args.epochs} " + ", ".join(loss_strings) + f" time={t1-t0:.1f}s")

#     # 保存权重
#     out_dir = "test_model_states"
#     os.makedirs(out_dir, exist_ok=True)
#     try:
#         torch.save(cond_module.state_dict(), os.path.join(out_dir, "condscvi_module.pt"))
#         torch.save(dest_module.state_dict(), os.path.join(out_dir, "destvi_module.pt"))
#         print("Saved module state_dicts to", out_dir)
#     except Exception as e:
#         print("Warning: could not save state_dicts:", e)


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import argparse
from collections import defaultdict
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

# 数据集与模型
from src.data import SingleCellDataset, SpatialDataset
from src.models.mycondscvi import CondSCVI
from src.models.mydestvi import DestVI

"""
python test_data.py \
  --sc-h5ad ../data/simulation_Mouse_Kidney_MERFISH/MERFISH_kidney_object.h5ad \
  --st-h5ad ../data/simulation_Mouse_Kidney_MERFISH/MK_simulated_ST.h5ad\
  --device cuda \
  --epochs-sc 20 \
  --epochs-st 50 \
  --batch-size 128 \
  --log-file log2.txt \
  --prop-csv mydestvi_predicted_proportions_mmdAndL1.csv
  模型不收敛
"""

def move_item_to_device(item: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}


def train_one_epoch_sc(module: torch.nn.Module,
                       dataloader: DataLoader,
                       optimizer,
                       device: torch.device) -> float:
    module.train()
    total_loss = 0.0
    n_batches = 0
    for item in dataloader:
        item = move_item_to_device(item, device)
        out = module.forward(item)
        loss = out["loss"]
        loss_val = loss if loss.dim() == 0 else loss.mean()
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        total_loss += float(loss_val.detach().cpu().item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def train_one_epoch_st(module: torch.nn.Module,
                       dataloader: DataLoader,
                       optimizer,
                       device: torch.device) -> dict:
    module.train()
    epoch_losses = defaultdict(float)
    n_batches = 0
    for item in dataloader:
        item = move_item_to_device(item, device)
        out = module.forward(item)  # {'loss','reconstruction_loss','kl_local','kl_global','mmd'}
        loss = out["loss"]
        loss_val = loss if loss.dim() == 0 else loss.mean()
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        for k, v in out.items():
            v_scalar = v if (not torch.is_tensor(v) or v.dim() == 0) else v.mean()
            epoch_losses[k] += float(v_scalar.detach().cpu().item())
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in epoch_losses.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc-h5ad", required=True)
    parser.add_argument("--st-h5ad", required=True)
    parser.add_argument("--device", default="cpu")
    # 新增：独立 epoch 参数
    parser.add_argument("--epochs", type=int, default=None, help="全局默认 epoch；若提供 --epochs-sc/--epochs-st 将分别覆盖")
    parser.add_argument("--epochs-sc", type=int, default=None, help="CondSCVI 训练轮数")
    parser.add_argument("--epochs-st", type=int, default=None, help="DestVI 训练轮数")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer", default="counts")
    parser.add_argument("--labels-key", default="free_annotation")
    parser.add_argument("--log-file", default="log.txt")
    parser.add_argument("--prop-csv", default="destvi_predicted_proportions_mmdAndL1.csv")
    args = parser.parse_args()

    # 解析独立/全局 epoch
    epochs_sc = args.epochs_sc if args.epochs_sc is not None else (args.epochs if args.epochs is not None else 3)
    epochs_st = args.epochs_st if args.epochs_st is not None else (args.epochs if args.epochs is not None else 3)

    # 打开日志文件（覆盖写）
    os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
    log_f = open(args.log_file, "w", encoding="utf-8")

    def log(msg: str):
        log_f.write(msg + "\n")
        log_f.flush()

    device = torch.device(args.device)
    log(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"参数: {args}")
    log(f"实际使用 epochs_sc={epochs_sc}, epochs_st={epochs_st}")

    # 读取 AnnData
    log(f"Loading sc AnnData: {args.sc_h5ad}")
    sc_adata = sc.read_h5ad(args.sc_h5ad)
    log(str(sc_adata))

    log(f"Loading st AnnData: {args.st_h5ad}")
    st_adata = sc.read_h5ad(args.st_h5ad)
    log(str(st_adata))

    # 构建数据集
    sc_dataset = SingleCellDataset(sc_adata, label_key=args.labels_key, use_layer=args.layer)
    st_dataset = SpatialDataset(st_adata, use_layer=args.layer, spatial_key="spatial")

    label_names_sc = list(map(str, sc_dataset.label_names))
    cell_type_mapping = {name: i for i, name in enumerate(label_names_sc)}
    log(f"cell_type_mapping（与SC编码顺序一致）: {cell_type_mapping}")

    # DataLoader
    sc_loader = DataLoader(sc_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    st_loader = DataLoader(st_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # CondSCVI
    n_input = sc_dataset.X.shape[1]
    n_labels = int(sc_dataset.n_labels)
    log(f"Instantiating CondSCVI: n_input={n_input}, n_labels={n_labels}")
    cond_model = CondSCVI(n_input=n_input, n_labels=n_labels, n_batch=getattr(sc_dataset, "n_batch", 1))
    cond_module = getattr(cond_model, "module", cond_model)
    cond_module.to(device)

    cond_params = [p for p in cond_module.parameters() if p.requires_grad]
    cond_opt = torch.optim.Adam(cond_params, lr=args.lr)

    log(f"Training CondSCVI for {epochs_sc} epochs on sc data")
    for ep in range(epochs_sc):
        t0 = time.time()
        avg_loss = train_one_epoch_sc(cond_module, sc_loader, cond_opt, device)
        t1 = time.time()
        log(f"[CondSCVI] Epoch {ep+1}/{epochs_sc} avg_loss={avg_loss:.4f} time={t1-t0:.1f}s")

    # 提取解码器权重
    decoder_state = cond_module.decoder.state_dict() if hasattr(cond_module, "decoder") else None
    px_decoder_state = cond_module.px_decoder.state_dict() if hasattr(cond_module, "px_decoder") else None

    # DestVI 参数
    n_spots = int(st_dataset.X.shape[0])
    n_genes = int(st_dataset.X.shape[1])
    n_latent = int(getattr(cond_module, "n_latent", 5))
    n_hidden = int(cond_module.px_decoder[0].in_features) if hasattr(cond_module, "px_decoder") else 128
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
        # 你设置的正则/MMD
        l1_reg=10.0,
        dirichlet_alpha=0.4,
        dirichlet_mmd_reg=2.0,
    )

    log("Instantiating DestVI for ST training")
    destvi_model = DestVI(**destvi_kwargs)
    dest_module = getattr(destvi_model, "module", destvi_model)
    dest_module.to(device)

    # 冻结共享解码器
    try:
        if hasattr(dest_module, "decoder"):
            for p in dest_module.decoder.parameters():
                p.requires_grad = False
        if hasattr(dest_module, "px_decoder"):
            for p in dest_module.px_decoder.parameters():
                p.requires_grad = False
    except Exception as e:
        log(f"冻结解码器时出现异常: {e}")

    st_params = [p for p in dest_module.parameters() if p.requires_grad]
    st_opt = torch.optim.Adam(st_params, lr=args.lr)

    log(f"Training DestVI for {epochs_st} epochs on ST data")
    for ep in range(epochs_st):
        t0 = time.time()
        avg_loss_dict = train_one_epoch_st(dest_module, st_loader, st_opt, device)
        t1 = time.time()
        main_loss = avg_loss_dict.get("loss", 0.0)
        others = ", ".join([f"{k}={v:.4f}" for k, v in avg_loss_dict.items() if k != "loss"])
        log(f"[DestVI] Epoch {ep+1}/{epochs_st} avg_loss={main_loss:.4f}, {others} time={t1-t0:.1f}s")

    # 预测比例
    log("Computing DestVI predicted proportions for full ST dataset...")
    with torch.no_grad():
        X_st_full = st_dataset.X.to(device)
        props = dest_module.get_proportions(x=X_st_full, keep_noise=False)
        if torch.is_tensor(props):
            props_np = props.detach().cpu().numpy()
        else:
            props_np = np.asarray(props)

    # 保存到 AnnData
    st_adata.obsm["proportions"] = props_np

    # 保存 CSV（行：spot，列：细胞类型）
    columns = label_names_sc[:props_np.shape[1]]
    prop_df = pd.DataFrame(props_np, index=st_adata.obs_names, columns=columns)
    prop_csv_path = args.prop_csv
    prop_df.to_csv(prop_csv_path, index=True)
    log(f"Saved predicted proportions to {prop_csv_path}")

    # 基本统计
    col_means = prop_df.mean(axis=0)
    log("Predicted proportions mean per cell type:")
    for ct, val in col_means.items():
        log(f"  {ct}: {val:.4f}")
    row_sum_mean = prop_df.sum(axis=1).mean()
    log(f"Average row sum (should be ~1): {row_sum_mean:.4f}")

    # 保存模型权重
    out_dir = "test_model_states"
    os.makedirs(out_dir, exist_ok=True)
    try:
        torch.save(cond_module.state_dict(), os.path.join(out_dir, "condscvi_module.pt"))
        torch.save(dest_module.state_dict(), os.path.join(out_dir, "destvi_module.pt"))
        log(f"Saved model state_dicts to {out_dir}")
    except Exception as e:
        log(f"Warning: could not save state_dicts: {e}")

    # 可选：保存带预测比例的 ST AnnData
    try:
        st_adata.write(os.path.join(out_dir, "st_with_proportions.h5ad"))
        log("Saved updated ST AnnData with proportions.")
    except Exception as e:
        log(f"Warning: could not write updated ST AnnData: {e}")

    log("训练完成。")
    log_f.close()


if __name__ == "__main__":
    main()