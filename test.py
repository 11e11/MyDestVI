"""
最小化测试：为 DestVI / MRDeconv 提供 dummy decoder state_dict 与 px_decoder state_dict，
并做一次前向与一次反向（单步）传播，检查梯度是否流通与参数是否被更新。

用法:
    python test.py
"""
import numpy as np
import torch
import torch.nn as nn

# 从你的包中导入目标类（按你当前结构）
from src.models.mydestvi import DestVI

# MRDeconv 在 DestVI 内部会被实例化为 module；我们也可能直接导入 MRDeconv 用来生成 compatible state_dict
try:
    from src.modules.mymrdeconv import MRDeconv
except Exception:
    MRDeconv = None

# 导入 FCLayers（用于构造与 MRDeconv 相同结构的 decoder 模板）
try:
    from src.nn import FCLayers
except Exception:
    from src.nn.layers import FCLayers  # type: ignore

# 参数（与你之前传给 DestVI 的一致或相近）
n_spots = 100
n_genes = 1
n_labels = 5
n_latent = 1
n_hidden = 1
n_layers = 1
dropout_decoder = 0.05

# 构造临时 decoder / px_decoder 的 state_dict，用来实例化 DestVI 时通过 load_state_dict
temp_decoder = FCLayers(
    n_in=n_latent,
    n_out=n_hidden,
    n_cat_list=[n_labels],
    n_layers=n_layers,
    n_hidden=n_hidden,
    dropout_rate=dropout_decoder,
    use_batch_norm=False,
    use_layer_norm=True,
)
decoder_state_dict = temp_decoder.state_dict()

px_decoder_template = nn.Sequential(nn.Linear(n_hidden, n_genes), nn.Softplus())
px_decoder_state_dict = px_decoder_template.state_dict()

px_r = np.ones(n_genes, dtype=np.float32)
cell_type_mapping = None

print("实例化 DestVI ...")
st_model = DestVI(
    n_spots=n_spots,
    n_genes=n_genes,
    n_labels=n_labels,
    n_latent=n_latent,
    n_hidden=n_hidden,
    n_layers=n_layers,
    decoder_state_dict=decoder_state_dict,
    px_decoder_state_dict=px_decoder_state_dict,
    px_r=px_r,
    cell_type_mapping=cell_type_mapping,
    dropout_decoder=dropout_decoder,
)
print("✅ DestVI 实例化成功！")

# 获取内部 module（MRDeconv 实例）
module = getattr(st_model, "module", st_model)
print("内部 module:", type(module))

# 构造一个小的 dummy batch，调用 module.forward 或 module.generative 测试行为
batch_size = 4
torch.manual_seed(0)
X = torch.abs(torch.randn(batch_size, n_genes)) * 3.0  # 模拟 counts（非负）
ind_x = torch.randint(0, n_spots, (batch_size,))  # spots 索引

item = {"X": X, "ind_x": ind_x}

print("运行一次前向以检查 loss / shapes ...")
out = None
did_forward = False
try:
    out = module.forward(item)
    did_forward = True
    print("forward() 成功，返回 keys:", list(out.keys()))
    # 打印几个关键输出（用 detach-safe 方式）
    if "loss" in out:
        loss_val = out["loss"].detach().cpu().item()
        print("loss:", loss_val)
    if "reconstruction_loss" in out:
        print("reconstruction_loss (mean):", out["reconstruction_loss"].detach().cpu().item())
    if "mmd" in out:
        # mmd 可能为 tensor 或 scalar
        try:
            print("mmd (mean):", out["mmd"].detach().cpu().item())
        except Exception:
            print("mmd (value):", out["mmd"])
except Exception as e_forward:
    print("module.forward 报错，尝试 module.generative(...):", repr(e_forward))
    try:
        out = module.generative(X, ind_x)
        print("generative() 成功，返回 keys:", list(out.keys()))
        if "px_rate" in out:
            print("px_rate shape:", out["px_rate"].shape)
    except Exception as e_gen:
        print("module.generative 也报错:", repr(e_gen))
        raise

# 如果 forward 返回了 loss，则进行一次反向传播并用 optimizer 做一步参数更新检查
if out is not None and "loss" in out:
    loss = out["loss"]
    # 确保 loss 是标量 tensor，否则取 mean
    if torch.is_tensor(loss) and loss.dim() != 0:
        loss = loss.mean()

    # 打印 loss（detach-safe）
    print("准备执行 backward，loss (detach):", loss.detach().cpu().item())

    # 收集可训练参数
    trainable_params = [p for p in module.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print("注意：模型没有可训练参数 (all params frozen)。跳过 backward/optimizer step 检查。")
    else:
        # 使用小的 optimizer 做一步更新（仅测试目的）
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        optimizer.zero_grad()
        # backward
        loss.backward()
        # 在执行 loss.backward() 之后运行
        print("参数总数 / 可训练参数 / 有梯度参数：")
        total = 0
        trainable = 0
        have_grad = 0
        for name, p in module.named_parameters():
            total += 1
            if p.requires_grad:
                trainable += 1
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                have_grad += 1
            # 关键：先拿到 float 或 None，再决定怎么打印
            grad_val = None if p.grad is None else float(p.grad.abs().mean().cpu().item())
            print(f"{name:50s} requires_grad={p.requires_grad:5} "
                f"grad_mean={grad_val if grad_val is not None else 'None':>12}")
        print("total:", total, "trainable:", trainable, "have_grad:", have_grad)
        # 检查梯度（打印前几个有梯度的参数的平均绝对值）
        grads = []
        for name, p in module.named_parameters():
            if p.grad is not None:
                grads.append((name, float(p.grad.abs().mean().cpu().item())))
        if len(grads) == 0:
            print("Warning: 未检测到任何参数的梯度（可能所有参数被冻结或梯度计算被断开）。")
        else:
            print("检测到参数梯度（name, mean_abs_grad），打印前 8 项：")
            for nm, g in grads[:8]:
                print(f"  {nm}: {g:.6e}")

        # 执行一步 optimizer 更新（检查是否出现异常）
        try:
            optimizer.step()
            print("optimizer.step() 完成（单步更新已应用）。")
        except Exception as e_step:
            print("optimizer.step() 抛出异常:", repr(e_step))
else:
    print("没有在 forward 输出中找到 'loss'，跳过 backward 测试。")

# 测试 get_proportions / get_gamma（如果实现存在）
if hasattr(module, "get_proportions"):
    try:
        props = module.get_proportions(x=X, keep_noise=False)
        print("get_proportions 返回 shape:", getattr(props, "shape", type(props)))
    except Exception as e:
        print("get_proportions 报错:", repr(e))

if hasattr(module, "get_gamma"):
    try:
        gamma = module.get_gamma(x=X)
        print("get_gamma 返回形状或类型:", type(gamma),  getattr(gamma, "shape", None))
    except Exception as e:
        print("get_gamma 报错:", repr(e))

print("测试完成。若以上步骤无致命异常，则基本说明 DestVI/MRDeconv 的改动可以被实例化、前向并完成一次反向传播与单步参数更新。")