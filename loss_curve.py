import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 以脚本所在目录为基准
BASE = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(BASE, 'results', 'log_dual_graph_PDAC_pseudo.txt')

with open(log_path, 'r', encoding='utf-8') as f:
    txt = f.read()

# ---- 抽数 ----
cond_pattern = re.compile(r'\[CondSCVI\] Epoch (\d+)/\d+ loss=([\d\.]+)')
dest_pattern = re.compile(r'\[DestVI\] Epoch (\d+)/\d+ avg_loss=([\d\.]+)')
cond = [(int(e), float(l)) for e, l in cond_pattern.findall(txt)]
dest = [(int(e), float(l)) for e, l in dest_pattern.findall(txt)]

cond_df = pd.DataFrame(cond, columns=['epoch', 'loss'])
dest_df = pd.DataFrame(dest, columns=['epoch', 'loss'])

dest_df = dest_df[9:]
# ---- 画图 ----
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)

# 子图 1：CondSCVI
axes[0].plot(cond_df.epoch, cond_df.loss, color='tab:blue', linewidth=1.2)
axes[0].set_ylabel('Loss')
axes[0].set_title('CondSCVI Training Curve')
axes[0].grid(alpha=0.3)

# 子图 2：DestVI（y 轴用 log 更直观）
axes[1].plot(dest_df.epoch, dest_df.loss, color='tab:orange', linewidth=1.2)
axes[1].set_yscale('log')          # 对数坐标
axes[1].set_ylabel('Loss (log scale)')
axes[1].set_xlabel('Epoch')
axes[1].set_title('DestVI Training Curve')
axes[1].grid(alpha=0.3)

plt.tight_layout()
out_png = log_path.replace('.txt', '_loss_curve_subplots.png')
plt.savefig(out_png, dpi=300)
plt.show()
print(f"Subplot figure saved to {out_png}")