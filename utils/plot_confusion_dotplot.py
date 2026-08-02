#!/usr/bin/env python3
"""
Dot plot of a soft confusion matrix between ground truth and predicted cell labels.

Each dot:
  - position: (gt_label, pred_label)
  - size:     fraction of gt cells predicted as that label
  - color:    same fraction (for emphasis)

Usage:
  python3 plot_confusion_dotplot.py \
      --gt    BG/BG_human_10K_obs.tsv \
      --gt_col Subclass \
      --pred  Zhuang-ABCA-1_shuffled/BG_human_10K.prediction.tsv \
      --pred_col prediction_by_subclass_top1 \
      --out   confusion_dotplot.png \
      --min_frac 0.02
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--gt",       required=True, help="Ground truth obs TSV (tab-separated, first col = index)")
parser.add_argument("--gt_col",   required=True, help="Column name in GT file to use as ground truth labels")
parser.add_argument("--pred",     required=True, help="Prediction TSV from predict.py")
parser.add_argument("--pred_col", required=True, help="Column name in prediction file to use")
parser.add_argument("--out",      default="confusion_dotplot.png", help="Output image path")
parser.add_argument("--min_frac", type=float, default=0.02,
                    help="Min fraction of a GT class to show a dot (default: 0.02)")
parser.add_argument("--min_gt_cells", type=int, default=10,
                    help="Min number of cells in a GT class to include it (default: 10)")
parser.add_argument("--figsize",  nargs=2, type=float, default=None,
                    help="Figure size in inches (width height). Auto-sized if omitted.")
parser.add_argument("--max_dot",  type=float, default=800,
                    help="Max dot area in points^2 (default: 800)")
args = parser.parse_args()

# ── load ──────────────────────────────────────────────────────────────────────
gt_df   = pd.read_csv(args.gt,   sep="\t", index_col=0)
pred_df = pd.read_csv(args.pred, sep="\t", comment="#")

gt   = gt_df[args.gt_col].astype(str).values
pred = pred_df[args.pred_col].astype(str).values

assert len(gt) == len(pred), f"Row count mismatch: {len(gt)} vs {len(pred)}"

# ── build confusion matrix (row-normalized by GT class) ───────────────────────
gt_labels   = sorted(set(gt),   key=lambda x: -np.sum(gt == x))   # sort by freq
gt_labels   = [g for g in gt_labels if np.sum(gt == g) >= args.min_gt_cells]
pred_labels_all = sorted(set(pred), key=lambda x: -np.sum(pred == x))

# only keep pred labels that appear above min_frac in at least one GT class
rows = []
for g in gt_labels:
    mask = gt == g
    n = mask.sum()
    counts = pd.Series(pred[mask]).value_counts()
    fracs  = counts / n
    for p_label, frac in fracs.items():
        if frac >= args.min_frac:
            rows.append({"gt": g, "pred": p_label, "frac": frac, "count": int(counts[p_label])})

df = pd.DataFrame(rows)

# order pred labels by total weight across all GT classes
pred_order = (df.groupby("pred")["frac"].sum().sort_values(ascending=False).index.tolist())

# ── plot ──────────────────────────────────────────────────────────────────────
n_gt   = len(gt_labels)
n_pred = len(pred_order)

if args.figsize:
    figw, figh = args.figsize
else:
    figw = max(10, n_pred * 0.45 + 3)
    figh = max(6,  n_gt   * 0.45 + 3)

fig, ax = plt.subplots(figsize=(figw, figh))

gt_idx   = {g: i for i, g in enumerate(gt_labels)}
pred_idx = {p: i for i, p in enumerate(pred_order)}

x = df["pred"].map(pred_idx).values
y = df["gt"].map(gt_idx).values
s = df["frac"].values

scatter = ax.scatter(
    x, y,
    s=s * args.max_dot,
    c=s,
    cmap="Blues",
    vmin=0, vmax=1,
    edgecolors="grey", linewidths=0.3,
    alpha=0.85,
)

plt.colorbar(scatter, ax=ax, label="Fraction of GT class")

ax.set_xticks(range(n_pred))
ax.set_xticklabels(pred_order, rotation=60, ha="right", fontsize=14)
gt_counts = {g: int(np.sum(gt == g)) for g in gt_labels}
gt_tick_labels = [f"{g} (n={gt_counts[g]:,})" for g in gt_labels]
ax.set_yticks(range(n_gt))
ax.set_yticklabels(gt_tick_labels, fontsize=14)
ax.set_xlabel("Predicted label", fontsize=15)
ax.set_ylabel("Ground truth label", fontsize=15)

ax2 = ax.twiny()
ax2.set_xticks(range(n_pred))
ax2.set_xticklabels(pred_order, rotation=60, ha="left", fontsize=14, rotation_mode="anchor")
ax.set_title(f"{args.gt_col}  vs  {args.pred_col}\n(dots ≥ {args.min_frac:.0%} of GT class)", fontsize=13)
ax.set_xlim(-0.5, n_pred - 0.5)
ax2.set_xlim(-0.5, n_pred - 0.5)
ax.set_ylim(-0.5, n_gt - 0.5)
ax.grid(True, linewidth=0.3, alpha=0.4)

# legend for dot size
for frac in [0.25, 0.5, 1.0]:
    ax.scatter([], [], s=frac * args.max_dot, c="steelblue", alpha=0.7,
               label=f"{frac:.0%}", edgecolors="grey", linewidths=0.3)
ax.legend(title="Fraction", loc="upper right", framealpha=0.7, fontsize=8)

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved → {args.out}")
