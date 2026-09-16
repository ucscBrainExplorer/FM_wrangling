from __future__ import annotations

"""
Plot a saved 2D UMAP/PUMAP embedding, colored by one or more obs columns.

Works on the output of run_umap.py, or any other (n_cells, 2) .npy array
(e.g. a PUMAP embedding) — as long as its row order matches an obs table.

Usage:
    python plot_umap.py --umap umap.npy --obs mouse_mammary_epi_noPal_obs.tsv.gz \
        --color-by Annotation Dataset Stage

    # custom category order + output naming
    python plot_umap.py --umap umap_e30.npy --obs mouse_mammary_epi_noPal_obs.tsv.gz \
        --color-by Stage \
        --order "Stage=Embryonic,Neonatal,Virgin adult,Pregnant,Lactating,Involuted,Menopause" \
        --output-prefix umap_e30

Output: one PNG per --color-by column, named "<output-prefix>_<column>.png"
(output-prefix defaults to the --umap filename stem).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_order(order_args: list[str] | None) -> dict[str, list[str]]:
    """Parse repeated --order "Column=val1,val2,..." args into a dict."""
    orders = {}
    for item in order_args or []:
        col, _, values = item.partition("=")
        if not values:
            raise ValueError(f"--order must be 'Column=val1,val2,...', got: {item!r}")
        orders[col] = [v.strip() for v in values.split(",")]
    return orders


def plot_one(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str,
    category_order: list[str] | None = None,
    point_size: float = 0.5,
    alpha: float = 0.5,
) -> None:
    present = set(labels)
    if category_order:
        categories = [c for c in category_order if c in present]
        categories += sorted(present - set(categories))  # any unlisted values, appended
    else:
        categories = sorted(present)

    cmap_name = "tab10" if len(categories) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name, len(categories))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=point_size, alpha=alpha, rasterized=True)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c], markersize=6, label=c)
        for c in categories
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--umap", required=True, help="Path to a 2D UMAP/PUMAP .npy embedding")
    parser.add_argument("--obs", required=True, help="Path to the matching obs .tsv/.tsv.gz")
    parser.add_argument("--color-by", nargs="+", required=True, help="One or more obs columns to color by")
    parser.add_argument(
        "--order",
        nargs="+",
        default=None,
        help='Optional category order per column, e.g. --order "Stage=A,B,C"',
    )
    parser.add_argument("--output-prefix", default=None, help="Output filename prefix (default: umap filename stem)")
    parser.add_argument("--point-size", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    embedding = np.load(args.umap)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError(f"Expected a (n_cells, 2) embedding, got shape {embedding.shape}")

    obs = pd.read_csv(args.obs, sep="\t")
    if len(obs) != len(embedding):
        raise ValueError(f"obs rows ({len(obs)}) != embedding rows ({len(embedding)})")

    orders = parse_order(args.order)
    prefix = args.output_prefix or os.path.splitext(os.path.basename(args.umap))[0]

    for col in args.color_by:
        labels = obs[col].astype(str).values
        plot_one(
            embedding,
            labels,
            title=f"UMAP colored by {col}",
            output_path=f"{prefix}_{col}.png",
            category_order=orders.get(col),
            point_size=args.point_size,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
