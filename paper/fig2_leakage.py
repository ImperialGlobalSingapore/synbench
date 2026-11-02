"""Figure 2 ("No free lunch"): pre-training leakage vs synthetic quality (MAUVE) and MIA AUC, Llama-3.2-1B.

x  log10(1 + leakage %) of the train split in RedPajama (results/leakage/redpajama_summary.csv,
   src/leakage/summarize_leakage.py).
y  MAUVE (mauve_star, synthetic train vs real test), max over DP-Gen / AUG-PE / DP-Dfs / EPSVec and
   eps in {0.5, 1, 2, 4}; MIA AUC (mean over trials, rounded to 2 decimals as in the paper), max over
   DP-Gen / AUG-PE and eps in {0.5, 1, 2, 4} (only the 4 datasets with MIA runs).
Dashed lines are least-squares fits; correlations (Pearson, Spearman with its exact permutation p) are printed.

    python paper/fig2_leakage.py
"""
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from common import DATASETS, DP_EPS, FIGS, leakage, mia_results, paper_mauve  # noqa: E402

ORDER = ["hallmarks_of_cancer", "N2C2_2008", "PsyTar", "DMSAFN", "Lc37Fnd", "ZSTFNS", "AYSLAX", "EURLEX"]
LABEL = {"PsyTar": "Psytar"}  # the figure's spelling
BLUE, RED = "#4479b4", "#c8463f"
# label placement: (series, dataset) -> (dx, dy in points, ha, va); unlisted labels go up-right
PLACE = {("Mauve", "ZSTFNS"): (-7, 3, "right", "bottom"), ("Mauve", "PsyTar"): (-7, -3, "right", "top"),
         ("Mauve", "hallmarks_of_cancer"): (0, 8, "center", "bottom"), ("Mauve", "DMSAFN"): (-6, 6, "right", "bottom"),
         ("Mauve", "Lc37Fnd"): (6, 4, "left", "bottom"), ("Mauve", "EURLEX"): (-8, 0, "right", "center"),
         ("Mauve", "N2C2_2008"): (-8, 2, "right", "bottom"), ("Mauve", "AYSLAX"): (-8, -2, "right", "top"),
         ("AUC", "PsyTar"): (8, 0, "left", "center"), ("AUC", "DMSAFN"): (0, -9, "center", "top"),
         ("AUC", "N2C2_2008"): (8, 0, "left", "center"), ("AUC", "AYSLAX"): (8, 0, "left", "center")}


def mauve(ds):
    vals = [v for m in ["dp-transformers", "aug-pe", "dp-diffusion", "EPSVec"] for e in DP_EPS
            if (v := paper_mauve(ds, m, e)) is not None]
    return max(vals) if vals else None


def auc(ds):
    try:
        return max(round(mia_results(ds, g)[f"e{e}"][0][4], 2) for g in ["augpe", "dp-transformers"] for e in DP_EPS)
    except FileNotFoundError:
        return None


def correlations(name, x, y):
    rho = spearmanr(x, y)[0]
    perm = [spearmanr(x, p)[0] for p in itertools.permutations(y)]
    p_exact = np.mean([abs(r) >= abs(rho) - 1e-12 for r in perm])
    r, p = pearsonr(x, y)
    print(f"{name}: n={len(x)}  Pearson r={r:.2f} (p={p:.3f})  Spearman rho={rho:.2f} (exact p={p_exact:.3f})")


def main():
    leak = leakage()
    ys = {"Mauve": {d: mauve(d) for d in ORDER}, "AUC": {d: auc(d) for d in ORDER}}
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    for ax, name, color, marker in ((ax1, "Mauve", BLUE, "o"), (ax2, "AUC", RED, "s")):
        ds = [d for d in ORDER if ys[name][d] is not None]
        x = np.log10([leak[d] + 1 for d in ds])
        y = [ys[name][d] for d in ds]
        correlations(f"leakage vs {name}", x, y)
        ax.scatter(x, y, color=color, marker=marker, s=80, label=name, alpha=0.85, edgecolor="k", linewidth=0.7)
        for xi, yi, d in zip(x, y, ds):
            dx, dy, ha, va = PLACE.get((name, d), (6, 4, "left", "bottom"))
            ax.annotate(LABEL.get(d, DATASETS[d][1]), (xi, yi), xytext=(dx, dy), textcoords="offset points",
                        fontsize=11, ha=ha, va=va, color=color)
        m, b = np.polyfit(x, y, 1)
        xs = np.sort(x)
        ax.plot(xs, m * xs + b, "--", color=color, label=f"{name} Trend", linewidth=2)
        ax.set_ylabel(name, color=color, fontsize=14)
        ax.tick_params(axis="y", labelcolor=color, labelsize=12)
    ax1.set_xlabel(r"$\log_{10}$ (leakage percentage + 1)", fontsize=14)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    lo, hi = ax1.get_xlim()  # room for the labels left of the zero-leakage datasets
    ax1.set_xlim(lo - 0.12 * (hi - lo), hi)
    ax1.set_xticks([t for t in ax1.get_xticks() if 0 <= t <= hi])
    for ax in (ax1, ax2):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.04 * (hi - lo), hi)
    ax1.set_title("Leakage vs Mauve (blue) and AUC (red)", fontsize=15)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=11, frameon=True)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig2_leakage.{ext}"), dpi=300)
    print(f"wrote {FIGS}/fig2_leakage.{{pdf,png}}")


if __name__ == "__main__":
    main()
