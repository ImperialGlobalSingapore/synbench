"""Figure 3 (MIA): mean ROC (95% CI band) of the bigram reference attack, 2 x 4 panels.

Rows: AUG-PE (top), DP-Gen (bottom); columns: PsyTAR, N2C2'08, DMSAFN, AsyLax. Each panel shows one curve
per eps in {inf, 4, 2, 1, 0.5} with its mean AUC [CI], and dashed the (eps, delta)-DP upper bound on any
attack's ROC for eps in {4, 2, 1, 0.5} (same color as that eps):

    TPR <= min(e^eps * FPR + delta, 1 - e^-eps * (1 - FPR - delta)),   delta = 1e-5

For AUG-PE the eps labels stand for its noise levels (results/mia/README.md).

Reads results/mia/{DS}_{gen}_mia_results.pkl (src/mia/roc_plot.py; format in results/mia/README.md).

    python paper/fig3_mia_roc.py
"""
import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset  # noqa: E402

from common import FIGS, RESULTS  # noqa: E402

RESULTS = os.path.join(RESULTS, "mia")
EPS = ["inf", 4, 2, 1, 0.5]
EPS_KEY = {"inf": "e0", 4: "e4", 2: "e2", 1: "e1", 0.5: "e0.5"}
COLORS = ["r", "g", "b", "y", "purple"]
DELTA = 1e-5
DATASETS = [("PsyTar", "Psytar"), ("N2C2_2008", "N2C2'08"), ("DMSAFN", "DMSAFN"), ("AYSLAX", "AsyLax")]
ROWS = [("augpe", "aug-pe"), ("dp-transformers", "dp-gen")]
# zoom insets of the paper figure: (generator, dataset) -> (xlim, ylim, loc)
INSETS = {("augpe", "PsyTar"): ((0.1, 0.2), (0.2, 0.3), "right"),
          ("augpe", "N2C2_2008"): ((0.0, 0.1), (0.05, 0.18), "right"),
          ("augpe", "DMSAFN"): ((0.78, 0.92), (0.87, 1.0), "center right"),
          ("dp-transformers", "PsyTar"): ((0.85, 1.0), (0.9, 1.0), "right"),
          ("dp-transformers", "DMSAFN"): ((0.83, 0.91), (0.9, 0.95), "right")}


def dp_roc_bound(eps, delta=DELTA, n=100):
    """Highest TPR any test can reach at each FPR against an (eps, delta)-DP mechanism."""
    fpr = np.linspace(0, 1, n)[1:]  # from the 2nd grid point, as the paper's curves (their FPR = 0 entry was empty)
    tpr = np.minimum(np.exp(eps) * fpr + delta, 1 - np.exp(-eps) * (1 - fpr - delta))
    return fpr, np.clip(tpr, 0, 1)


def panel(ax, datalist, bounds, title, inset=None):
    for i, (mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc) in enumerate(datalist):
        ax.plot(mean_fpr, mean_tpr, color=COLORS[i], lw=2,
                label=f"$\\epsilon=$ {EPS[i]} (AUC = {mean_auc:.2f} [{lower_auc:.2f}, {upper_auc:.2f}])")
        ax.fill_between(mean_fpr, lower_tpr, upper_tpr, color=COLORS[i] if inset else None, alpha=0.2)
    for i, (fpr, tpr) in enumerate(bounds):
        ax.plot(fpr, tpr, color=COLORS[i + 1], linestyle="--", lw=2)
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mean ROC Curve for {title}")
    ax.legend(loc="lower right", fontsize=8 if inset else 7)
    ax.grid(True)
    if inset:  # zoom on the smallest eps (last curve) and its bound
        xlim, ylim, loc = inset
        axins = inset_axes(ax, width="30%", height="30%", loc=loc, borderpad=2)
        mean_fpr, mean_tpr, lower_tpr, upper_tpr = datalist[-1][:4]
        axins.plot(mean_fpr, mean_tpr, color=COLORS[len(datalist) - 1], lw=2)
        axins.fill_between(mean_fpr, lower_tpr, upper_tpr, color=COLORS[len(datalist) - 1], alpha=0.2)
        axins.plot(*bounds[-1], color=COLORS[len(datalist) - 1], linestyle="--", lw=2)
        axins.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        axins.set_xlim(xlim)
        axins.set_ylim(ylim)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.tick_params(bottom=False, left=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGS, "fig3_mia_roc.png"))
    ap.add_argument("--dpi", type=int, default=100, help="the paper figure was saved at matplotlib's default 100")
    args = ap.parse_args()

    bounds = [dp_roc_bound(e) for e in EPS[1:]]
    fig = plt.figure(figsize=(15, 7.5))
    for row, (gen, gen_label) in enumerate(ROWS):
        for col, (ds, ds_label) in enumerate(DATASETS):
            with open(os.path.join(RESULTS, f"{ds}_{gen}_mia_results.pkl"), "rb") as f:
                results = pickle.load(f)
            ax = fig.add_subplot(2, 4, row * 4 + col + 1)
            panel(ax, [results[EPS_KEY[e]][0] for e in EPS], bounds, f"{ds_label} ({gen_label})", INSETS.get((gen, ds)))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf")
    print(f"wrote {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
