"""Figure 4: generator size vs MAUVE (left) and LoRA vs full fine-tuning utility (right), DP-Gen.

Left   MAUVE (mauve_star, synthetic train vs real test) averaged over eps in {inf, 0.5, 1, 2, 4}, per generator:
       Llama-3.2-1B (full fine-tuning; results/fidelity/paper), Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct
       (results/fidelity/instruct_mauve.csv); N2C2'08, PsyTAR, DMSAFN. Single runs, no interval.
Right  Delta micro-F1 = LoRA (DP-Gen on Llama-3.2-1B-Instruct) - full fine-tuning (DP-Gen on Llama-3.2-1B) per
       dataset and DP eps, from the 3-seed runs (results/utility/scores.csv), datasets ordered by RedPajama leakage.
       --agg max: each arm takes its best classifier by seed mean; error bar = 95% CI of the difference,
           sqrt(h_lora^2 + h_full^2), h = that classifier's CI half-width (t_{0.975,2} * sd / sqrt(3)).
       --agg mean (paper): per-seed average over the three classifiers; error bar = Welch 95% CI of the difference.

    python paper/fig4_size_lora.py [--agg max|mean]
"""
import argparse
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from common import ALL_EPS, CLASSIFIERS, DP_EPS, FIGS, Scores, ci_half, instruct_mauve, leakage, paper_mauve  # noqa: E402

LABEL = {"N2C2_2008": "N2C2'08", "PsyTar": "PsyTAR", "DMSAFN": "DMSAFN", "AYSLAX": "AsyLax"}
GENERATORS = [("Llama-3.2-1B", None), ("Llama-3.2-3B", "Llama-3.2-3B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct")]
# styled as the MIA figures: eps colors of the ROC figure (4 g, 2 b, 1 y, 0.5 purple), default grid
EPS_COLORS = {"0.5": "purple", "1": "y", "2": "b", "4": "g"}
DS_COLORS = ["r", "darkorange", "c"]  # datasets: colors not used for eps


def plot_mauve(ax, datasets):
    for i, ds in enumerate(datasets):
        y = []
        for _, approach in GENERATORS:
            vals = [paper_mauve(ds, "dp-transformers", e) if approach is None else instruct_mauve(ds, approach, e) for e in ALL_EPS]
            vals = [v for v in vals if v is not None]
            y.append(np.mean(vals) if len(vals) == len(ALL_EPS) else np.nan)
        print(f"MAUVE {ds:10s} " + "  ".join(f"{g}={v:.3f}" for (g, _), v in zip(GENERATORS, y)))
        ax.plot([1, 2, 3], y, marker="os^"[i], ms=9, lw=2, color=DS_COLORS[i], markeredgecolor="k", label=LABEL[ds])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([g for g, _ in GENERATORS])
    ax.set_xlabel("Generator size")
    ax.set_ylabel(r"MAUVE (mean over $\epsilon$)")
    ax.set_title("MAUVE by generator size")
    ax.legend(fontsize=11, markerscale=1.1)
    ax.grid(True)


def arm(res, ds, source, eps, agg):
    """(mean, half-width) of one arm; per-seed values for --agg mean."""
    cell = f"dp-transformers-e{eps}-S2R"
    per_clf = {m: res.scores(ds, source, m, cell) for m, _ in CLASSIFIERS[ds]}
    if agg == "max":
        m = max(per_clf, key=lambda k: st.mean(per_clf[k]) if per_clf[k] else -1)
        return st.mean(per_clf[m]), ci_half(per_clf[m]), per_clf[m], m
    n = min(len(v) for v in per_clf.values())
    seeds = [st.mean(v[s] for v in per_clf.values()) for s in range(n)]
    return st.mean(seeds), ci_half(seeds), seeds, "mean"


def welch_half(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if va + vb > 0 else 1
    return stats.t.ppf(0.975, dof) * math.sqrt(va + vb)


def plot_lora(ax, res, datasets, leak, agg):
    order = sorted(datasets, key=lambda d: leak.get(d, 0))
    width = 0.18
    x = np.arange(len(order))
    print(f"{'dataset':10s} {'eps':>4s} {'LoRA':>14s} {'full':>14s} {'delta':>14s}")
    for i, eps in enumerate(DP_EPS):
        ys, errs = [], []
        for ds in order:
            lo, hl, sl, ml = arm(res, ds, "Llama-3.2-1B-Instruct", eps, agg)
            fu, hf, sf, mf = arm(res, ds, "Llama-3.2-1B", eps, agg)
            d = lo - fu
            h = math.hypot(hl, hf) if agg == "max" else welch_half(sl, sf)
            ys.append(d)
            errs.append(h)
            print(f"{ds:10s} {eps:>4s} {lo:7.1f}±{hl:4.1f} {fu:7.1f}±{hf:4.1f} {d:+7.1f}±{h:4.1f}"
                  + (f"   ({ml.split('/')[-1]} vs {mf.split('/')[-1]})" if agg == "max" else ""))
        ax.bar(x + (i - 1.5) * width, ys, width=width, yerr=errs, capsize=2.5, color=EPS_COLORS[eps],
               error_kw=dict(elinewidth=0.9, ecolor="#333333"), label=f"$\\epsilon=$ {eps}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABEL[d]}\nLeak: {leak.get(d, 0):.2f}%" for d in order])
    ax.set_xlabel("Dataset (RedPajama leakage %)")
    what = "best classifier" if agg == "max" else "mean over classifiers"
    ax.set_ylabel(rf"$\Delta$ micro-F1 (LoRA $-$ full fine-tuning), {what}")
    ax.axhline(0, color="navy", ls="--", lw=1)
    ax.set_title("LoRA vs full fine-tuning (DP-Gen)")
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", choices=["max", "mean"], default="mean")
    ap.add_argument("--out", default=os.path.join(FIGS, "fig4_size_lora.png"))
    args = ap.parse_args()

    leak = leakage()

    res = Scores()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    plot_mauve(axs[0], ["N2C2_2008", "PsyTar", "DMSAFN"])
    plot_lora(axs[1], res, ["AYSLAX", "N2C2_2008", "DMSAFN", "PsyTar"], leak, args.agg)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf")
    print(f"wrote {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
