"""Table 5: downstream micro-F1 of DP-Gen synthetic data by generator size (rows 1B / 3B / 8B per dataset,
columns eps = inf, 4, 2, 1, 0.5). Each cell is a mean with its 95% CI half-width (t-distribution, over seeds) as a
subscript (\pmci); bold: the best generator size per column.

  --stat avg   mean over the dataset's three classifiers; CI of the per-seed classifier average (main text)
  --stat best  the classifier with the highest seed mean; CI of that classifier (appendix, *_best.tex)
3B = Llama-3.2-3B-Instruct, 8B = Llama-3.1-8B-Instruct; 1B = Llama-3.2-1B-Instruct (paper) or, with --one-b base,
Llama-3.2-1B (full fine-tuning). Reads results/utility/scores.csv.

    python paper/tab5_size.py [--stat avg|best] [--one-b instruct|base]
"""
import argparse
import math
import os
import statistics as st

from scipy import stats

from common import CLASSIFIERS, DASH, TABLES, Scores, write

DATASETS = [("PsyTar", r"\textsc{PsyTAR}"), ("N2C2_2008", r"\textsc{N2C2'08}"), ("DMSAFN", r"\textsc{DMSAFN}"),
            ("AYSLAX", r"\textsc{AsyLax}")]
EPS = ["nodp", "4", "2", "1", "0.5"]
EPS_HEAD = [r"$\varepsilon=\infty$", r"$\varepsilon=4$", r"$\varepsilon=2$", r"$\varepsilon=1$", r"$\varepsilon=0.5$"]
OUT_DEFAULT = os.path.join(TABLES, "tab5_size.tex")


def half(vals):
    if len(vals) < 2:
        return None
    return stats.t.ppf(0.975, len(vals) - 1) * st.stdev(vals) / math.sqrt(len(vals))


def cell(res, ds, source, eps, stat):
    """(mean, CI half-width, star) of one dataset, generator and eps; mean rounded to 0.1, (None, None, "") if missing.
    stat "avg": the per-seed average over the three classifiers; "best": the classifier with the highest seed mean."""
    per_clf = [res.scores(ds, source, m, f"dp-transformers-e{eps}-S2R") for m, _ in CLASSIFIERS[ds]]
    if not all(per_clf):
        return None, None, ""
    n = min(map(len, per_clf))
    vals = [st.mean(v[s] for v in per_clf) for s in range(n)] if stat == "avg" else max(per_clf, key=st.mean)
    star = r"$^\ast$" if n < len(res.seeds) else ""
    return round(st.mean(vals), 1), half(vals), star


def fmt(mean, ci, star, top):
    """Mean with its CI half-width as a subscript; bold when it is the column's best (ties all bold)."""
    if mean is None:
        return DASH
    m = rf"\textbf{{{mean:.1f}}}" if mean == top else f"{mean:.1f}"
    return m + (rf"\pmci{{{ci:.1f}}}" if ci is not None else "") + star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-b", choices=["base", "instruct"], default="instruct",
                    help="1B row: Llama-3.2-1B-Instruct (paper) or Llama-3.2-1B (full fine-tuning)")
    ap.add_argument("--stat", choices=["avg", "best"], default="avg",
                    help="avg: mean over the three classifiers (main text); best: the best classifier (appendix)")
    ap.add_argument("--out", default=None, help=f"default: {OUT_DEFAULT} (+ _best for --stat best)")
    args = ap.parse_args()
    out = args.out or OUT_DEFAULT.replace(".tex", "_best.tex" if args.stat == "best" else ".tex")
    res = Scores("ci95")
    sizes = [("1B", "Llama-3.2-1B" if args.one_b == "base" else "Llama-3.2-1B-Instruct"),
             ("3B", "Llama-3.2-3B-Instruct"), ("8B", "Llama-3.1-8B-Instruct")]

    lines = [r"% needs \usepackage{multirow} and \newcommand{\pmci}[1]{\textsubscript{\scriptsize$\pm$#1}}",
             r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r" & Size & " + " & ".join(EPS_HEAD) + r" \\", r"\midrule"]
    for k, (ds, label) in enumerate(DATASETS):
        grid = {size: [cell(res, ds, source, e, args.stat) for e in EPS] for size, source in sizes}
        # bold the best generator size per column
        top = [max((grid[z][j][0] for z, _ in sizes if grid[z][j][0] is not None), default=None) for j in range(len(EPS))]
        lines.append(rf"\multirow{{{len(sizes)}}}{{*}}{{{label}}}")
        for size, _ in sizes:
            lines.append(" & ".join(["", size] + [fmt(*c, top[j]) for j, c in enumerate(grid[size])]) + r" \\")
        lines.append(r"\midrule" if k < len(DATASETS) - 1 else r"\bottomrule")
    what = "mean over classifiers" if args.stat == "avg" else "best classifier"
    note = (r" Best-classifier results are in Table~\ref{tab:text-llama-utility-best}." if args.stat == "avg" else "")
    lines += [r"\end{tabular}",
              r"\caption{Micro-F1 (" + what + r") of synthetic data from DP-fine-tuned Llama models of increasing "
              r"size. Subscripts: 95\% confidence interval over " + str(len(res.seeds)) + r" seeds." + note + "}",
              r"\label{tab:text-llama-utility" + ("" if args.stat == "avg" else "-best") + "}", r"\end{table}", ""]
    write("\n".join(lines), out)


if __name__ == "__main__":
    main()
