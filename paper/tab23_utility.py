"""Tables 2 and 3: downstream micro-F1 of Llama-3.2-1B synthetic data (DP-Gen / AUG-PE / DP-Dfs / EPSVec x eps),
three classifiers per dataset, mean over 3 seeds with the 95% CI half-width in small type below.

Per dataset: a header with the Random / Majority baselines, then the three classifiers side by side with their
real-data ("original") score. Bold: the best method in each column (by the rounded mean). DP-Dfs settings
whose synthetic text was unusable stay "--". Biomedical datasets and finance + legal datasets are two table* floats in one file.
Reads results/utility/scores.csv and baselines.csv.

    python paper/tab23_utility.py [--interval ci95|sd|none]
"""
import argparse
import csv
import os
import statistics as st

from common import CLASSIFIERS, DASH, RESULTS, TABLES, Scores, write

DATASET_NAME = {"hallmarks_of_cancer": "HoC", "N2C2_2008": "N2C2'08", "PsyTar": "PsyTAR", "MIMIC": "Mimic",
                "DMSAFN": "DMSAFN", "Lc37Fnd": "Lc37Fnd", "ZSTFNS": "ZsTfns", "AYSLAX": "Asylax", "EURLEX": "Eurlex"}
EPS_HEADER = [r"$\epsilon=\infty$", r"$\epsilon=0.5$", r"$\epsilon=1$", r"$\epsilon=2$", r"$\epsilon=4$"]

SOURCE = "Llama-3.2-1B"
METHODS = [("DP-Gen", "dp-transformers"), ("AUG-PE", "aug-pe"), ("DP-Dfs", "dp-diffusion"), ("EPSVec", "EPSVec")]
EPS = ["nodp", "0.5", "1", "2", "4"]
# DP-Dfs cells left blank in the paper: the generated text was too poor to train on.
OMIT = {("PsyTar", "dp-diffusion"): {"0.5"}, ("DMSAFN", "dp-diffusion"): {"0.5", "1"},
        ("ZSTFNS", "dp-diffusion"): {"0.5"}, ("EURLEX", "dp-diffusion"): {"0.5"}}
# (label, datasets) of the two tables, in the paper's dataset order.
PARTS = [("tab:utility-llama-bio", ["hallmarks_of_cancer", "N2C2_2008", "PsyTar", "MIMIC"]),
         ("tab:utility-llama-finlegal", ["DMSAFN", "Lc37Fnd", "ZSTFNS", "AYSLAX", "EURLEX"])]
CI_ROW_SPACE = "-3pt"  # pulls each interval row up towards its mean row
# per cell: significantly above Majority? (paired bootstrap + BH, src/utility/majority_test.py)
MAJORITY_TEST = os.path.join(RESULTS, "utility", "majority_test.csv")


def above_majority(path=MAJORITY_TEST):
    """(dataset, classifier dir, cell) -> significantly above Majority; {} when the test was not run."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(r["dataset"], r["classifier"], r["cell"]): r["above_majority"] == "True" for r in csv.DictReader(f)}


def table(res, datasets, label, caption, above):
    ncol = 1 + 5 * 3
    lines = [r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{|l|" + "|".join(["ccccc"] * 3) + "|}", r"\hline"]
    for ds in datasets:
        clfs = CLASSIFIERS[ds]
        lines.append(rf"\multicolumn{{{ncol}}}{{|c|}}{{\textsc{{Dataset {DATASET_NAME[ds]}}} "
                     rf"(\textbf{{Random}}: {res.random(ds):.1f}, \textbf{{Majority}}: {res.majority(ds):.1f})}} \\ \hline")
        heads = [rf"\multicolumn{{5}}{{c|}}{{\cellcolor{{gray!20}}\textbf{{{name} "
                 rf"(original={res.original(ds, model)})}}}}" for model, name in clfs]
        lines.append(r"\cellcolor{gray!20} & " + " & ".join(heads) + r" \\")
        lines.append("Method & " + " & ".join([" & ".join(EPS_HEADER)] * 3) + r" \\")
        rows = []  # (name, cells, rounded means) of the methods used on this dataset
        for name, method in METHODS:
            if OMIT.get((ds, method), set()) >= set(EPS):  # method not used on this dataset: no row
                continue
            keys = [(model, e) for model, _ in clfs for e in EPS]
            omitted = [e in OMIT.get((ds, method), ()) for _, e in keys]
            scores = [[] if o else res.scores(ds, SOURCE, m, f"{method}-e{e}-S2R") for (m, e), o in zip(keys, omitted)]
            cells = [(DASH, "") if o else res.parts(v) for o, v in zip(omitted, scores)]
            flat = [above.get((ds, m.replace("/", "_"), f"{method}-e{e}-S2R"), True) for m, e in keys]
            rows.append((name, cells, [round(st.mean(v), 1) if v else None for v in scores], flat))
        # bold the best method mean in every column (classifier x eps); ties are all bold
        best = [max((r[2][j] for r in rows if r[2][j] is not None), default=None) for j in range(len(clfs) * len(EPS))]
        for name, cells, vals, flat in rows:
            means = [rf"\textbf{{{m}}}" if v is not None and v == best[j] else m
                     for j, ((m, _), v) in enumerate(zip(cells, vals))]
            # grey: not significantly above Majority
            means = [m if ok or v is None else rf"\textcolor{{gray}}{{{m}}}" for m, v, ok in zip(means, vals, flat)]
            halves = [rf"{{\scriptsize {h}}}" if h else "" for _, h in cells]
            if any(halves):
                lines.append(" & ".join([name] + means) + rf" \\[{CI_ROW_SPACE}]")
                lines.append(" & ".join([""] + halves) + r" \\")
            else:  # no intervals in this row (no results, or --interval none): skip the empty line
                lines.append(" & ".join([name] + means) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}", ""]
    return "\n".join(lines)


def tables(res):
    (bio_label, bio), (fl_label, finlegal) = PARTS
    interval = "" if res.interval == "none" else rf" (small type: $\pm$ the {res.interval_text()})"
    first = (r"Random/majority guess baselines, and downstream classification micro-F1 scores for three models "
             r"per dataset, trained on synthetic data from Llama-3.2-1B across different privacy budgets "
             rf"($\epsilon$): biomedical datasets. Each value is the mean over {len(res.seeds)} seeds{interval}. "
             r"``--'': no result (DP-Dfs settings whose synthetic data were unusable). Bold: the best method in each "
             r"column; grey: not significantly above Majority (paired bootstrap over test examples, one-sided, "
             r"Benjamini-Hochberg-corrected $p \geq 0.05$); "
             r"$^\ast$: fewer seeds finished.")
    second = rf"As Table~\ref{{{bio_label}}}, for the finance and legal datasets."
    above = above_majority()
    return table(res, bio, bio_label, first, above) + "\n" + table(res, finlegal, fl_label, second, above)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", choices=["ci95", "sd", "none"], default="ci95")
    ap.add_argument("--out", default=os.path.join(TABLES, "tab23_utility.tex"))
    args = ap.parse_args()
    write(tables(Scores(args.interval)), args.out)


if __name__ == "__main__":
    main()
