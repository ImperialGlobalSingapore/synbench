"""Table 4: fidelity of Llama-3.2-1B synthetic data: MAUVE / NER-entity 1-gram KL / length KL per dataset,
method and eps (columns eps = inf, 4, 2, 1, 0.5).

DP-Gen / AUG-PE / DP-Dfs and the "Original" line are the paper's published values (written out below); their
MAUVE is cross-checked against the paper's result files in results/fidelity/paper/ (manifest.csv maps every
table cell to its file) and differences > 0.01 are printed. EPSVec's MAUVE / NER / length KL come from
results/fidelity/epsvec/ (src/fidelity/run_metrics.py with configs/fidelity/). src/fidelity computes N over
overlapping 512-token windows; the DP-Gen / AUG-PE / DP-Dfs N values were computed on text truncated to 512
tokens, so EPSVec's N can read 0-0.4 higher next to them (L and the Original row are unaffected).

    python paper/tab4_fidelity.py
"""
import argparse
import glob
import json
import os

from common import RESULTS, TABLES

PAPER_DIR = os.path.join(RESULTS, "fidelity", "paper")
EPSVEC_DIR = os.path.join(RESULTS, "fidelity", "epsvec")
EPS = ["nodp", "4", "2", "1", "0.5"]  # paper column order
EPS_HEAD = [r"$\epsilon=\infty$", r"$\epsilon=4$", r"$\epsilon=2$", r"$\epsilon=1$", r"$\epsilon=0.5$"]
NONE = "--/--/--"

# dataset -> (label, name in results/fidelity/paper/, Original M/N/L, {method: [cells in EPS order]}); paper values.
PAPER = {
    "hallmarks_of_cancer": (r"\textsc{HoC}", "hallmarks_of_cancer", "0.99/2.19/0.004", {
        "DP-Gen": ["0.65/3.71/0.06", "0.18/5.26/0.40", "0.16/5.23/0.417", "0.14/5.18/0.44", "0.12/5.21/0.47"],
        "AUG-PE": ["0.01/3.80/1.49", "0.01/3.92/1.40", "0.01/4.06/1.35", "0.01/4.45/1.25", "0.01/4.72/1.25"],
        "DP-Dfs": ["0.01/4.70/0.06", "0.01/6.49/0.36", "0.01/6.62/0.37", "0.01/6.855/0.42", "0.01/6.972/1.04"]}),
    "N2C2_2008": (r"\textsc{N2C2'08}", "n2c2_2008", "0.99/0.90/0.17", {
        "DP-Gen": ["0.42/2.34/0.75", "0.02/9.07/1.37", "0.02/6.79/1.40", "0.02/7.12/1.42", "0.02/7.19/1.54"],
        "AUG-PE": ["0.02/5.75/3.72", "0.03/6.50/3.80", "0.02/6.66/3.87", "0.02/6.75/3.63", "0.02/6.98/3.42"],
        "DP-Dfs": ["0.01/4.97/3.09", "0.01/6.06/2.91", "0.01/6.31/2.91", "0.01/6.45/2.91", "0.01/6.719/4.01"]}),
    "PsyTar": (r"\textsc{PsyTAR}", "psytar", "0.99/1.08/0.007", {
        "DP-Gen": ["0.61/2.60/0.03", "0.42/3.17/0.03", "0.34/3.24/0.05", "0.37/3.63/0.03", "0.35/3.71/0.03"],
        "AUG-PE": ["0.02/4.38/3.21", "0.02/4.62/3.27", "0.02/4.87/3.23", "0.02/5.24/3.34", "0.02/5.49/3.38"],
        "DP-Dfs": ["0.06/2.62/0.12", "0.01/8.08/0.16", "0.01/8.28/0.06", "0.01/8.59/0.07", NONE]}),
    "MIMIC": (r"\textsc{Mimic}", "mimic", "0.99/0.24/0.003", {
        "DP-Gen": ["0.96/0.30/0.01", "0.01/3.35/2.69", "0.01/3.65/2.59", "0.01/3.80/2.49", "0.01/4.89/2.76"],
        "AUG-PE": ["0.01/5.44/3.83", "0.01/5.40/6.21", "0.01/5.41/2.41", "0.01/5.51/2.42", "0.01/5.79/6.20"],
        "DP-Dfs": ["0.01/4.46/5.79", "0.01/5.72/5.79", "0.01/5.66/5.38", "0.01/5.55/5.38", "0.01/5.73/5.38"]}),
    "DMSAFN": (r"\textsc{DMSAFN}", "Daniel-ML", "0.86/1.23/0.012", {
        "DP-Gen": ["0.21/2.40/0.06", "0.18/2.84/0.02", "0.16/2.74/0.04", "0.15/3.04/0.03", "0.12/3.04/0.06"],
        "AUG-PE": ["0.01/3.85/2.36", "0.01/4.41/2.35", "0.01/4.66/2.34", "0.01/5.17/2.24", "0.01/4.63/2.30"],
        "DP-Dfs": ["0.01/10.04/0.12", "0.01/9.53/0.43", "0.01/9.49/0.37", NONE, NONE]}),
    "Lc37Fnd": (r"\textsc{Lc37Fnd}", "luckycat37", "0.99/0.88/0.002", {
        "DP-Gen": ["0.90/1.17/0.03", "0.01/4.61/3.71", "0.01/4.77/4.04", "0.01/4.77/3.95", "0.01/4.64/4.88"],
        "AUG-PE": ["0.01/4.22/2.52", "0.01/4.45/2.67", "0.01/5.20/2.82", "0.01/5.14/2.91", "0.01/5.45/2.87"],
        "DP-Dfs": ["0.01/7.00/1.37", "0.01/6.87/1.86", "0.01/7.10/1.87", "0.01/7.39/1.72", "0.01/7.39/1.65"]}),
    "ZSTFNS": (r"\textsc{ZsTfns}", "tfns", "0.99/1.29/0.002", {
        "DP-Gen": ["0.96/1.75/0.01", "0.04/4.89/0.09", "0.04/5.04/0.15", "0.05/5.06/0.19", "0.43/2.44/0.01"],
        "AUG-PE": ["0.02/3.55/4.36", "0.02/3.78/4.39", "0.02/3.97/4.38", "0.02/4.13/4.46", "0.01/4.26/4.39"],
        "DP-Dfs": ["0.01/7.38/0.87", "0.01/7.10/0.87", "0.01/7.22/0.77", "0.01/7.18/1.22", NONE]}),
    "AYSLAX": (r"\textsc{AsyLax}", "asylax", "0.98/0.11/0.004", {
        "DP-Gen": ["0.03/0.61/0.43", "0.01/3.02/6.41", "0.01/3.24/5.95", "0.01/6.20/3.58", "0.01/6.54/1.23"],
        "AUG-PE": ["0.01/5.53/1.41", "0.01/4.95/1.41", "0.01/5.55/1.92", "0.01/5.63/6.12", "0.01/5.92/1.41"],
        "DP-Dfs": ["0.01/8.01/3.42", "0.01/8.03/3.22", "0.01/8.04/3.23", "0.01/7.93/4.43", "0.01/8.21/6.91"]}),
    "EURLEX": (r"\textsc{Eurlex}", "eurlex", "0.98/0.26/0.006", {
        "DP-Gen": ["0.05/0.52/0.06", "0.10/0.79/0.23", "0.07/0.83/0.27", "0.03/0.93/0.22", "0.01/1.42/0.45"],
        "AUG-PE": ["0.01/1.97/0.05", "0.01/2.04/0.06", "0.01/2.12/0.04", "0.01/2.36/0.08", "0.01/2.40/0.08"],
        "DP-Dfs": ["0.01/9.26/0.28", "0.01/8.20/0.47", "0.01/8.61/0.47", "0.01/8.37/0.39", NONE]}),
}
METHOD_DIR = {"DP-Gen": "dp-transformers", "AUG-PE": "aug-pe", "DP-Dfs": "dp-diffusion"}


def paper_file_mauve(name, method, eps):
    f = f"{PAPER_DIR}/mauve-{name}-train{eps}-{METHOD_DIR[method]}-vs-test-original.json"
    return json.load(open(f))["mauve_star"] if os.path.exists(f) else None


def epsvec_cell(ds, eps):
    """M/N/L of one EPSVec setting from this repo's runs; "--" for a metric not computed."""
    tag = f"{ds}-train{eps}-EPSVec-vs-test-original.json"
    m = f"{EPSVEC_DIR}/mauve-{tag}"
    n = glob.glob(f"{EPSVEC_DIR}/ngramFrequency_ngram-1_ner-*-{tag}")
    l = f"{EPSVEC_DIR}/text-length-distribution-10bins-{tag}"
    vals = [f"{json.load(open(m))['mauve_star']:.2f}" if os.path.exists(m) else "--",
            f"{json.load(open(n[0]))['1-gram']:.2f}" if n else "--",
            f"{json.load(open(l))['comparison']['kl_div']:.2f}" if os.path.exists(l) else "--"]
    return "/".join(vals)


def check(ds, name, rows):
    for method, cells in rows.items():
        for e, c in zip(EPS, cells):
            if c == NONE:
                continue
            v = paper_file_mauve(name, method, e)
            if v is None:
                print(f"  no result file: {ds} {method} eps {e}")
            elif abs(v - float(c.split("/")[0])) > 0.01:
                print(f"  MAUVE differs: {ds} {method} eps {e}: table {c.split('/')[0]} vs file {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(TABLES, "tab4_fidelity.tex"))
    args = ap.parse_args()

    # each eps is three columns (M / N / L) joined by spaced slashes, so the numbers stand apart and align.
    # Bold: per dataset, eps and metric, the best method (highest M, lowest N and L); ties are all bold,
    # unless all methods tie.
    sep = r"@{\hspace{3pt}/\hspace{3pt}}"
    group = f"r{sep}r{sep}l"
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
             r"\renewcommand{\arraystretch}{0.95}",
             r"\begin{tabular}{l|" + "|".join([group] * len(EPS)) + "}", r"\toprule",
             r"\textbf{Method} & " + " & ".join(rf"\multicolumn{{3}}{{c{'|' if i < len(EPS) - 1 else ''}}}{{{h}}}"
                                                 for i, h in enumerate(EPS_HEAD)) + r" \\",
             " & " + " & ".join([r"$\mathcal{M}\uparrow$ & $\mathcal{N}\downarrow$ & $\mathcal{L}\downarrow$"] * len(EPS)) + r" \\",
             r"\midrule"]
    for ds, (label, name, orig, rows) in PAPER.items():
        check(ds, name, rows)
        table_rows = dict(rows)
        table_rows["EPSVec"] = [epsvec_cell(ds, e) for e in EPS]
        grid = {m: [c.split("/") for c in cells] for m, cells in table_rows.items()}  # method -> eps -> [M, N, L]
        best = {}
        for j in range(len(EPS)):
            for k, pick in enumerate((max, min, min)):
                vals = [float(grid[m][j][k]) for m in grid if grid[m][j][k] not in ("--", "")]
                # no bold when every method has the same value (e.g. MAUVE 0.01 across the board)
                best[j, k] = pick(vals) if vals and len(set(vals)) > 1 else None
        fmt = lambda v, j, k: rf"\textbf{{{v}}}" if v not in ("--", "") and float(v) == best[j, k] else v
        lines.append(rf"\rowcolor{{gray!20}} {label} & \multicolumn{{{3 * len(EPS)}}}{{c}}{{(\textbf{{Original}}: {orig.replace('/', ' / ')})}} \\")
        for method, cells in grid.items():
            lines.append(" & ".join([method] + [fmt(v, j, k) for j, c in enumerate(cells) for k, v in enumerate(c)]) + r" \\")
        lines.append(r"\midrule" if ds != list(PAPER)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}",
              r"\caption{Fidelity of Llama-3.2-1B synthetic data to the real test set: MAUVE ($\mathcal{M}$), "
              r"KL divergence of NER-entity 1-gram distributions ($\mathcal{N}$) and of text-length distributions "
              r"($\mathcal{L}$). Bold: the best method per dataset, $\epsilon$ and metric (none when all methods tie). ``--'': not available "
              r"(DP-Dfs settings whose synthetic data were unusable).}",
              r"\label{tab:fidelity-llama}", r"\end{table*}", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
