"""Collect the micro-F1 of every finished classifier run into results/utility/scores.csv.

Each run (configs from make_utility_configs.py, trained by run_downstream.py / classify.py) leaves
    <models>/<dataset>/<source>/<classifier>/<cell>/seed<seed>/{test_results_cls.txt, experiment-args.json}
This scores test_results_cls.txt against data/<dataset>/original/test-original.jsonl. Also writes results/utility/baselines.csv with each dataset's
Random (copied from the paper) and Majority (top-1 train label; top-3 for multi-label) micro-F1.

    python src/utility/collect_scores.py --models <run outputs>/models [--data data] [--out results/utility]
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from label_metrics import load_gold, load_predictions, majority_labels, micro_f1  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
# generators reported in the paper (Llama-3.2-1B also holds the EPSVec cells)
SOURCES = {"original", "Llama-3.2-1B", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct",
           "OLMo-2-0425-1B-10epochs"}
DATASETS = ["hallmarks_of_cancer", "N2C2_2008", "PsyTar", "MIMIC", "DMSAFN", "Lc37Fnd", "ZSTFNS", "AYSLAX", "EURLEX"]
# "Random" is the paper's value (expected micro-F1 of uniform random labels); "Majority" is recomputed.
RANDOM = {"hallmarks_of_cancer": 3.7, "N2C2_2008": 46.1, "PsyTar": 25.4, "MIMIC": 48.8, "DMSAFN": 30.5,
          "Lc37Fnd": 41.3, "ZSTFNS": 48.7, "AYSLAX": 38.4, "EURLEX": 24.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="run output root holding <dataset>/<source>/<classifier>/...")
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "utility"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, base = [], []
    for ds in DATASETS:
        gold = load_gold(os.path.join(args.data, ds, "original", "test-original.jsonl"))
        train = load_gold(os.path.join(args.data, ds, "original", "train-original.jsonl"))
        maj = majority_labels(train, k=3 if max(len(s) for s in train) > 1 else 1)
        base.append({"dataset": ds, "random": RANDOM[ds], "majority": round(100 * micro_f1(gold, [maj] * len(gold)), 4)})
        for pred_file in sorted(glob.glob(os.path.join(args.models, ds, "*", "*", "*", "seed*", "test_results_cls.txt"))):
            run = os.path.dirname(pred_file)
            parts = os.path.relpath(run, args.models).split(os.sep)  # ds, source, clf, cell, seedN
            if parts[1] not in SOURCES or not os.path.isfile(os.path.join(run, "experiment-args.json")):
                continue
            preds = load_predictions(pred_file)
            if len(preds) != len(gold):
                raise ValueError(f"{pred_file}: {len(preds)} predictions for {len(gold)} test rows")
            rows.append({"dataset": ds, "source": parts[1], "classifier": parts[2], "cell": parts[3],
                         "seed": parts[4][len("seed"):],
                         "micro_f1": round(100 * micro_f1(gold, preds), 4)})
        print(f"{ds}: {sum(r['dataset'] == ds for r in rows)} runs")

    for name, data in [("scores.csv", rows), ("baselines.csv", base)]:
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {os.path.join(args.out, name)} ({len(data)} rows)")


if __name__ == "__main__":
    main()
