"""Which utility cells are significantly above the Majority baseline? Writes results/utility/majority_test.csv.

For every (dataset, source, classifier, cell) of synthetic data, the statistic is the cell's table value minus
Majority: mean micro-F1 over the seeds minus the micro-F1 of the Majority predictor (top-1 train label; top-3 for
multi-label, as in baselines.csv). Paired bootstrap over test examples: each resample draws one set of test
indices and scores every seed and Majority on it. One-sided p = share of resamples with delta <= 0, then
Benjamini-Hochberg over all tested cells. A cell is above Majority when p_bh < alpha and delta > 0.
constant_pred marks cells whose classifiers output the same label set for every test example in all seeds.

    python src/utility/majority_test.py --models <run outputs>/models [--data data] [--sources Llama-3.2-1B]
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_scores import DATASETS  # noqa: E402
from label_metrics import load_gold, load_predictions, majority_labels, micro_f1_from_counts, per_sample_counts  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
# the synthetic-data cells of Tables 2-3
CELL = re.compile(r"^(dp-transformers|aug-pe|dp-diffusion|EPSVec)-e(nodp|0|0\.5|1|2|4)-S2R$")


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="run output root holding <dataset>/<source>/<classifier>/...")
    ap.add_argument("--data", default=os.path.join(REPO, "data"))
    ap.add_argument("--sources", nargs="+", default=["Llama-3.2-1B"])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "utility", "majority_test.csv"))
    args = ap.parse_args()

    rows = []
    for ds in DATASETS:
        gold = load_gold(os.path.join(args.data, ds, "original", "test-original.jsonl"))
        train = load_gold(os.path.join(args.data, ds, "original", "train-original.jsonl"))
        maj = majority_labels(train, k=3 if max(len(s) for s in train) > 1 else 1)
        mtp, mfp, mfn = per_sample_counts(gold, [maj] * len(gold))
        idx = np.random.default_rng(args.seed).integers(0, len(gold), size=(args.boot, len(gold)))
        maj_f1 = float(micro_f1_from_counts(mtp.sum(), mfp.sum(), mfn.sum()))
        maj_boot = micro_f1_from_counts(mtp[idx].sum(1), mfp[idx].sum(1), mfn[idx].sum(1))

        cells = collections.defaultdict(list)
        for source in args.sources:
            for f in sorted(glob.glob(os.path.join(args.models, ds, source, "*", "*", "seed*", "test_results_cls.txt"))):
                parts = os.path.relpath(os.path.dirname(f), args.models).split(os.sep)  # ds, source, clf, cell, seedN
                if not CELL.match(parts[3]):
                    continue
                cells[tuple(parts[1:4])].append(f)
        for (source, clf, cell), files in sorted(cells.items()):
            f1, boot, preds = [], [], set()
            for f in files:
                pred = load_predictions(f)
                tp, fp, fn = per_sample_counts(gold, pred)
                f1.append(float(micro_f1_from_counts(tp.sum(), fp.sum(), fn.sum())))
                boot.append(micro_f1_from_counts(tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1)))
                preds |= {frozenset(p) for p in pred}
            deltas = np.mean(boot, axis=0) - maj_boot
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            rows.append({"dataset": ds, "source": source, "classifier": clf, "cell": cell, "n_seeds": len(files),
                         "micro_f1": round(100 * np.mean(f1), 2), "majority": round(100 * maj_f1, 2),
                         "delta": round(100 * (np.mean(f1) - maj_f1), 2), "ci_low": round(100 * lo, 2),
                         "ci_high": round(100 * hi, 2), "p_raw": float(np.mean(deltas <= 0)),
                         "constant_pred": len(preds) == 1})
        print(f"{ds}: {sum(r['dataset'] == ds for r in rows)} cells", flush=True)

    for r, p in zip(rows, bh([r["p_raw"] for r in rows])):
        r["p_bh"] = round(float(p), 6)
        r["above_majority"] = bool(p < args.alpha and r["delta"] > 0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_above = sum(r["above_majority"] for r in rows)
    print(f"wrote {args.out}: {n_above}/{len(rows)} cells above Majority (BH, alpha {args.alpha}); "
          f"{sum(r['constant_pred'] for r in rows)} constant predictors")


if __name__ == "__main__":
    main()
