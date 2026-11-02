"""Label-set I/O and micro-F1 for the downstream-utility results.

Gold and predicted labels are handled as sets, so single- and multi-label tasks share one code path;
micro-F1 pools true/false positives and false negatives over all test rows.
"""
from __future__ import annotations

import ast
import collections
import json
from typing import List, Sequence, Set, Tuple

import numpy as np


def per_sample_counts(
    y_true: Sequence[Set[str]], y_pred: Sequence[Set[str]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tp, fp, fn) int arrays, one entry per sample."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} pred")
    n = len(y_true)
    tp = np.empty(n, dtype=np.int64)
    fp = np.empty(n, dtype=np.int64)
    fn = np.empty(n, dtype=np.int64)
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        tp[i] = len(t & p)
        fp[i] = len(p - t)
        fn[i] = len(t - p)
    return tp, fp, fn


def micro_f1_from_counts(tp, fp, fn) -> np.ndarray:
    """Micro-F1 from (possibly vectorised) TP/FP/FN totals.

    Accepts scalars or arrays. 0/0 (no positives predicted or present) -> 0.0,
    matching sklearn's micro-F1 convention.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    denom = 2.0 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return f1


def micro_f1(y_true: Sequence[Set[str]], y_pred: Sequence[Set[str]]) -> float:
    tp, fp, fn = per_sample_counts(y_true, y_pred)
    return float(micro_f1_from_counts(tp.sum(), fp.sum(), fn.sum()))


# --------------------------------------------------------------------------- #
# Label I/O
# --------------------------------------------------------------------------- #


def _as_label_set(value) -> Set[str]:
    """Normalise a stored label value to a set of strings.

    Handles list-valued (multi-label) and scalar (single-label) storage, and the
    string repr of a python list as written by classify.py's prediction file.
    """
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return {str(v) for v in parsed}
            except (ValueError, SyntaxError):
                pass
        return {s}
    # numeric / other scalar
    return {str(value)}


def load_gold(path: str, target_col: str = "label") -> List[Set[str]]:
    gold = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gold.append(_as_label_set(obj[target_col]))
    return gold


def load_predictions(path: str) -> List[Set[str]]:
    """Parse a classify.py ``test_results_cls.txt`` (index<TAB>prediction)."""
    preds = []
    with open(path) as f:
        header = f.readline()  # 'index\tprediction'
        if "prediction" not in header:
            # No header (unexpected) -- rewind and treat first line as data.
            f.seek(0)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            preds.append(_as_label_set(parts[1]))
    return preds


def majority_labels(train_gold: Sequence[Set[str]], k: int = 1) -> Set[str]:
    """The ``k`` most frequent labels by document frequency in the TRAINING split.

    The paper's "Majority" baseline predicts these ``k`` labels for *every* test
    sample (a constant multi-hot vector).

    Multi-label datasets use k=3 (N2C2'08: {Hypertension, Diabetes, CAD}, 53.2 micro-F1);
    single-label datasets use k=1, the ordinary majority class.
    """
    counter: collections.Counter = collections.Counter()
    for s in train_gold:
        for lab in s:
            counter[lab] += 1
    if not counter:
        raise ValueError("no labels found in training split")
    return {lab for lab, _ in counter.most_common(k)}
