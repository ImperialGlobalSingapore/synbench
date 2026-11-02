# MIA results — file format (`*_mia_results.pkl`)

One file per (dataset, generator): `{dataset}_{generator}_mia_results.pkl`, e.g.
`PsyTar_dp-transformers_mia_results.pkl`. Datasets: PsyTar, N2C2_2008, DMSAFN, AYSLAX; generators:
`augpe` (AUG-PE), `dp-transformers` (DP-Gen). Attack: bigram (n = 2) reference attack,
score(x) = log P_release(x) − mean_m log P_ref_m(x), whitespace `split()` tokenization
(`src/mia/attack.py`, `src/mia/ngram.py`).

```python
import pickle
results = pickle.load(open("PsyTar_dp-transformers_mia_results.pkl", "rb"))   # protocol 5
roc_summary, (labels, scores, targets) = results["e1"]
```

## Top level: `results[eps]`

| key | meaning |
|---|---|
| `"e0"` | ε = ∞ (no DP) |
| `"e4"`, `"e2"`, `"e1"`, `"e0.5"` | ε = 4, 2, 1, 0.5 |

For `augpe` the keys stand for AUG-PE noise levels (dataset-dependent):

| key | PsyTar / DMSAFN | N2C2_2008 | AYSLAX |
|---|---|---|---|
| `e0` | 0 | 0 | 0 |
| `e4` | 2.96 | 2.61 | 2.82 |
| `e2` | 5.34 | 4.59 | 5.03 |
| `e1` | 9.77 | 8.19 | 9.12 |
| `e0.5` | 18.01 | 14.68 | 16.64 |

## `results[eps] = [roc_summary, [labels, scores, targets]]`

`roc_summary` (list of 7), from `src/mia/roc_plot.py::cal_mean_roc_with_ci`:

| index | name | type | meaning |
|---|---|---|---|
| 0 | `mean_fpr` | ndarray (100,) | FPR grid, 0 → 1 |
| 1 | `mean_tpr` | ndarray (100,) | TPR averaged over the trials |
| 2, 3 | `lower_tpr`, `upper_tpr` | ndarray (100,) | `mean_tpr ∓ 1.96 · sd / √R` (clipped to [0, 1]) |
| 4 | `mean_auc` | float | AUC averaged over the trials |
| 5, 6 | `lower_auc`, `upper_auc` | float | `mean_auc ∓ 1.96 · sd / √R` |

The interval is a 95% CI of the **mean** over the R = 50 trials. It reflects only which non-member
candidate a trial drew, not the sampling of releases, and is much narrower than the spread of single-trial
AUCs (e.g. DMSAFN DP-Gen ε = ∞: CI 0.812–0.823, 2.5–97.5th percentile of trial AUCs 0.789–0.851).

`labels`, `scores`, `targets`: each a list of R = 50 trials × D releases, aligned position by position
(position j of every trial is the same release):

| object | meaning |
|---|---|
| `labels[t][j]` | 1 = member release (a target outlier was inserted into its private set), 0 = non-member |
| `scores[t][j]` | attack score of the scored text; higher = more member-like; comparable within one ε only |
| `targets[t][j]` | `index` of the scored outlier (as in `sampling_labels.json` / `vulsamples/all_outliers.jsonl`): `special_index` for a member release; for a non-member release, the candidate drawn from its `not_added_samples` |

D = 100 (releases 0–99 in order), except where a release's synthetic file is missing, which is then
skipped: AYSLAX `augpe` `e2` and `e0.5` lack release 1 (D = 99; positions 1.. are releases 2..99).
Labels, member scores and member targets are the same in every trial; only non-member draws vary.

## Notes

- `roc_summary`, `labels` and `scores` are the paper's attack run. For a non-member release, `targets` is
  the candidate whose attack score equals the stored score; in PsyTar ~3% of non-member trials have two
  candidates with the same score (a duplicated text, or short texts whose bigrams are all unseen), and
  `targets` then holds the first in pool order — either gives the same score.
- Running the attack: `python src/mia/roc_plot.py <dataset> <generator>` (from the repo root, data under `data/`)
  writes this format to `outputs/mia/` directly; trial t at ε draws with `random.Random(f"{seed}-{eps}-{t}")`
  (seed 0), so a rerun repeats the draws.
- Figure 3: `python paper/fig3_mia_roc.py`.
