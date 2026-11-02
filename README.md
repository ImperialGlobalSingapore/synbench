# SynBench

Code and results for the SynBench paper: a benchmark of differentially private synthetic text generation
(DP-Gen, AUG-PE, DP-Dfs, EPSVec) on nine biomedical, financial and legal classification datasets, measuring
utility, fidelity, membership-inference risk and pre-training leakage.

## Reproduce the paper's figures and tables

Everything the paper reports is regenerated from the small result files in `results/` — no data or GPU needed:

```bash
conda env create -f environment.yml && conda activate synbench
make paper            # writes paper/figs/*.{pdf,png} and paper/tables/*.tex
```

| Paper | Script | Reads |
|---|---|---|
| Figure 2 — leakage vs MAUVE and MIA AUC | `paper/fig2_leakage.py` | `results/leakage/redpajama_summary.csv`, `results/fidelity/`, `results/mia/` |
| Figure 3 — MIA ROC curves (2 × 4) | `paper/fig3_mia_roc.py` | `results/mia/` |
| Figure 4 — generator size vs MAUVE; LoRA vs full fine-tuning | `paper/fig4_size_lora.py` | `results/fidelity/`, `results/utility/` |
| Tables 2–3 — utility of Llama-3.2-1B synthetic data | `paper/tab23_utility.py` | `results/utility/` |
| Table 4 — fidelity (MAUVE / NER KL / length KL) | `paper/tab4_fidelity.py` | `results/fidelity/` |
| Table 5 — utility by generator size | `paper/tab5_size.py` | `results/utility/` |

The LaTeX tables need `booktabs` and `\usepackage[table]{xcolor}`.

## Results (`results/`)

| Path | Content |
|---|---|
| `utility/scores.csv` | micro-F1 (×100) of every classifier run: dataset, source (generator), classifier, cell (method-ε), seed |
| `utility/baselines.csv` | Random and Majority micro-F1 per dataset |
| `utility/majority_test.csv` | per synthetic-data cell: significantly above Majority? (paired bootstrap over test examples, BH-corrected; greys cells in Tables 2–3) |
| `fidelity/paper/` | the paper's MAUVE / NER-KL / length-KL result files (`manifest.csv` maps table cells to files) |
| `fidelity/epsvec/` | the same metrics for EPSVec |
| `fidelity/instruct_mauve.csv` | MAUVE of DP-Gen on Llama-3.2-1B/3B-Instruct and Llama-3.1-8B-Instruct |
| `mia/` | MIA ROC summaries, labels, scores and scored targets per trial (format: `mia/README.md`) |
| `leakage/redpajama/` | per-document Infini-gram counts in RedPajama: one query per short text (≤ 1000 chars), one per 1000-char chunk of a long text |
| `leakage/redpajama_summary.csv` | per-dataset leakage: share of queried chunks with a verbatim match in RedPajama (`src/leakage/summarize_leakage.py`) |

No result file contains dataset text, except the normalized queries of the public HoC, DMSAFN and ZSTFNS
sentences in `leakage/redpajama/`.

## Data

All released data is on Hugging Face: [Ydsun/SynBench](https://huggingface.co/datasets/Ydsun/SynBench)

```python
from datasets import load_dataset
ds = load_dataset("Ydsun/SynBench", "DMSAFN")   # configs: hallmarks_of_cancer, DMSAFN, Lc37Fnd, ZSTFNS, AYSLAX, EURLEX
```

N2C2_2008, MIMIC and PsyTar are not redistributed (data use agreements); see `data/README.md` for how to
obtain them and the expected layout under `data/`.

## Rerunning the experiments (`src/`)

These steps need the data (see [Data](#data)) and, for training and fidelity, GPUs. Run from the repo root.

| Step | Command |
|---|---|
| Utility: write classifier configs, train, collect scores | `python src/utility/make_utility_configs.py`; `python src/utility/run_downstream.py --gpu 0 --datasets PsyTar --sources original Llama-3.2-1B EPSVec`; `python src/utility/collect_scores.py --models outputs/models`; `python src/utility/majority_test.py --models outputs/models` |
| Fidelity (MAUVE, NER-entity KL, length KL) | `PYTHONPATH=. python src/fidelity/run_metrics.py <synthetic.jsonl> data/<ds>/original/test-original.jsonl --metrics_cfg configs/fidelity/sim-metrics-<ds>.jsonnet` |
| MIA: sample target/reference sets; attack | `python src/mia/samplingdatasets.py <ds>`; `python src/mia/roc_plot.py <ds> <generator>` |
| Leakage: Infini-gram queries; summary | `python src/leakage/contamination_window.py ...` (long texts: `--chunks`); `python src/leakage/query_normalized.py --mode detok\|strip-url ...`; `python src/leakage/summarize_leakage.py` |

Notes:
- Classifier hyperparameters are in `src/utility/make_utility_configs.py` (per-dataset sequence length, batch
  size, and two learning-rate overrides: N2C2 Longformer-large 1e-5, PsyTAR DeBERTa 3e-6; 3e-5 otherwise).
  `run_downstream.py --sources` takes `original`, `Llama-3.2-1B`, `OLMo-2-0425-1B-10epochs`, `EPSVec` and the
  Instruct generators (`Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct`).
- The MIA n-gram model tokenizes with `str.split()`.
- The Infini-gram API is rate-limited per IP: one process, ≥ 0.5 s between queries. Querying all datasets
  takes days; the per-document results are in `results/leakage/`.
- Synthetic data generation (DP-Gen, AUG-PE, DP-Dfs, EPSVec) uses the methods' own code; the generated sets
  are provided as data.

## Citation

```bibtex
@inproceedings{sun2026synbench,
  title     = {SynBench: A Benchmark for Differentially Private Text Generation},
  author    = {Sun, Yidan and Schlegel, Viktor and Nandakumar, Srinivasan and Zahid, Iqra and Wu, Yuping and
               Wu, Yulong and Li, Hao and Zhang, Jie and Del-Pinto, Warren and Nenadic, Goran and
               Lam, Siew Kei and Bharath, Anil Anthony},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026}
}
```

## License

Copyright 2026 Imperial Global Singapore. The code is released under the Apache License 2.0 (`LICENSE`).
`src/utility/classify.py` is adapted from the Hugging Face Transformers examples (Apache 2.0). The released
data keeps the license of each source dataset (see the Hugging Face dataset card).
