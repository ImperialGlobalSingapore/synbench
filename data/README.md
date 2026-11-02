# Data

The data is not in this repository. Download the public part from Hugging Face
([Ydsun/SynBench](https://huggingface.co/datasets/Ydsun/SynBench)) into `data/`; the restricted datasets must
be obtained from their providers.

## Layout

```
data/<dataset>/
├── original/                      train-original.jsonl, test-original.jsonl   ({"text": ..., "label": [...]})
├── Llama-3.2-1B/<method>/         train<eps>-<method>.jsonl      method: dp-transformers (DP-Gen), aug-pe, dp-diffusion (DP-Dfs), EPSVec
├── Llama-3.2-{1B,3B}-Instruct/, Llama-3.1-8B-Instruct/dp-transformers/   train<eps>-dp-transformers-<generator>.jsonl
├── OLMo-2-0425-1B-10epochs/dp-transformers/                              train<eps>-dp-transformers.jsonl
└── mia/                           vulsamples/ (outliers), D_sample/ (target and reference sets, sampling_labels.json),
                                   D_synth/<generator>/finalresults/ (synthetic release per target set and eps)
```

`eps` is one of `nodp` (ε = ∞; `0` for OLMo), `0.5`, `1`, `2`, `4`. For AUG-PE the ε labels stand for noise
levels (`results/mia/README.md`).

## Availability

| Dataset | Domain | Released here |
|---|---|---|
| HoC (hallmarks_of_cancer) | biomedical | yes |
| DMSAFN, Lc37Fnd, ZSTFNS | finance | yes |
| AYSLAX (AsyLax), EURLEX | legal | yes |
| N2C2_2008 | clinical | no — n2c2 data use agreement (DBMI Data Portal) |
| MIMIC | clinical | no — PhysioNet credentialed access |
| PsyTar | patient drug reviews | no |

For the restricted datasets neither the original nor the synthetic data (which is derived from it) is
redistributed. With access to the originals, place them under `data/<dataset>/original/` in the format above.
