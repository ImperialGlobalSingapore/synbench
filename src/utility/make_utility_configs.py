"""Generate the downstream-utility classifier configs for every dataset x source x classifier.

Layout (mirrors data/ and outputs/models):
    configs/utility/generated/<DS>/<SOURCE>/<clf>/<cell>.jsonnet
      SOURCE = original | Llama-3.2-1B | OLMo-2-0425-1B-10epochs | Llama-3.2-{1B,3B}-Instruct | Llama-3.1-8B-Instruct
      cell   = original-R2R | <method>-e<eps>-S2R
    output_dir = ./outputs/models/<DS>/<SOURCE>/<clf>/<cell>
EPSVec (Llama-3.2-1B/EPSVec data) is emitted into the Llama-3.2-1B folder as EPSVec-e<eps>-S2R; it is
only run when the driver is given --sources EPSVec, and only where the data file exists.

Paths use the "./data" and "./outputs" prefixes; src/utility/run_downstream.py rewrites them to
DATA_ROOT / OUTPUT_ROOT. Per-dataset hyperparameters are in DATASETS below.

    python src/utility/make_utility_configs.py
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_ROOT = os.path.join(REPO, "configs", "utility", "generated")

LLAMA_METHODS = ["aug-pe", "dp-diffusion", "dp-transformers"]
LLAMA_EPS = ["0.5", "1", "2", "4", "nodp"]
OLMO_EPS = ["0", "0.5", "1", "2", "4"]  # e0 = non-private baseline
EPSVEC_EPS = ["0.5", "1", "2", "4", "nodp"]
# Instruct generators (DP-Gen only), generated for four datasets; run with --sources <generator>.
INSTRUCT_GENS = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct"]
INSTRUCT_DATASETS = {"AYSLAX", "N2C2_2008", "PsyTar", "DMSAFN"}

# Per dataset: classifiers (HF ids), max_seq_length, batch size, lr. Optional per-classifier overrides.
DATASETS = {
    "DMSAFN": dict(clfs=["ProsusAI/finbert", "ahmedrachid/FinancialBERT-Sentiment-Analysis",
                         "yiyanghkust/finbert-tone"], seq=160, bs=64),
    # seq 128: real texts are <= 77 tokens; few synthetic texts are truncated.
    "ZSTFNS": dict(clfs=["ProsusAI/finbert", "ahmedrachid/FinancialBERT-Sentiment-Analysis",
                         "yiyanghkust/finbert-tone"], seq=128, bs=64),
    "Lc37Fnd": dict(clfs=["ProsusAI/finbert", "ahmedrachid/FinancialBERT-Sentiment-Analysis",
                          "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"], seq=512, bs=64),
    "AYSLAX": dict(clfs=["FacebookAI/roberta-base", "law-ai/InLegalBERT",
                         "nlpaueb/legal-bert-base-uncased"], seq=512, bs=64),
    "EURLEX": dict(clfs=["FacebookAI/roberta-base", "law-ai/InLegalBERT",
                         "nlpaueb/legal-bert-base-uncased"], seq=512, bs=64),
    "MIMIC": dict(clfs=["allenai/biomed_roberta_base", "emilyalsentzer/Bio_ClinicalBERT",
                        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"], seq=512, bs=64),
    "hallmarks_of_cancer": dict(clfs=["bert-large-uncased", "emilyalsentzer/Bio_ClinicalBERT",
                                      "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"], seq=128, bs=64),
    # seq 4096 / bs 2 (truncates 2.4% of the real notes). BigBird runs without gradient checkpointing (fits
    # in memory). Longformer-large uses lr 1e-5 (it does not learn the real data at 3e-5).
    "N2C2_2008": dict(clfs=["allenai/longformer-large-4096", "yikuan8/Clinical-BigBird",
                            "yikuan8/Clinical-Longformer"], seq=4096, bs=2,
                      overrides={"yikuan8/Clinical-BigBird": {"gradient_checkpointing": "false"},
                                 "allenai/longformer-large-4096": {"learning_rate": "1e-5"}}),
    # DeBERTa-v2-xlarge uses lr 3e-6 (it collapses at 3e-5).
    "PsyTar": dict(clfs=["bert-base-uncased", "bert-large-uncased", "microsoft/deberta-v2-xlarge"], seq=128, bs=64,
                   overrides={"microsoft/deberta-v2-xlarge": {"learning_rate": "3e-6"}}),
}

TEMPLATE = """local run_name = "{run_name}";
{{
    model_name_or_path: "{model}",
    output_dir: "./outputs/models/"+run_name,
    overwrite_output_dir: true,
    do_train: true,
    do_eval: true,
    do_predict:true,
    train_file: "{train_file}",
    validation_file: "./data/{ds}/original/test-original.jsonl",
    test_file: "./data/{ds}/original/test-original.jsonl",
    overwrite_cache: true,
    max_seq_length: {seq},
    per_device_train_batch_size: {bs},
    learning_rate: {lr},
    num_train_epochs: 20,
    run_name: run_name,
    eval_strategy: "epoch",
    warmup_ratio: 0.1,
    target_col: 'label',
    sentence_col: 'text',
    load_best_model_at_end: "true",
    metric_for_best_model: 'f1',
    save_total_limit: 1,
    save_strategy: 'best',
    gradient_checkpointing: '{gc}',
    bf16: 'true',
    tf32: 'true',
}}
"""


def cells(ds):
    """(source, cell, train_file) for every run of a dataset."""
    d = f"./data/{ds}"
    yield "original", "original-R2R", f"{d}/original/train-original.jsonl"
    for m in LLAMA_METHODS:
        for e in LLAMA_EPS:
            yield "Llama-3.2-1B", f"{m}-e{e}-S2R", f"{d}/Llama-3.2-1B/{m}/train{e}-{m}.jsonl"
    for e in OLMO_EPS:
        yield ("OLMo-2-0425-1B-10epochs", f"dp-transformers-e{e}-S2R",
               f"{d}/OLMo-2-0425-1B-10epochs/dp-transformers/train{e}-dp-transformers.jsonl")
    for e in EPSVEC_EPS:
        yield "Llama-3.2-1B", f"EPSVec-e{e}-S2R", f"{d}/Llama-3.2-1B/EPSVec/train{e}-EPSVec.jsonl"
    if ds in INSTRUCT_DATASETS:
        for g in INSTRUCT_GENS:
            for e in LLAMA_EPS:
                yield g, f"dp-transformers-e{e}-S2R", f"{d}/{g}/dp-transformers/train{e}-dp-transformers-{g}.jsonl"


def render(ds, spec, model, cell, source, train_file):
    clf = model.replace("/", "_")
    opts = {"gradient_checkpointing": "true", "learning_rate": "3e-5", **spec.get("overrides", {}).get(model, {})}
    return TEMPLATE.format(run_name=f"{ds}/{source}/{clf}/{cell}", model=model, train_file=train_file,
                           ds=ds, seq=spec["seq"], bs=spec["bs"], lr=opts["learning_rate"],
                           gc=opts["gradient_checkpointing"])


def main():
    n = 0
    for ds, spec in DATASETS.items():
        for model in spec["clfs"]:
            clf = model.replace("/", "_")
            for source, cell, train_file in cells(ds):
                out = os.path.join(EXP_ROOT, ds, source, clf, f"{cell}.jsonnet")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    f.write(render(ds, spec, model, cell, source, train_file))
                n += 1
    print(f"wrote {n} configs under {EXP_ROOT}")


if __name__ == "__main__":
    main()
