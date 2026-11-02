"""Shared loaders for the paper's figures and tables; everything is read from results/."""
import csv
import json
import math
import os
import pickle
import statistics as st

from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
FIGS = os.path.join(REPO, "paper", "figs")
TABLES = os.path.join(REPO, "paper", "tables")
SEEDS = ["42", "1337", "2024"]
DASH = "--"

# dataset id -> (name in the paper's result files, display name)
DATASETS = {"hallmarks_of_cancer": ("hallmarks_of_cancer", "HoC"), "N2C2_2008": ("n2c2_2008", "N2C2'08"),
            "PsyTar": ("psytar", "PsyTAR"), "MIMIC": ("mimic", "Mimic"), "DMSAFN": ("Daniel-ML", "DMSAFN"),
            "Lc37Fnd": ("luckycat37", "Lc37Fnd"), "ZSTFNS": ("tfns", "ZsTfns"), "AYSLAX": ("asylax", "AsyLax"),
            "EURLEX": ("eurlex", "Eurlex")}
DP_EPS = ["0.5", "1", "2", "4"]
ALL_EPS = ["nodp", "0.5", "1", "2", "4"]
# classifiers per dataset (HF id, display name), in the paper's column order
CLASSIFIERS = {
    "hallmarks_of_cancer": [("bert-large-uncased", "BERT-large"), ("emilyalsentzer/Bio_ClinicalBERT", "BioClinicalBERT"),
                            ("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract", "BiomedBERT")],
    "N2C2_2008": [("allenai/longformer-large-4096", "LongFormer-large"), ("yikuan8/Clinical-BigBird", "Clinical-BigBird"),
                  ("yikuan8/Clinical-Longformer", "Clinical-LongFormer")],
    "PsyTar": [("bert-base-uncased", "BERT-base"), ("bert-large-uncased", "BERT-large"),
               ("microsoft/deberta-v2-xlarge", "DeBERTa-xlarge")],
    "MIMIC": [("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract", "BiomedBERT"),
              ("emilyalsentzer/Bio_ClinicalBERT", "BioClinicalBERT"), ("allenai/biomed_roberta_base", "BiomedRobertaBase")],
    "DMSAFN": [("ProsusAI/finbert", "ProsusAI-finbert"), ("yiyanghkust/finbert-tone", "finbert-tone"),
               ("ahmedrachid/FinancialBERT-Sentiment-Analysis", "FinancialBERT")],
    "Lc37Fnd": [("ProsusAI/finbert", "ProsusAI-finbert"),
                ("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "distilroberta"),
                ("ahmedrachid/FinancialBERT-Sentiment-Analysis", "FinancialBERT")],
    "ZSTFNS": [("ProsusAI/finbert", "ProsusAI-finbert"), ("yiyanghkust/finbert-tone", "finbert-tone"),
               ("ahmedrachid/FinancialBERT-Sentiment-Analysis", "FinancialBERT")],
    "AYSLAX": [("nlpaueb/legal-bert-base-uncased", "legal-bert-base"), ("law-ai/InLegalBERT", "InLegalBERT"),
               ("FacebookAI/roberta-base", "Roberta-base")],
    "EURLEX": [("nlpaueb/legal-bert-base-uncased", "legal-bert-base"), ("law-ai/InLegalBERT", "InLegalBERT"),
               ("FacebookAI/roberta-base", "Roberta-base")],
}


class Scores:
    """Per-run micro-F1 (x100) from results/utility/scores.csv, with the main-table interval."""

    def __init__(self, interval="ci95"):
        self.interval, self.seeds = interval, SEEDS  # interval: ci95 | sd | none
        self.runs = {}
        with open(os.path.join(RESULTS, "utility", "scores.csv")) as f:
            for r in csv.DictReader(f):
                self.runs[(r["dataset"], r["source"], r["classifier"], r["cell"], r["seed"])] = float(r["micro_f1"])
        with open(os.path.join(RESULTS, "utility", "baselines.csv")) as f:
            self.base = {r["dataset"]: (float(r["random"]), float(r["majority"])) for r in csv.DictReader(f)}

    def scores(self, ds, source, model, cell):
        """micro-F1 of every finished seed of one cell, in seed order."""
        key = (ds, source, model.replace("/", "_"), cell)
        return [self.runs[key + (s,)] for s in SEEDS if key + (s,) in self.runs]

    def majority(self, ds):
        return self.base[ds][1]

    def random(self, ds):
        return self.base[ds][0]

    def parts(self, vals):
        """(mean, interval) strings: mean over seeds and the 95% CI half-width (t-distribution)."""
        if not vals:
            return DASH, ""
        mean = f"{st.mean(vals):.1f}" + (r"$^\ast$" if len(vals) < len(SEEDS) else "")
        if len(vals) < 2 or self.interval == "none":
            return mean, ""
        return mean, rf"$\pm${(st.stdev(vals) if self.interval == 'sd' else ci_half(vals)):.1f}"

    def interval_text(self):
        return "95\\% confidence interval (t-distribution)" if self.interval == "ci95" else "standard deviation"

    def original(self, ds, model):
        m, h = self.parts(self.scores(ds, "original", model, "original-R2R"))
        return m + (rf"{{\scriptsize{h}}}" if h else "")


def ci_half(vals):
    """95% CI half-width of a mean over seeds (t-distribution)."""
    return stats.t.ppf(0.975, len(vals) - 1) * st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0


def paper_mauve(ds, method, eps):
    """mauve_star of a Llama-3.2-1B setting (paper runs; EPSVec from this release's runs), else None."""
    if method == "EPSVec":
        f = os.path.join(RESULTS, "fidelity", "epsvec", f"mauve-{ds}-train{eps}-EPSVec-vs-test-original.json")
    else:
        f = os.path.join(RESULTS, "fidelity", "paper", f"mauve-{DATASETS[ds][0]}-train{eps}-{method}-vs-test-original.json")
    return json.load(open(f))["mauve_star"] if os.path.exists(f) else None


def instruct_mauve(ds, generator, eps):
    with open(os.path.join(RESULTS, "fidelity", "instruct_mauve.csv")) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == DATASETS[ds][0] and r["generator"] == generator and r["epsilon"] == eps:
                return float(r["mauve_star"])
    return None


def mia_results(ds, generator):
    with open(os.path.join(RESULTS, "mia", f"{ds}_{generator}_mia_results.pkl"), "rb") as f:
        return pickle.load(f)


def leakage():
    with open(os.path.join(RESULTS, "leakage", "redpajama_summary.csv")) as f:
        return {r["dataset"]: float(r["leakage_pct"]) for r in csv.DictReader(f)}


def write(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")

