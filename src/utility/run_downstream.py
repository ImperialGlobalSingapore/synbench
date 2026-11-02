"""Resumable multi-seed driver for the downstream-utility classifiers (one GPU lane per process).

Runs the configs written by src/utility/make_utility_configs.py in a fixed order
(dataset -> source -> classifier -> cell -> seed). Per run it:
  * relocates "./data" / "./outputs" to DATA_ROOT / OUTPUT_ROOT and adds the seed;
  * skips runs already finished with the same hyperparameters (marker: experiment-args.json);
    a finished run with different hyperparameters is deleted and rerun;
  * after training, sanity-checks the result (eval_f1 == 0 or a constant predictor is flagged)
    and deletes model weights / optimizer state, keeping metrics and predictions.
Every run appends a row to <OUTPUT_ROOT>/logs/utility/runs.tsv.

    python src/utility/run_downstream.py --gpu 3 --datasets hallmarks_of_cancer
    python src/utility/run_downstream.py --gpu 0 --datasets PsyTar --sources EPSVec Llama-3.2-3B-Instruct
    python src/utility/run_downstream.py --gpu 0 --datasets N2C2_2008 --clfs longformer-large --dry-run
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import _jsonnet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_utility_configs import DATASETS, cells  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_ROOT = os.path.join(REPO, "configs", "utility", "generated")  # written by make_utility_configs.py
SCRIPT = os.path.join(REPO, "src", "utility", "classify.py")
DEFAULT_SOURCES = ["original", "Llama-3.2-1B", "OLMo-2-0425-1B-10epochs"]
# Hyperparameters that must match for a finished run to count as done.
MATCH_KEYS = ["learning_rate", "per_device_train_batch_size", "max_seq_length", "num_train_epochs"]
WEIGHT_GLOBS = ["*.safetensors", "pytorch_model*.bin", "optimizer.pt", "scheduler.pt", "rng_state*.pth"]


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def flat_args(path):
    a = {}
    for v in json.load(open(path)).values():
        if isinstance(v, dict):
            a.update(v)
    return a


def plan(datasets, sources, clf_filter):
    """Yield (ds, source, clf, cell, config_path) in run order."""
    for ds in datasets:
        for source in sources:
            for model in DATASETS[ds]["clfs"]:
                clf = model.replace("/", "_")
                if clf_filter and not any(s in clf for s in clf_filter):
                    continue
                for src, cell, _ in cells(ds):
                    is_epsvec = cell.startswith("EPSVec")
                    if source == "EPSVec":
                        if not is_epsvec:
                            continue
                    elif src != source or is_epsvec:
                        continue
                    yield ds, source, clf, cell, os.path.join(EXP_ROOT, ds, src, clf, f"{cell}.jsonnet")


def build_config(cfg_path, seed, data_root, output_root):
    c = json.loads(_jsonnet.evaluate_file(cfg_path))
    for k in ("train_file", "validation_file", "test_file"):
        c[k] = c[k].replace("./data", data_root, 1)
    c["output_dir"] = c["output_dir"].replace("./outputs", output_root, 1) + f"/seed{seed}"
    c["run_name"] = c["run_name"] + f"-seed{seed}"
    c["seed"] = seed
    c["disable_tqdm"] = True
    return c


def status(c):
    """'todo' | 'done' | 'stale:<reason>'"""
    marker = os.path.join(c["output_dir"], "experiment-args.json")
    if not os.path.isfile(marker):
        return "todo"
    a = flat_args(marker)
    diffs = [f"{k}={a.get(k)}->{c[k]}" for k in MATCH_KEYS if float(a.get(k, -1)) != float(c[k])]
    if os.path.basename(str(a.get("train_file"))) != os.path.basename(c["train_file"]):
        diffs.append("train_file")
    return "stale:" + ",".join(diffs) if diffs else "done"


def prune(out):
    for d in glob.glob(os.path.join(out, "checkpoint-*")):
        shutil.rmtree(d, ignore_errors=True)
    for pat in WEIGHT_GLOBS:
        for f in glob.glob(os.path.join(out, pat)):
            os.remove(f)


def check(out):
    """Return (eval_f1, n_unique_predictions, flags)."""
    flags = []
    try:
        f1 = json.load(open(os.path.join(out, "eval_results.json")))["eval_f1"]
    except Exception:
        return None, None, ["NO_EVAL"]
    preds = set()
    try:
        with open(os.path.join(out, "test_results_cls.txt")) as f:
            next(f)
            preds = {line.rstrip("\n").split("\t", 1)[1] for line in f if "\t" in line}
    except Exception:
        flags.append("NO_PRED")
    if f1 == 0:
        flags.append("ZERO_F1")
    if len(preds) == 1:
        flags.append("CONSTANT_PRED")
    return f1, len(preds), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--clfs", nargs="*", default=None, help="substring filter on classifier dir names")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-prune", action="store_true")
    args = ap.parse_args()

    log_dir = os.path.join(args.output_root, "logs", "utility")
    tmp_dir = os.path.join(log_dir, "configs")
    os.makedirs(tmp_dir, exist_ok=True)
    tsv = os.path.join(log_dir, "runs.tsv")
    cache = os.path.join(args.output_root, "cache")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=f"{REPO}:{os.path.join(REPO, 'src')}",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               # wandb, datasets / evaluate caches and temp files stay under the output root
               WANDB_DIR=os.path.join(cache, "wandb"),
               HF_DATASETS_CACHE=os.path.join(cache, "hf_datasets_cache"),
               TMPDIR=os.path.join(cache, "tmp"),
               HF_METRICS_CACHE=os.path.join(cache, "hf_metrics_cache"))
    # wandb logs offline under WANDB_DIR (upload with `wandb sync`); WANDB_MODE=online overrides.
    env.setdefault("WANDB_MODE", "offline")
    for k in ("WANDB_DIR", "TMPDIR", "HF_METRICS_CACHE"):
        os.makedirs(env[k], exist_ok=True)

    runs = [(r, seed) for r in plan(args.datasets, args.sources, args.clfs) for seed in args.seeds]
    counts = {"done": 0, "todo": 0, "stale": 0, "nodata": 0}
    for (ds, source, clf, cell, cfg), seed in runs:
        tag = f"{ds}/{source}/{clf}/{cell}/seed{seed}"
        if not os.path.isfile(cfg):
            log(f"MISSING CONFIG {cfg}")
            continue
        c = build_config(cfg, seed, args.data_root, args.output_root)
        if not os.path.isfile(c["train_file"]):
            counts["nodata"] += 1
            log(f"NO DATA  {tag}  ({c['train_file']})")
            continue
        st = status(c)
        counts[st.split(":")[0]] += 1
        if st == "done":
            log(f"SKIP     {tag}")
            continue
        log(f"{'RUN' if st == 'todo' else 'RERUN'}      {tag}  {st if st != 'todo' else ''}")
        if args.dry_run:
            continue
        out = c["output_dir"]
        if st.startswith("stale"):
            shutil.rmtree(out)
        os.makedirs(out, exist_ok=True)
        cfg_file = os.path.join(tmp_dir, tag.replace("/", "__") + ".jsonnet")
        with open(cfg_file, "w") as f:
            json.dump(c, f, indent=2)
        t0 = time.time()
        with open(os.path.join(out, "train.log"), "w") as lf:
            rc = subprocess.run([args.python, SCRIPT, cfg_file], cwd=REPO, env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        f1, n_pred, flags = check(out)
        if rc != 0:
            flags.insert(0, f"FAILED(rc={rc})")
        if not args.no_prune:
            prune(out)
        log(f"  -> eval_f1={f1} n_pred_labels={n_pred} {mins:.1f}min {' '.join(flags)}")
        with open(tsv, "a") as f:
            f.write("\t".join(map(str, [time.strftime("%Y-%m-%d %H:%M:%S"), args.gpu, ds, source, clf, cell,
                                        seed, rc, f1, n_pred, f"{mins:.1f}", ",".join(flags)])) + "\n")
    log(f"lane finished: {counts}")


if __name__ == "__main__":
    main()
