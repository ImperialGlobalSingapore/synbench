import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import json
import random
import pickle
from attack import NGramReferenceAttack, ngram_attack_sampled_datasets
from tqdm import tqdm
import os

def cal_mean_roc_with_ci(y_list, y_hat_list):
    """
    Calculate the mean ROC curve and 95% confidence interval from multiple runs.
    Args:
        y_list: list of true label lists
        y_hat_list: list of predicted score lists
    Returns:
        [mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc]
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    n = len(y_list)

    for y, y_hat in zip(y_list, y_hat_list):
        fpr, tpr, _ = roc_curve(y, y_hat)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    tprs = np.array(tprs)
    aucs = np.array(aucs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # 95% confidence interval
    ci_factor = 1.96 / np.sqrt(n)
    lower_tpr = mean_tpr - ci_factor * std_tpr
    upper_tpr = mean_tpr + ci_factor * std_tpr
    lower_auc = mean_auc - ci_factor * std_auc
    upper_auc = mean_auc + ci_factor * std_auc

    # Clip bounds to [0,1]
    lower_tpr = np.clip(lower_tpr, 0, 1)
    upper_tpr = np.clip(upper_tpr, 0, 1)
    lower_auc = max(lower_auc, 0)
    upper_auc = min(upper_auc, 1)

    resultlist = [mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc]
    return resultlist


EPS_KEYS = ["e0", "e4", "e2", "e1", "e0.5"]  # e0 = no DP (eps = inf); augpe keys stand for noise levels
N_REPEAT = 50
NGRAM = 2


def mia_results_path(datasetname, baselinename):
    return f"./outputs/mia/{datasetname}_{baselinename}_mia_results.pkl"


def mia_collect_results_ci(datasetname, baselinename, seed=0, n_repeat=N_REPEAT):
    """Run the bigram reference attack over the 100 releases, n_repeat trials per epsilon, and save

        results[eps] = [roc_summary, [labels, scores, targets]]     (eps in EPS_KEYS)

    roc_summary = [mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc]
    (cal_mean_roc_with_ci: mean over trials +- 1.96 * sd / sqrt(n_repeat)); labels / scores / targets are
    n_repeat x n_releases lists: membership (1 = member), attack score, and the outlier `index` that was
    scored (members: special_index; non-members: the candidate drawn from not_added_samples). Trial t at
    epsilon eps draws with random.Random(f"{seed}-{eps}-{t}"), so a rerun gives the same draws.
    datasetname: e.g. DMSAFN; baselinename: augpe | dp-transformers. Existing results are loaded, not rerun.
    """
    label_path = f"./data/{datasetname}/mia/D_sample/sampled_private_datasets/sampling_labels.json"
    target_jsonl_dir = f"./data/{datasetname}/mia/D_synth/{baselinename}/finalresults/"
    os.makedirs("./outputs/mia/", exist_ok=True)
    results_filepath = mia_results_path(datasetname, baselinename)
    if os.path.exists(results_filepath):
        print(f"Result file {results_filepath} already exists. Loading and returning existing results.")
        with open(results_filepath, 'rb') as f:
            return pickle.load(f)

    with open(label_path, "r") as f:
        label_records = json.load(f)
    d_idx_list = list(range(len(label_records)))
    ref_dir = f"./data/{datasetname}/mia/D_sample/sampled_4ref_datasets_outliers/"
    attackmodel = NGramReferenceAttack([f"{ref_dir}dataset_{i}.jsonl" for i in range(4)], ngram_n=NGRAM, text_key="text")
    print(f"Dataset: {datasetname}, Baseline: {baselinename}, bigram reference attack, {n_repeat} trials, seed {seed}")

    results_dic = {}
    for para in EPS_KEYS:
        labels, scores, targets = [], [], []
        for t in tqdm(range(n_repeat), desc=para):
            y, y_hat, tgt = ngram_attack_sampled_datasets(d_idx_list, para, target_jsonl_dir, label_records,
                                                          attackmodel, rng=random.Random(f"{seed}-{para}-{t}"))
            assert len(y) == len(y_hat) == len(tgt), f"Length mismatch: {len(y)} / {len(y_hat)} / {len(tgt)}"
            labels.append(y)
            scores.append(y_hat)
            targets.append(tgt)
        results_dic[para] = [cal_mean_roc_with_ci(labels, scores), [labels, scores, targets]]
        print(f"{para}: mean AUC {results_dic[para][0][4]:.3f}")

    pickle.dump(results_dic, open(results_filepath, 'wb'), protocol=5)
    print(f"Saved {results_filepath}")
    return results_dic


def plot_roc_with_ci(mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc, label, color="C0"):
    """
    Plot ROC curve with confidence interval.
    """
    plt.plot(mean_fpr, mean_tpr, color=color, label=f"{label} (AUC={mean_auc:.3f} [{lower_auc:.3f}, {upper_auc:.3f}])")
    plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color=color, alpha=0.2)



if __name__ == "__main__":
    # Run from the repo root: python src/mia/roc_plot.py DMSAFN dp-transformers
    import sys
    datasetname = sys.argv[1] if len(sys.argv) > 1 else "DMSAFN"
    baselinename = sys.argv[2] if len(sys.argv) > 2 else "dp-transformers"
    results_dic = mia_collect_results_ci(datasetname, baselinename)

    plt.figure(figsize=(8, 6))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for idx, (para, (roc_data, _)) in enumerate(results_dic.items()):
        plot_roc_with_ci(*roc_data, label=f"{para}", color=colors[idx % len(colors)])
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve with 95% CI ({datasetname}, {baselinename})")
    plt.legend()
    plt.tight_layout()
    fig_path = f"./outputs/figs/mia_{datasetname}_{baselinename}_roc.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    print(f"ROC curve saved to {fig_path}")
