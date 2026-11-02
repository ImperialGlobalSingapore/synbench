from ngram import NGramModel
import numpy as np
import os
import random
import warnings

class NGramReferenceAttack:
    def __init__(self, reference_jsonl_paths, ngram_n=2, text_key="text"):
        """
        Fit M ngram models on reference datasets (D').
        """
        self.text_key = text_key
        self.reference_models = []
        for ref_path in reference_jsonl_paths:
            model = NGramModel(n=ngram_n)
            model.fit(ref_path, text_key=text_key)
            self.reference_models.append(model)

    def fit_target(self, target_jsonl_path):
        """
        Fit ngram model on the target dataset (D).
        """
        self.target_model = NGramModel(n=self.reference_models[0].n)
        self.target_model.fit(target_jsonl_path, text_key=self.text_key)

    def predict_logit(self, sample_text):
        """
        For a given sample x, compute:
          - a: log-likelihood of x under the target model
          - b: mean log-likelihood of x under the reference models
        Return logit = log(a/b)
        """
        ref_probs = [model.prob(sample_text) for model in self.reference_models]
        avg_ref_prob = np.mean(ref_probs)
        target_prob = self.target_model.prob(sample_text)
        logit = target_prob - avg_ref_prob  # log(a/b)
        return logit


def ngram_attack_sampled_datasets(d_idx_list,
    epsilon,
    target_jsonl_dir,
    label_records,
    attackmodel,
    rng=None,
):
    """
    One trial of the attack over the releases in d_idx_list at one epsilon.

    For each release (synthetic dataset_{idx}_{epsilon}_samples.jsonl), fit the target n-gram model on it,
    then score ONE text:
      - member release (added_special): the inserted outlier, special_text; label 1
      - non-member release: one of its `not_added_samples` drawn at random with `rng`; label 0
    Returns y (labels), y_hat (scores) and targets (the outlier `index` that was scored: special_index for
    members, the drawn candidate's index for non-members), aligned position by position. A release whose
    synthetic file is missing is skipped with a warning, so the lists are then shorter than d_idx_list.
    rng: a random.Random (seed it for a reproducible trial); defaults to the module-level `random`.
    """
    rng = rng or random
    y, y_hat, targets = [], [], []

    for idx in d_idx_list:
        record = label_records[idx]
        target_json_path = target_jsonl_dir + f"dataset_{idx}_{epsilon}_samples.jsonl"
        if not os.path.exists(target_json_path):
            warnings.warn(f"missing synthetic release {target_json_path}; release {idx} skipped", RuntimeWarning)
            continue
        attackmodel.fit_target(target_json_path)
        if record["added_special"]:
            y.append(1)
            y_hat.append(attackmodel.predict_logit(record["special_text"]))
            targets.append(int(record["special_index"]))
        else:
            y.append(0)
            chosen = rng.choice(record["not_added_samples"])
            y_hat.append(attackmodel.predict_logit(chosen["text"]))
            targets.append(int(chosen["index"]))
    return y, y_hat, targets
