"""Bigram (n-gram) language model used by the reference MIA (src/mia/attack.py).

  - whitespace `str.split()` tokenization
  - order n=2 (bigram) by default
  - sentence padding: ["<s>"]*(n-1) + tokens + ["</s>"]
  - add-1 (Laplace) smoothing: prob = (count + 1) / (total + V), V = |vocab| + 1
  - prob() returns the SUM of log-probabilities (a log-likelihood, base e)
"""

import json
import math
from collections import defaultdict, Counter


class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.total_ngrams = 0

    @staticmethod
    def _tokenize(text):
        return text.split()

    def fit(self, jsonl_path, text_key="text"):
        """Fit the n-gram model on a jsonl dataset."""
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                tokens = self._tokenize(text)
                self.vocab.update(tokens)
                tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
                for i in range(len(tokens) - self.n + 1):
                    context = tuple(tokens[i:i + self.n - 1])
                    word = tokens[i + self.n - 1]
                    self.ngram_counts[context][word] += 1
                    self.context_counts[context] += 1
                    self.total_ngrams += 1

    def prob(self, text):
        """Log-probability (sum of log terms) of `text` under the fitted model,
        with add-1 smoothing."""
        tokens = self._tokenize(text)
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob = 0.0
        V = len(self.vocab) + 1  # +1 for unseen words
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            word = tokens[i + self.n - 1]
            count = self.ngram_counts[context][word]
            total = self.context_counts[context]
            prob = (count + 1) / (total + V)
            log_prob += math.log(prob)
        return log_prob
