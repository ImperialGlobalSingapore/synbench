"""
Fidelity metrics comparing a synthetic corpus with a reference corpus: MAUVE, NER-entity n-gram KL and
text-length distribution.
"""

import re
from typing import Any
from abc import ABC, abstractmethod
import mauve
from handystuff.loaders import write_json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chisquare, ks_2samp, entropy, wasserstein_distance
import spacy
from transformers import AutoTokenizer, pipeline

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import TokenCounter, ascii_histogram, serialize_for_json


class Metric(ABC):
    name = "metric"

    def get_name(self) -> str:
        """Returns the name of the metrics.

        Override if any metric-specific parameters are to be appended.
        (e.g. N-gram metric, but what is the number of N-grams?)

        Returns:
            str: name of the metric - used for saving results.
        """
        return self.name

    @abstractmethod
    def calculate(self, corpus: list[dict], reference: list[dict]) -> Any:
        """Abstract method that calculates the metric comparing two corpora (e.g. GTs and predictions).

        Requires metric-specific implementation.

        Args:
            corpus (list[dict]): One corpus to compare.
            reference (list[dict]): Other corpus to compare to.


        Returns:
            Any: Whatever is appropriate and can be processed by print_metric. Ideally a JSON serialisable dict.
        """
        raise NotImplementedError()

    @abstractmethod
    def print_result(self, result=None):

        raise NotImplementedError()

    def save_metric(self, save_details, result=None):
        result = result or self.result
        if not result:
            raise ValueError("Need to calculate metric first!")
        save_details.update(**result)
        save_path = save_details.pop("file_name")
        write_json(serialize_for_json(save_details), save_path)


class CorpusMetric(Metric):
    def __call__(self, corpus, reference_corpus):
        self.result = self.calculate(corpus, reference_corpus)
        return self.result


class Mauve(CorpusMetric):
    name = "mauve"

    def __init__(self, use_cuda: bool = False, mauve_args: dict = None, *args, **kwargs):
        self.mauve_args = mauve_args or {}
        self.use_cuda = use_cuda
        super().__init__(*args, **kwargs)

    def calculate(self, corpus, reference):
        # mauve_star / frontier_integral_star: the smoothed variants (Pillutla et al., JMLR 2023)
        if self.use_cuda:
            result = mauve.compute_mauve(p_text=corpus, q_text=reference, device_id=0, **self.mauve_args)
        else:
            result = mauve.compute_mauve(p_text=corpus, q_text=reference, **self.mauve_args)
        return {
            "mauve": result.mauve,
            "divergence": result.frontier_integral,
            "mauve_star": result.mauve_star,
            "divergence_star": result.frontier_integral_star,
            "densities": result.divergence_curve.tolist(),
            "histograms": (result.p_hist.tolist(), result.q_hist.tolist()),
        }

    def print_result(self, result=None):
        result = result or self.result
        if not result:
            raise ValueError("Need to calculate metric first!")
        print(f"Mauve Score: {result['mauve']}")
        print(f"Divergence: {result['divergence']}")
        print(f"Mauve* Score: {result['mauve_star']}")
        print(f"Divergence*: {result['divergence_star']}")


def split_entities(text):
    return re.split("\|", text)


class ngramFrequency(CorpusMetric):
    name = "ngramFrequency"

    def __init__(
        self,
        ngram: list = None,
        topk=0,
        vectorizer_args: dict = None,
        ner_only=False,
        ner_model="blaze999/Medical-NER",
        entity_types=None,  # Filter specific entity types (PER, LOC, ORG, etc.)
        aggregation_strategy="simple",
        batch_size=32,
        use_cuda=True,
        split_by_sentence=False,
        max_length=512,
        stride=128,
        *args,
        **kwargs,
    ):
        """max_length / stride: inputs longer than max_length tokens (the NER models' trained length)
        are split into overlapping windows that share `stride` tokens, so no text is dropped; set
        stride=None to truncate at max_length instead."""
        self.ngram = ngram or [1, 2, 3]
        self.vectorizer_args = vectorizer_args or {}
        self.topk = topk
        self.ner_only = ner_only
        self.ner_model = ner_model
        self.entity_types = entity_types
        self.aggregation_strategy = aggregation_strategy
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.split_by_sentence = split_by_sentence
        self.max_length = max_length
        self.stride = stride
        self._is_init = False

        # Initialize NER pipeline if needed
        self.init()
        super().__init__(*args, **kwargs)

    def init(self):
        if self._is_init:
            return
        if self.ner_only:
            tokenizer = AutoTokenizer.from_pretrained(self.ner_model)
            tokenizer.model_max_length = self.max_length
            extra = {"stride": self.stride} if self.stride else {}
            self.ner_pipeline = pipeline(
                "ner",
                tokenizer=tokenizer,
                model=self.ner_model,
                aggregation_strategy=self.aggregation_strategy,
                device="cuda" if self.use_cuda else "cpu",
                **extra,
            )
            if self.split_by_sentence:
                self.nlp = spacy.blank("en")
                self.nlp.add_pipe("sentencizer")
            else:
                self.nlp = None
        self._is_init = True

    def merge_entities(self, entities):
        """
        Merge entities where end position of one equals start position of the next.

        Args:
            entities: List of entity dictionaries with 'start', 'end', and 'word' keys

        Returns:
            List of merged entity dictionaries
        """
        if not entities:
            return []

        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: e["start"])

        merged = []
        current = sorted_entities[0].copy()

        for entity in sorted_entities[1:]:
            # Check if this entity starts where the previous one ends
            if entity["start"] == current["end"]:
                # Merge the entities
                current["word"] = current["word"] + " " + entity["word"]
                current["end"] = entity["end"]

                # Store all entity types
                if "all_entity_groups" not in current:
                    current["all_entity_groups"] = [current["entity_group"]]
                current["all_entity_groups"].append(entity["entity_group"])

                # Average the scores
                current["score"] = (current["score"] + entity["score"]) / 2
            else:
                # This entity doesn't connect to the previous one
                merged.append(current)
                current = entity.copy()

        # Don't forget to add the last entity
        merged.append(current)

        return merged

    def get_name(self) -> str:
        """Returns the customized name of the metric."""
        name = f"{self.name}_ngram-{'-'.join(map(str, self.ngram))}"
        if self.ner_only:
            model_name = self.ner_model.split("/")[-1]
            name += f"_ner-{model_name}"
            if self.entity_types:
                name += f"_entities-{'-'.join(self.entity_types)}"
        return name

    def extract_entities_from_corpus(self, texts):
        """Extract named entities from a corpus of texts using batched processing."""
        if not texts:
            return []
        if self.split_by_sentence:
            num_texts = len(texts)
            texts = [s.text.strip() for doc in self.nlp.pipe(texts) for s in doc.sents]
            print(f"Converted {num_texts} docs into {len(texts)} sentences...")
        print(f"Extracting entities from {len(texts)} texts...")
        # Get all entities using pipeline's built-in batching
        all_entities_results = self.ner_pipeline(texts, batch_size=self.batch_size)

        # Process the results (converting to "|"-separated entity strings)
        entity_texts = []
        for text_entities in all_entities_results:
            text_entities = self.merge_entities(text_entities)

            # Filter by entity type if specified
            if self.entity_types:
                # For merged entities, check if any of the original types match
                text_entities = [
                    e
                    for e in text_entities
                    if ("all_entity_groups" in e and any(typ in self.entity_types for typ in e["all_entity_groups"]))
                    or (e["entity_group"] in self.entity_types)
                ]

            if text_entities:
                entity_texts.append("|".join(e["word"].lower() for e in text_entities))

        print(f"Found {len(entity_texts)} documents with entities")
        return entity_texts

    def vectorize(self, vectorizer, corpus):
        freq = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names_out()
        # keep top k features and frequencies
        freq = freq.sum(axis=0) / len(corpus)
        if self.topk:
            topk_indices = freq.argsort()[-self.topk :]
        else:
            topk_indices = range(len(freq))
        return {feat: val for feat, val in zip(feature_names[topk_indices], freq[topk_indices])}

    def calculate(self, corpus, reference):
        self.init()
        # Extract NER entities if needed
        if self.ner_only:
            corpus_texts = self.extract_entities_from_corpus(corpus)
            reference_texts = self.extract_entities_from_corpus(reference)
            # Handle edge case where there are no entities
            if not corpus_texts or not reference_texts:
                return {"error": "No entities found in one or both corpora"}
        else:
            corpus_texts = corpus
            reference_texts = reference
        print(sum(1 for r in corpus_texts for _ in r.split("|")), "entities (corpus)")
        print(
            len(set(d for r in corpus_texts for d in r.split("|"))),
            "unique entities (corpus)",
        )

        print(
            sum(1 for r in reference_texts for _ in r.split("|")),
            "entities (reference)",
        )
        print(
            len(set(d for r in reference_texts for d in r.split("|"))),
            "unique entities (reference)",
        )
        self.result = {}
        for n in self.ngram:
            if self.ner_only:
                vectorizer = CountVectorizer(ngram_range=(n, n), tokenizer=split_entities, **self.vectorizer_args)
            else:
                vectorizer = CountVectorizer(ngram_range=(n, n), **self.vectorizer_args)
            corpus_topk = self.vectorize(vectorizer, corpus_texts)
            reference_topk = self.vectorize(vectorizer, reference_texts)
            corpus_freq, reference_freq = [], []
            for feat in set(corpus_topk.keys()) | set(reference_topk.keys()):
                corpus_freq.append(corpus_topk.get(feat, 1e-6))
                reference_freq.append(reference_topk.get(feat, 1e-6))
            # calculate KL divergence
            kl_div = entropy(pk=reference_freq, qk=corpus_freq)
            self.result[f"{n}-gram"] = float(kl_div)
        return self.result

    def print_result(self, result=None):
        result = result or self.result
        if not result:
            raise ValueError("Need to calculate metric first!")
        for ngram, score in result.items():
            print(f"{ngram} KL Divergence: {score}")


class TextLengthDistribution(CorpusMetric):
    name = "text-length-distribution"

    def __init__(
        self,
        model="meta-llama/Llama-3.1-8B-Instruct",
        show_histogram=True,
        bins=10,
        *args,
        **kwargs,
    ):
        """
        Initialize TextLengthDistribution metric.

        Args:
            model: The model to use for tokenization
            show_histogram: Whether to show ASCII histogram
            bins: Number of bins for length histogram
        """
        self.model = model
        self.show_histogram = show_histogram
        self.bins = bins
        self.token_counter = TokenCounter(model)
        super().__init__(*args, **kwargs)

    def get_name(self) -> str:
        """Returns the customized name of the metric."""
        return f"{self.name}-{self.bins}bins"

    def calculate(self, corpus, reference):
        """Calculate text length statistics and compare distributions between two corpora."""
        # Calculate statistics for both corpora
        corpus_stats, corpus_lengths = self._calculate_text_stats(corpus)
        reference_stats, reference_lengths = self._calculate_text_stats(reference)

        # Compare distributions
        comparison = self._compare_length_distribution(corpus_lengths, reference_lengths)

        # Generate text output for corpus
        corpus_output = self._generate_stats_output(corpus_stats, corpus_lengths)

        # Generate text output for reference
        reference_output = self._generate_stats_output(reference_stats, reference_lengths)

        # Combine results
        result = {
            "corpus_stats": corpus_stats,
            "reference_stats": reference_stats,
            "corpus_output": corpus_output,
            "reference_output": reference_output,
            "comparison": comparison,
        }

        return result

    def _calculate_text_stats(self, data):
        """Calculate statistics about text lengths in a dataset of dictionaries."""
        # Count tokens per text
        lengths = [self.token_counter.count_tokens(record) for record in data]

        # Print non-zero lengths
        print(f"Non-zero lengths: {len([l for l in lengths if l > 0])}")

        # Calculate statistics
        stats = {
            "#": len(lengths),
            "mean": np.mean(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
            "p90": np.percentile(lengths, 90),
            "p95": np.percentile(lengths, 95),
            "p99": np.percentile(lengths, 99),
        }

        # Round all values to 2 decimal places
        stats = {k: round(v, 2) for k, v in stats.items()}
        return stats, lengths

    def _generate_stats_output(self, stats, lengths):
        """Generate text output for statistics."""
        output = []
        output.append("Text Length Statistics:")
        output.append("-" * 20)
        for stat, value in stats.items():
            output.append(f"{stat:>4}: {value:<8.1f}")

        if self.show_histogram:
            output.append("\nLength Distribution:")
            output.append("-" * 20)
            result, hist, bins = ascii_histogram(lengths)
            output.append(result)

        return "\n".join(output)

    def _compare_length_distribution(self, lengths1, lengths2):
        """Compare length distribution differences between two datasets."""
        min_len = min(min(lengths1), min(lengths2))
        max_len = max(max(lengths1), max(lengths2))

        # Create histograms
        hist1, bins = np.histogram(lengths1, bins=self.bins, range=(min_len, max_len))
        hist2, _ = np.histogram(lengths2, bins=self.bins, range=(min_len, max_len))

        # Normalize histograms (avoiding division by zero)
        sum1 = np.sum(hist1)
        sum2 = np.sum(hist2)

        if sum1 == 0 or sum2 == 0:
            print("Warning: One of the histograms has zero elements. Comparison may not be meaningful.")
            norm_hist1 = hist1.astype(float) + 1e-6
            norm_hist2 = hist2.astype(float) + 1e-6
        else:
            norm_hist1 = hist1 / sum1 + 1e-6
            norm_hist2 = hist2 / sum2 + 1e-6

        # Calculate chi-square test (using normalized histograms)
        chi_stat, chi_p_value = chisquare(f_exp=norm_hist1, f_obs=norm_hist2)

        # Calculate KS test
        ks_stat, ks_p_value = ks_2samp(lengths1, lengths2)

        # Earth Mover's Distance
        emd = wasserstein_distance(norm_hist1, norm_hist2)

        # KL divergence
        kl_div = entropy(pk=norm_hist1, qk=norm_hist2)

        return {
            "chisquare": (chi_stat, chi_p_value),
            "ks": (ks_stat, ks_p_value),
            "emd": emd,
            "kl_div": kl_div,
            "bins": bins.tolist(),
            "hist1": hist1.tolist(),
            "hist2": hist2.tolist(),
            "norm_hist1": norm_hist1.tolist(),
            "norm_hist2": norm_hist2.tolist(),
        }

    def print_result(self, result=None):
        """Print the results."""
        result = result or self.result
        if not result:
            raise ValueError("Need to calculate metric first!")

        # Print corpus stats output
        print("Corpus:")
        print(result["corpus_output"])

        # Print reference stats output
        print("\nReference:")
        print(result["reference_output"])

        # Print comparison results
        print("\nDistribution Comparison:")
        print("-" * 20)
        comparison = result["comparison"]
        print(f"Chi-square: {comparison['chisquare'][0]:.4f} (p-value: {comparison['chisquare'][1]:.4f})")
        print(f"KS test: {comparison['ks'][0]:.4f} (p-value: {comparison['ks'][1]:.4f})")
        print(f"Earth Mover's Distance: {comparison['emd']:.4f}")
        print(f"KL Divergence: {comparison['kl_div']:.4f}")
