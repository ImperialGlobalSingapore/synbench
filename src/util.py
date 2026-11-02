from collections import defaultdict
from pathlib import Path
import re
import pandas as pd
import numpy as np
import tiktoken
from _jsonnet import evaluate_file

import json
import os
import sys

from handystuff.loaders import load_json


# Root for pipeline outputs; override with the OUTPUT_ROOT env var.
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")


def serialize_for_json(obj):
    """
    Recursively convert numpy arrays, tensors, and other non-serializable objects
    to native Python types for JSON serialization.

    Args:
        obj: Any Python object to be serialized

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    elif hasattr(obj, "tolist"):  # For PyTorch tensors and some other libraries
        return obj.tolist()
    elif hasattr(obj, "numpy"):  # For TensorFlow tensors
        return obj.numpy().tolist()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, defaultdict):
        return {k: serialize_for_json(v) for k, v in dict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return [serialize_for_json(item) for item in obj]
    # Add more type conversions as needed
    return obj


def parse_data_path(path):
    """(dataset, generator, approach) of .../data/<dataset>/[<generator>/]<approach>/<file>.

    The release data has a generator level (e.g. Llama-3.2-1B, OLMo-2-0425-1B-10epochs); real data
    sits under <dataset>/original and gets generator "original". Paths without a generator level give
    generator None.
    """
    parts = Path(path).parts
    k = len(parts) - 1 - parts[::-1].index("data")
    dataset, middle = parts[k + 1], parts[k + 2:-1]
    if middle == ("original",):
        return dataset, "original", "original"
    if len(middle) == 2:
        return dataset, middle[0], middle[1]
    return dataset, None, middle[-1]


def get_save_path(dataset1, dataset2, metric, approach1=None, approach2=None):
    # Extract metric name
    metric_name = metric.get_name()

    if not os.path.isdir(dataset1) and "data" in Path(dataset1).parts and "data" in Path(dataset2).parts:
        # Results go to fidelity/<generator>/ so runs of different generators (whose file names are
        # otherwise identical) do not overwrite each other; real-vs-real goes to fidelity/original/.
        name1, gen1, app1 = parse_data_path(dataset1)
        name2, gen2, app2 = parse_data_path(dataset2)
        file1 = Path(dataset1).name.rsplit(".", 1)[0]
        file2 = Path(dataset2).name.rsplit(".", 1)[0]
        e_1 = next((k.group() for k in re.finditer(r"\d*\.?\d+", file1)), "")
        e_2 = next((k.group() for k in re.finditer(r"\d*\.?\d+", file2)), "")
        name2_str = f"-{name2}" if name2 != name1 else ""
        result = {
            "file_name": f"{OUTPUT_ROOT}/fidelity/{gen1 or 'unknown'}/"
                         f"{metric_name}-{name1}-{file1}-vs{name2_str}-{file2}.json",
            "dataset_1": {"name": name1, "split": file1.replace(e_1, "").replace("nodp", "").split("-")[0],
                          "generator": gen1, "approach": approach1 or app1},
            "dataset_2": {"name": name2, "split": file2.replace(e_2, "").replace("nodp", "").split("-")[0],
                          "generator": gen2, "approach": approach2 or app2},
            "metric": metric_name,
        }
        if e_1:
            result["dataset_1"]["epsilon"] = e_1
        if e_2:
            result["dataset_2"]["epsilon"] = e_2
        return result

    raise ValueError(f"expected two dataset files under data/, got {dataset1!r} and {dataset2!r}")


def get_args(*arg_types):
    from transformers import HfArgumentParser

    parser = HfArgumentParser(arg_types)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".jsonnet"):
        conf = json.loads(evaluate_file(sys.argv[1]))
        args = parser.parse_dict(conf)
    else:
        args = parser.parse_args_into_dataclasses()
    return args


class DatasetLoader:
    """Helper class to load a jsonl / json / csv dataset as a list of records."""

    def __init__(self, file):
        self.file = file
        self.data = None

    def load_data(self):
        if self.file.endswith(".jsonl"):
            # split on "\n" only: str.splitlines() (used by handystuff's load_jsonl) also breaks on
            # U+2028 etc. inside JSON strings, which some generated texts contain
            with open(self.file, encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f.read().split("\n") if line.strip()]
        elif self.file.endswith(".csv"):
            self.data = pd.read_csv(self.file).to_dict("records")
        elif self.file.endswith(".json"):
            self.data = load_json(self.file)
        else:
            raise ValueError(f"Unsupported file format: {self.file}")
        return self.data


class TokenCounter:
    """Helper class to manage different tokenizers"""

    def __init__(self, model_name):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.tokenizer = None

        # Try tiktoken first
        try:
            if any(name in model_name.lower() for name in ["gpt", "text-davinci", "ada"]):
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                self.type = "tiktoken"
            else:
                raise ValueError("Not a tiktoken model")
        except (ValueError, KeyError):
            # Fall back to HuggingFace tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.type = "huggingface"
            except Exception as e:
                raise ValueError(f"Could not load tokenizer for {model_name}: {str(e)}")

    def count_tokens(self, text):
        """Count tokens in the given text"""
        if self.type == "tiktoken":
            return len(self.tokenizer.encode(text))
        else:  # huggingface
            return len(self.tokenizer.encode(text, add_special_tokens=True))


def ascii_histogram(data, bins=10, width=50):
    """
    Create an ASCII histogram

    Args:
        data: List of numbers
        bins: Number of bins
        width: Width of the histogram in characters
    """
    # Calculate bins
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)

    # Generate histogram
    result = []
    for count, bin_start, bin_end in zip(hist, bin_edges[:-1], bin_edges[1:]):
        bar_length = int(width * count / max_count)
        bar = "█" * bar_length
        result.append(f"{int(bin_start):4d}-{int(bin_end):<4d} | {bar:<{width}} | {count:3d}")

    return "\n".join(result), hist, bin_edges
