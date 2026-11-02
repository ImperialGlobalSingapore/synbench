import os
import sys

from fire import Fire
from handystuff.imports import construct_from_config
from handystuff.loaders import load_jsonnet

from src.fidelity.metrics import Metric

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import DatasetLoader, get_save_path


def main(
    dataset1: str,
    dataset2: str,
    metrics_cfg: str,
    key1="text",
    key2=None,
    overwrite: bool = False,
    ignore_metrics: list[str] = None,
):
    """
    Compute the fidelity metrics of a metrics config between a synthetic and a reference dataset.

    Args:
        dataset1: Path to the synthetic dataset (jsonl)
        dataset2: Path to the reference dataset (jsonl)
        metrics_cfg: jsonnet file with the metrics to run (configs/fidelity/sim-metrics-<dataset>.jsonnet)
        key1: Text field of dataset1
        key2: Text field of dataset2 (default: key1)
        overwrite: Recompute metrics whose result file already exists
        ignore_metrics: Names of metrics in the config to skip
    """
    ignore_metrics = ignore_metrics if ignore_metrics else []
    if not isinstance(ignore_metrics, (list, set)):
        ignore_metrics = [ignore_metrics]
    ignore_metrics = set(ignore_metrics)

    metrics = load_jsonnet(metrics_cfg)

    key2 = key2 or key1
    texts1 = [d[key1] for d in DatasetLoader(dataset1).load_data()]
    texts2 = [d[key2] for d in DatasetLoader(dataset2).load_data()]

    for k, metric in metrics.items():
        if k in ignore_metrics:
            print(f"Ignoring {k}...")
            continue
        metric_instance: Metric = construct_from_config(metric)
        save_details = get_save_path(dataset1, dataset2, metric_instance)
        print(f"Will save to: {save_details['file_name']}")
        if os.path.exists(save_details["file_name"]) and not overwrite:
            print(
                f"skipping {metric_instance.get_name()}, because {save_details['file_name']} exists."
                "To force, append `--overwrite`."
            )
            continue
        print("Running", metric_instance.get_name())
        print(
            f"Comparing {save_details['dataset_1']['name']} ({save_details['dataset_1']['approach']}) "
            f"to {save_details['dataset_2']['name']} ({save_details['dataset_2']['approach']})."
        )
        metric_instance(texts1, texts2)
        metric_instance.print_result()

        metric_instance.save_metric(save_details=save_details)
        del metric_instance  # Free any GPU resources


if __name__ == "__main__":
    Fire(main)
