#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from json import JSONEncoder
from typing import Any, Optional

import evaluate
import handystuff.logs
import numpy as np
import sklearn.metrics
import wandb
from transformers import EarlyStoppingCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import get_args


class Encoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value


def dump_args(*args, file_name):
    d = {str(type(arg).__name__): asdict(arg) for arg in args}
    with open(file_name, "w+") as f:
        json.dump(d, f, cls=Encoder, indent=4)


from datasets import load_dataset, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

check_min_version("4.5.0")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )

    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    sentence_col: Optional[str] = field(default=None, metadata={"help": "Name of column to use as input"})

    target_col: str = field(default="label", metadata={"help": "Name of column to use as label"})

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    reinit_pretrained_head: bool = field(
        default=True,
        metadata={"help": "Re-initialize a classification head that was loaded from the checkpoint (e.g. a sentiment "
                          "head of a finance model), so that every classifier starts from a random head."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    model_args, data_args, training_args = get_args(ModelArguments, DataTrainingArguments, TrainingArguments)
    data_args: DataTrainingArguments
    model_args: ModelArguments
    training_args: TrainingArguments
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    handystuff.logs.setup_logging()
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

    file_format = "csv" if data_args.train_file.endswith(".csv") else "json"
    # Load each split independently rather than in one load_dataset() call.
    # Synthetic train files (e.g. OLMo) may carry an extra column such as
    # `prompt` that the real validation/test splits lack; a combined load casts
    # all splits to one schema and raises DatasetGenerationCastError on the
    # mismatch. Per-split loading avoids that -- any extra column is harmless
    # (sentence_col/target_col are used explicitly, and Trainer drops the rest).
    datasets = DatasetDict(
        {split: load_dataset(file_format, data_files={split: path})[split] for split, path in data_files.items()}
    )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    if isinstance(datasets["train"][0][data_args.target_col], list):
        # Labels are stored as a list per example (e.g. ["neutral"] or ["a", "b"]): multi-label if any
        # example carries more than one label, otherwise unwrapped to a scalar label below.
        label_lists = list(datasets["train"][data_args.target_col])
        for split in ("validation", "test"):
            if split in datasets:
                label_lists += datasets[split][data_args.target_col]
        is_multilabel = max(len(ls) for ls in label_lists) > 1
        label_list = sorted(set(l for ls in label_lists for l in ls))
        if is_multilabel:
            label2id = {class_: id for id, class_ in enumerate(label_list)}
            id2label = {id: class_ for class_, id in label2id.items()}
        else:
            # At most one label per example: unwrap the list to a scalar so this goes through
            # the standard softmax/cross-entropy path below instead of sigmoid/BCE.
            def _unwrap_label(example):
                ls = example[data_args.target_col]
                example[data_args.target_col] = ls[0] if len(ls) > 0 else -1
                return example
            datasets = datasets.map(_unwrap_label)
    else:
        label_list = datasets["train"].unique(data_args.target_col)
        is_multilabel = False
        label_list.sort()  # Let's sort it for determinism

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_kwargs = dict(
        pretrained_model_name_or_path=(model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if is_multilabel:
        model_kwargs["id2label"] = id2label
        model_kwargs["label2id"] = label2id
        model_kwargs["problem_type"] = "multi_label_classification"
    config = AutoConfig.from_pretrained(**model_kwargs)
    if "longformer" in model_args.model_name_or_path or "roberta" in model_args.model_name_or_path:
        config.update({"gradient_checkpointing": True})
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        output_loading_info=True,
    )
    # A checkpoint with a classification head of the same size (finance sentiment models) loads that head,
    # which then predicts the dataset's labels before training (in a label order that may not match ours).
    # Re-initialize it so that all classifiers start from a random head.
    head = [n for n, _ in model.named_modules() if n == "classifier"]
    head_loaded = head and not any(k.startswith("classifier.") for k in loading_info["missing_keys"])
    if model_args.reinit_pretrained_head and head_loaded:
        for m in model.classifier.modules():
            model._init_weights(m)
        logger.warning(f"Re-initialized the pretrained classification head of {model_args.model_name_or_path}.")
    print(f"pretrained_head_loaded={bool(head_loaded)} reinit={bool(head_loaded and model_args.reinit_pretrained_head)}")

    # Preprocessing the datasets

    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in datasets["train"].column_names if name != data_args.target_col]
    sentence1_key, sentence2_key = data_args.sentence_col, None
    if not sentence1_key:
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if "sentence" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence", None
            elif len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id
        != PretrainedConfig(num_labels=num_labels).label2id
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k: v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result."
            )
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and data_args.target_col in examples:
            if is_multilabel:
                labels = []
                for all_labels in examples[data_args.target_col]:
                    label = [0.0 for i in range(len(label_list))]
                    for l in all_labels:
                        label_id = label_to_id[l]
                        label[label_id] = 1.0
                    labels.append(label)

                result["label"] = labels

            else:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[data_args.target_col]]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        batch_size=5,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            indices = random.sample(range(len(train_dataset)), data_args.max_train_samples)
            train_dataset = train_dataset.select(indices)

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    if is_multilabel:
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = sigmoid(predictions)
            predictions = (predictions > 0.5).astype(int).reshape(-1)
            return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
    else:
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            return {
                "accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
                "mcc": sklearn.metrics.matthews_corrcoef(preds, p.label_ids),
                # Configs set metric_for_best_model: 'f1'; macro-average since these are
                # multi-class (not binary) single-label tasks, so checkpoint selection
                # isn't dominated by whichever class happens to be most common.
                "f1": sklearn.metrics.f1_score(p.label_ids, preds, average="macro"),
                # Also report micro-F1 so single-label rows are comparable to the
                # multi-label branch's "f1" (which is micro, see evaluate's default
                # average="binary" on a flattened multi-hot array). Note this is
                # mathematically identical to "accuracy" above for single-label
                # multi-class -- it's reported for table consistency, not new signal.
                "f1_micro": sklearn.metrics.f1_score(p.label_ids, preds, average="micro"),
            }

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = ["cls"]
        test_datasets = [test_dataset]

        for test_dataset, task in zip(test_datasets, tasks):
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.argmax(predictions, axis=1) if not is_multilabel else (sigmoid(predictions) > 0.5).astype(int)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item] if not is_multilabel else [k for k,v in label_to_id.items() if item[v] > 0.5]
                        writer.write(f"{index}\t{str(item)}\n")
    fn = os.path.join(training_args.output_dir, "experiment-args.json")
    dump_args(model_args, data_args, training_args, file_name=fn)
    wandb.save(fn)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
