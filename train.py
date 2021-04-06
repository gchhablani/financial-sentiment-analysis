"""
The train and predict script.

This script uses datasets, omegaconf and transformers libraries.
Please install them in order to run this script.

Usage:
    $python train.py -config_dir ./configs/bert-base

"""
import argparse
import json
import os
import pickle as pkl

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.mapper import configmapper
from src.utils.misc import seed


class MyEncoder(json.JSONEncoder):
    """Class to convert NumPy stuff to JSON-writeable."""

    def default(self, obj):
        """Convert NumPy stuff to regular Python stuff.

        Args:
            obj (object): Object to be converted.

        Returns:
            object: Converted object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


dirname = os.path.dirname(__file__)
## Config
parser = argparse.ArgumentParser(
    prog="train.py", description="Train a model and predict."
)
parser.add_argument(
    "-config_dir",
    type=str,
    action="store",
    help="The configuration for training",
    default=os.path.join(dirname, "./configs/bert-base"),
)

parser.add_argument(
    "--only_predict",
    action="store_true",
    help="Whether to just predict, or also train",
    default=False,
)
parser.add_argument(
    "--load_predictions",
    action="store_true",
    help="Whether to load_predictions from raw_predictions_file or predict from scratch",
    default=False,
)
args = parser.parse_args()
train_config = OmegaConf.load(os.path.join(args.config_dir, "train.yaml"))
dataset_config = OmegaConf.load(os.path.join(args.config_dir, "dataset.yaml"))

seed(train_config.args.seed)

# Load datasets
print("### Loading Datasets ###")

train_dataset = configmapper.get("datasets", dataset_config.dataset_name)(
    **dataset_config.train
)
test_dataset = configmapper.get("datasets", dataset_config.dataset_name)(
    **dataset_config.test
)

# Train

if not args.only_predict:
    print("### Getting Training Args ###")
    train_args = TrainingArguments(**train_config.args)
else:
    print("### Getting Training Args from PreTrained###")
    train_args = torch.load(
        os.path.join(train_config.trainer.save_model_name, "training_args.bin")
    )
    print(train_args)
if not args.only_predict:
    print("### Loading Tokenizer for Trainer ###")
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.trainer.pretrained_tokenizer_name
    )
else:
    print("### Loading Tokenizer for Trainer from PreTrained ")
    tokenizer = AutoTokenizer.from_pretrained(train_config.trainer.save_model_name)
if not args.only_predict:
    print("### Loading Model ###")
    model = AutoModelForSequenceClassification.from_pretrained(
        train_config.model.pretrained_model_name
    )
else:
    print("### Loading Model From PreTrained ###")
    model = AutoModelForSequenceClassification.from_pretrained(
        train_config.trainer.save_model_name
    )

print("### Loading Trainer ###")
trainer = Trainer(
    model,
    train_args,
    train_dataset.custom_collate_fn,
    train_dataset,
    test_dataset,
    tokenizer,
)
if not args.only_predict:
    print("### Training ###")
    trainer.train()
    trainer.save_model(train_config.trainer.save_model_name)

# Predict
if not args.load_predictions:
    print("### Predicting ###")
    raw_predictions = trainer.predict(test_dataset)  # has predictions,label_ids,
    with open(train_config.misc.raw_predictions_file, "wb") as f:
        pkl.dump(raw_predictions, f)
else:
    print("### Loading Predictions ###")
    with open(train_config.misc.raw_predictions_file, "rb") as f:
        raw_predictions = pkl.load(f)

final_predictions = torch.argmax(raw_predictions.predictions, dim=-1).numpy()
with open(train_config.misc.final_predictions_file, "wb") as f:
    pkl.dump(final_predictions, f)

references = [ex["label"] for ex in test_dataset]


print("### Saving Metrics ###")
with open(train_config.misc.acc_metric_file, "w") as f:
    json.dump(accuracy_score(references, final_predictions), f)
with open(train_config.misc.f1_metric_file, "w") as f:
    json.dump(f1_score(references, final_predictions), f)
print("### Finished ###")
