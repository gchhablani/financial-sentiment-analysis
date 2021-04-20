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
from collections import Counter

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.utils.dummy_pt_objects import DataCollator

from src.datasets import *
from src.models import *
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

args = parser.parse_args()
train_config = OmegaConf.load(os.path.join(args.config_dir, "train.yaml"))
dataset_config = OmegaConf.load(os.path.join(args.config_dir, "dataset.yaml"))

seed(train_config.args.seed)

# Load datasets
print("### Loading Datasets ###")

train_dataset = configmapper.get("datasets", dataset_config.dataset_name)(
    **dataset_config.train
)

# Train

print("### Getting Training Args ###")
train_args = TrainingArguments(**train_config.args)

print("### Loading Tokenizer for Trainer ###")
tokenizer = AutoTokenizer.from_pretrained(
    train_config.trainer.pretrained_tokenizer_name
)

print("### Loading Model ###")
model = AutoModelForMaskedLM.from_pretrained(train_config.model.pretrained_model_name)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
print("### Loading Trainer ###")
trainer = Trainer(
    model,
    train_args,
    data_collator,
    train_dataset,
)

print("### (Pre)-Training ###")
trainer.train()
trainer.save_model(train_config.trainer.save_model_name)
