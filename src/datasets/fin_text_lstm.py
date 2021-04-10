import string
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset

from src.utils.mapper import configmapper
from src.utils.misc import tokenize

# from nltk.corpus import stopwords
# STOPWORDS = stopwords.words('english')

# https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
# tokenization


@configmapper.map("datasets", "fin_text_lstm")
class FinTextLstm(Dataset):
    def __init__(
        self,
        file_path=None,
        data_frame=None,
        vocab2index=None,
        words=None,
    ):
        super().__init__()
        if data_frame is None and file_path is None:
            raise ValueError("One of `file_path` or `data_frame` must be provided.")
        elif data_frame is None:
            self.df = pd.read_csv(file_path)
        else:
            self.df = data_frame

        if vocab2index is None:
            counts = Counter()
            for index, row in self.df.iterrows():
                counts.update(tokenize(row["text"]))
            self.vocab2index = {"": 0, "UNK": 1}
            self.words = ["", "UNK"]
            for word in counts:
                self.vocab2index[word] = len(self.words)
                self.words.append(word)
        else:
            self.vocab2index = vocab2index
            if words is None:
                raise ValueError(
                    "`words` cannot be of `NoneType` when `vocab2index` is not of `NoneType`."
                )
            self.words = words

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]
        label = self.df.iloc[index]["label"]
        encoding, length = self.encode_sentence(text)
        return encoding, length, label

    def custom_collate_fn(self, batch):
        # ids = []
        inputs = []
        lengths = []
        labels = []

        max_len = 0

        for encoding, length, label in batch:
            # ids.append(idx)
            inputs.append(encoding)
            lengths.append(length)
            max_len = max(max_len, len(encoding))
            labels.append(label)

        for i in range(len(inputs)):

            inputs[i] = inputs[i] + [0] * (max_len - len(inputs[i]))

        return {
            "inputs": torch.tensor(inputs),
            "lengths": torch.tensor(lengths),
            "labels": torch.tensor(labels),
        }

    def encode_sentence(self, text):
        tokenized = tokenize(text)
        encoding = [
            self.vocab2index.get(word, self.vocab2index["UNK"]) for word in tokenized
        ]
        length = len(encoding)
        return encoding, length
