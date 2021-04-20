# https://apoorvnandan.github.io/2020/08/15/bert-pretraining/
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from src.utils.mapper import configmapper


@configmapper.map("datasets", "pre_fin_text")
class PreFinText(Dataset):
    def __init__(
        self,
        file_path=None,
        data_frame=None,
        tokenizer_name="bert-base-uncased",
        **tokenizer_params,
    ):
        super().__init__()
        if data_frame is None and file_path is None:
            raise ValueError("One of `file_path` or `data_frame` must be provided.")
        elif data_frame is None:
            self.df = pd.read_csv(file_path)
        else:
            self.df = data_frame

        self.lines = list(self.df["text"].values)
        self.tokenizer_params = tokenizer_params
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.ids = self.encode_lines(self.lines)

    def encode_lines(self, lines):
        batch_encoding = self.tokenizer(lines, **self.tokenizer_params)
        return batch_encoding["input_ids"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)
