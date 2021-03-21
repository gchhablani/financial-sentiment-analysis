import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class FinText(Dataset):
    def __init__(
        self,
        file_path=None,
        data_frame=None,
        tokenizer_name="bert-base-uncased",
        feature_path=None,
    ):

        if data_frame is None and file_path is None:
            raise ValueError("One of `file_path` or `data_frame` must be provided.")
        elif data_frame is None:
            self.df = pd.read_csv(file_path)
        else:
            self.df = data_frame

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

        if feature_path is not None:
            raise NotImplementedError(
                "No implementation if present for adding features.\
                     Please ensure `feature_path` is None."
            )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        text = self.df["text"].iloc[idx]
        tokenized_text = self.tokenizer(text)
        return tokenized_text

    def custom_collate_fn(self, batch):
        input_idss = []
        token_type_idss = []
        attention_masks = []

        max_len = 0

        for sample in batch:

            input_idss.append(sample["input_ids"])
            max_len = max(max_len, len(sample["input_ids"]))
            token_type_idss.append(sample["token_type_ids"])
            attention_masks.append(sample["attention_mask"])

        for i in range(len(input_idss)):

            input_idss[i] = input_idss[i] + [self.tokenizer.pad_token_id] * (
                max_len - len(input_idss[i])
            )
            attention_masks[i] = attention_masks[i] + [self.tokenizer.pad_token_id] * (
                max_len - len(attention_masks[i])
            )
            token_type_idss[i] = token_type_idss[i] + [self.tokenizer.pad_token_id] * (
                max_len - len(token_type_idss[i])
            )

        return {
            "input_ids": torch.tensor(input_idss),
            "attention_mask": torch.tensor(attention_masks),
            "token_type_ids": torch.tensor(token_type_idss),
        }
