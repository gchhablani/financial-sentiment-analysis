import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel

from src.utils.mapper import configmapper


@configmapper.map("models", "bert_features")
class BertFeatures(BertPreTrainedModel):
    def __init__(self, config, num_features, num_labels):
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size + num_features, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(
        self,
        input_ids,
        features=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        linear_input = torch.cat(pooled_output, features, dim=1)
        logits = self.classifier(linear_input)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
