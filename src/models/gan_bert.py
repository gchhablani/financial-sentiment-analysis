# https://raw.githubusercontent.com/crux82/ganbert-pytorch/main/GANBERT_pytorch.ipynb
import torch
from torch.nn import Dropout, LeakyReLU, Linear, Module, Sequential
from transformers import AutoModel


class Generator(Module):
    def __init__(
        self, noise_size=100, output_size=768, hidden_sizes=[768], dropout_rate=0.1
    ):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    LeakyReLU(0.2, inplace=True),
                    Dropout(dropout_rate),
                ]
            )

        layers.append(Linear(hidden_sizes[-1], output_size))
        self.layers = Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


# ------------------------------
#   The Discriminator
#   https://www.aclweb.org/anthology/2020.acl-main.191/
#   https://github.com/crux82/ganbert
# ------------------------------
class Discriminator(Module):
    def __init__(
        self, input_size=768, hidden_sizes=[768], num_labels=3, dropout_rate=0.1
    ):
        super(Discriminator, self).__init__()
        self.input_dropout = Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [
                    Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    LeakyReLU(0.2, inplace=True),
                    Dropout(dropout_rate),
                ]
            )

        self.layers = Sequential(*layers)  # per il flatten
        self.logit = Linear(
            hidden_sizes[-1], num_labels + 1
        )  # +1 for the probability of this sample being fake/real.

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        return logits, last_rep


class GANBert(Module):
    def __init__(self, model_name):
        self.transformer = AutoModel.from_pretrained(model_name)
        pass

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        model_out = self.transformer(input_ids, attention_mask, token_type_ids)
        model_out = model_out[1]  # pooler out
        noise = torch.zeros(input_ids.shape[0], self.noise_size).uniform(0, 1)
        gen_out = self.generator(noise)
        disc_input = torch.cat([hidden_states, gen_rep], dim=0)
