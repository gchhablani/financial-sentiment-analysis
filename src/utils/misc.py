import random
import re
import string
from collections import Counter

import numpy as np
import spacy
import torch

tok = spacy.load("en_core_web_sm")


def seed(value=42):
    """Set random seed for everything.

    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)


def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile(
        "[" + re.escape(string.punctuation) + "0-9\\r\\t\\n]"
    )  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]
