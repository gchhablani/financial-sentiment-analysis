# https://chrisfotache.medium.com/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0
import os
import pickle as pkl

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from xgboost import XGBClassifier

from src.utils.misc import tokenize


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.field]


STOPWORDS = stopwords.words("english")

if __name__ == "__main__":
    np.random.seed(42)
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    X_train = train_df
    X_test = test_df
    Y_train = train_df["label"]
    Y_test = test_df["label"]
    classifier = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text",
                            Pipeline(
                                [
                                    ("colext", TextSelector("text")),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            tokenizer=tokenize,
                                            stop_words=STOPWORDS,
                                            min_df=0.0025,
                                            max_df=0.25,
                                            ngram_range=(1, 3),
                                        ),
                                    ),
                                    (
                                        "svd",
                                        TruncatedSVD(
                                            algorithm="randomized", n_components=300
                                        ),
                                    ),  # for XGB
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    max_depth=None,
                    n_estimators=1000,
                    learning_rate=0.1,
                    use_label_encoder=False,
                ),
            ),
        ]
    )

    classifier.fit(X_train, Y_train)
    preds = classifier.predict(X_test)

    if not os.path.exists("results/xgboost/preds"):
        os.makedirs("results/xgboost/preds")

    with open("results/xgboost/model", "wb") as f:
        pkl.dump(classifier, f)

    with open("results/xgboost/preds/acc.txt", "w") as f:
        f.write(str(accuracy_score(Y_test, preds)))
    with open("results/xgboost/preds/f1.txt", "w") as f:
        f.write(str(f1_score(Y_test, preds, average="weighted")))
