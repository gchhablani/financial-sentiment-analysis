# https://chrisfotache.medium.com/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0
# https://www.kaggle.com/gilbar/xgboost-learning-curve
import os
import pickle as pkl

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

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
    task = "sentiment_analysis"
    model_path = f"results/{task}/randomforest/models"
    direc = f"results/{task}/randomforest/preds"
    np.random.seed(42)
    train_df = pd.read_csv(f"../data/{task}/train.csv")
    test_df = pd.read_csv(f"../data/{task}/test.csv")
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
                RandomForestClassifier(
                    max_depth=3,
                    n_estimators=100,
                ),
            ),
        ]
    )

    classifier.fit(X_train, Y_train)
    train_preds = classifier.predict(X_train)
    preds = classifier.predict(X_test)

    if not os.path.exists(direc + "/train"):
        os.makedirs(direc + "/train")

    if not os.path.exists(direc + "/test"):
        os.makedirs(direc + "/test")

    with open(model_path, "wb") as f:
        pkl.dump(classifier, f)

    with open(f"{direc}/train/acc.txt", "w") as f:
        f.write(str(accuracy_score(Y_train, train_preds)))
    with open(f"{direc}/train/f1_weighted.txt", "w") as f:
        f.write(str(f1_score(Y_train, train_preds, average="weighted")))
    with open(f"{direc}/train/f1_macro.txt", "w") as f:
        f.write(str(f1_score(Y_train, train_preds, average="macro")))
    with open(f"{direc}/train/confusion_matrix.txt", "w") as f:
        f.write(str(confusion_matrix(Y_train, train_preds)))

    with open(f"{direc}/test/acc.txt", "w") as f:
        f.write(str(accuracy_score(Y_test, preds)))
    with open(f"{direc}/test/f1_weighted.txt", "w") as f:
        f.write(str(f1_score(Y_test, preds, average="weighted")))
    with open(f"{direc}/test/f1_macro.txt", "w") as f:
        f.write(str(f1_score(Y_test, preds, average="macro")))
    with open(f"{direc}/test/confusion_matrix.txt", "w") as f:
        f.write(str(confusion_matrix(Y_test, preds)))
