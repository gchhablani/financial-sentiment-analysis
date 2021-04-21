# https://chrisfotache.medium.com/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0
# https://www.kaggle.com/gilbar/xgboost-learning-curve
import os
import pickle as pkl

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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


def run_xgb(max_depth, n_estimators, learning_rate, X_train, Y_train, X_test, Y_test):
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
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    use_label_encoder=False,
                ),
            ),
        ]
    )

    classifier.fit(X_train, Y_train)
    preds = classifier.predict(X_test)

    return classifier, f1_score(Y_test, preds, average="macro")


if __name__ == "__main__":

    task = "sentiment_analysis"
    model_path = f"results/{task}/xgboost/models"
    direc = f"results/{task}/xgboost/preds"
    np.random.seed(42)
    train_df = pd.read_csv(f"../data/{task}/train.csv")
    test_df = pd.read_csv(f"../data/{task}/test.csv")
    X_train = train_df
    X_test = test_df
    Y_train = train_df["label"]
    Y_test = test_df["label"]

    max_depths = [2, 3, 5, 10, 15, None]
    n_estimatorss = [20, 50, 100, 200, 500, 1000]
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    best_score = 0
    best_classifier = None
    best_combo = [None, None, None]
    c = 0
    for max_depth in max_depths:
        for n_estimators in n_estimatorss:
            for learning_rate in learning_rates:
                c += 1
                classifier, f1_macro = run_xgb(
                    max_depth,
                    n_estimators,
                    learning_rate,
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                )
                print(f"{c}/{len(max_depths)*len(n_estimatorss)*len(learning_rates)}")
                if f1_macro > best_score:
                    print("Updating")
                    print(best_combo)
                    print(f1_macro)
                    best_score = f1_macro
                    best_classifier = classifier
                    best_combo = [max_depth, n_estimators, learning_rate]

    classifier = best_classifier
    train_preds = classifier.predict(X_train)
    preds = classifier.predict(X_test)
    print(best_combo)
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
