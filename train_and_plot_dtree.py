# https://chrisfotache.medium.com/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0
# https://www.kaggle.com/gilbar/xgboost-learning-curve
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.utils.misc import tokenize


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.field]


STOPWORDS = stopwords.words("english")


def run_dtree(max_depth, X_train, Y_train, X_test, Y_test):
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
                                    # (
                                    #     "svd",
                                    #     TruncatedSVD(
                                    #         algorithm="randomized", n_components=300
                                    #     ),
                                    # ),  # for XGB
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                ),
            ),
        ]
    )

    classifier.fit(X_train, Y_train)
    preds = classifier.predict(X_test)

    return classifier, f1_score(Y_test, preds, average="macro")


if __name__ == "__main__":
    task = "risk_profiling"
    model_path = f"results/{task}/dtree/models"
    direc = f"results/{task}/dtree/preds"
    np.random.seed(0)
    train_df = pd.read_csv(f"../data/{task}/train.csv")
    test_df = pd.read_csv(f"../data/{task}/test.csv")
    X_train = train_df
    X_test = test_df
    Y_train = train_df["label"]
    Y_test = test_df["label"]

    max_depths = [2, 3, 5, 10, 15, None]
    best_score = 0
    best_classifier = None
    best_combo = [None]
    c = 0
    for max_depth in max_depths:
        c += 1

        classifier, f1_macro = run_dtree(
            max_depth,
            X_train,
            Y_train,
            X_test,
            Y_test,
        )
        if max_depth == 5:
            plot_tree(
                classifier["clf"],
                feature_names=classifier["features"]
                .transformer_list[0][1]["tfidf"]
                .get_feature_names(),
                class_names=["Positive", "Negative", "Neutral"],
                filled=True,
                rounded=True,
            )
            plt.savefig(direc + "/depth5_dtree.pdf", bbox_inches='tight')
            plt.savefig(direc + "/depth5_dtree.eps", bbox_inches='tight')
            plt.show()
        # print(classifier["features"])
        print(f"{c}/{len(max_depths)}")
        if f1_macro > best_score:
            print("Updating")
            print(best_combo)
            print(f1_macro)
            best_score = f1_macro
            best_classifier = classifier
            best_combo = [max_depth]

    classifier = best_classifier
    train_preds = classifier.predict(X_train)
    preds = classifier.predict(X_test)
    print(best_combo)
    if not os.path.exists(direc + "/train"):
        os.makedirs(direc + "/train")

    if not os.path.exists(direc + "/test"):
        os.makedirs(direc + "/test")

    plot_tree(
        classifier["clf"],
        feature_names=classifier["features"]
        .transformer_list[0][1]["tfidf"]
        .get_feature_names(),
        class_names=["Positive", "Negative", "Neutral"],
        filled=True,
        rounded=True,
    )
    plt.savefig(direc + "/best_dtree.pdf", bbox_inches='tight')
    plt.savefig(direc + "/best_dtree.eps", bbox_inches='tight')
    plt.show()

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
