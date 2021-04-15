import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

task = "sentiment_analysis"
direc = f"results/{task}/baseline_neutral/preds"
if __name__ == "__main__":
    np.random.seed(42)
    # train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv(f"../data/{task}/test.csv")
    # X_train = train_df
    X_test = test_df
    # Y_train = train_df["label"]
    Y_test = test_df["label"]
    # (array([0, 1, 2]), array([21, 28, 13]))
    # preds = np.random.randint(low=0, high=3, size=Y_test.shape[0])
    preds = np.ones(Y_test.shape[0]) * 1
    if not os.path.exists(direc):
        os.makedirs(direc)
    with open(f"{direc}/acc.txt", "w") as f:
        f.write(str(accuracy_score(Y_test, preds)))
    with open(f"{direc}/f1_weighted.txt", "w") as f:
        f.write(str(f1_score(Y_test, preds, average="weighted")))
    with open(f"{direc}/f1_macro.txt", "w") as f:
        f.write(str(f1_score(Y_test, preds, average="macro")))
