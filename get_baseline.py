import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    np.random.seed(42)
    # train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    # X_train = train_df
    X_test = test_df
    # Y_train = train_df["label"]
    Y_test = test_df["label"]
    [20, 30, 10]
    [20, 20, 10] = [20, 30, 0]
    # (array([0, 1, 2]), array([21, 28, 13]))
    preds = np.random.randint(low=0, high=3, size=Y_test.shape[0])
    if not os.path.exists("results/baseline/preds"):
        os.makedirs("results/baseline/preds")
    with open("results/baseline/preds/acc.txt", "w") as f:
        f.write(str(accuracy_score(Y_test, preds)))
    with open("results/baseline/preds/f1.txt", "w") as f:
        f.write(str(f1_score(Y_test, preds, average="weighted")))
