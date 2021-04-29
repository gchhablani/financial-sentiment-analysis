import os

import numpy as np
import pandas as pd

model_name_mapping = {
    "xgboost": "XGBoost",
    "lstm-model": "LSTM",
    "bert-base": "BERT",
    "randomforest": "RandomForest",
    "pretrain-bert-base": "BERT-PT",
    "finbert-base": "FinBERT",
    "pretrain-bert-base-train": "BERT-PT-TR",
    "distilbert-base": "DistilBERT",
    "baseline_neutral": "Baseline (Neu)",
    "baseline_random": "Baseline (Rndm)",
}
name_to_info_dict = {}
for root, dirs, files in os.walk("./results/risk_profiling/"):
    for fil in files:
        if "f1_macro.txt" in fil:
            if "test" in root:
                with open(os.path.join(root, fil)) as f:
                    score = float(f.read())
                task = (
                    root.replace("./results/risk_profiling/", "")
                    .replace("/preds", "")
                    .replace("/train", "")
                    .replace("/test", "")
                    .replace("f1_macro.txt", "")
                )
                if task not in name_to_info_dict:
                    name_to_info_dict[task] = {"test": score}
                else:
                    name_to_info_dict[task]["test"] = score
            elif "baseline" in fil:
                with open(os.path.join(root, fil)) as f:
                    score = float(f.read())
                task = (
                    root.replace("./results/risk_profiling/", "")
                    .replace("/preds", "")
                    .replace("/train", "")
                    .replace("/test", "")
                    .replace("f1_macro.txt", "")
                )
                if task not in name_to_info_dict:
                    name_to_info_dict[task] = {"test": score, "train": np.nan}
            else:
                with open(os.path.join(root, fil)) as f:
                    score = float(f.read())
                task = (
                    root.replace("./results/risk_profiling/", "")
                    .replace("/preds", "")
                    .replace("/train", "")
                    .replace("/test", "")
                    .replace("f1_macro.txt", "")
                )
                if task not in name_to_info_dict:
                    name_to_info_dict[task] = {"train": score}
                else:
                    name_to_info_dict[task]["train"] = score

df = pd.DataFrame(name_to_info_dict)
df = df.T
df = df.reset_index()
df.rename({"index": "Model", "test": "Test", "train": "Train"}, inplace=1, axis=1)
df["Model"] = df["Model"].apply(model_name_mapping.get)
df["Test"] = df["Test"].round(3)
df["Train"] = df["Train"].round(3)
df.to_csv("risk_profiling.csv", columns=["Model", "Train", "Test"], index=False)

name_to_info_dict = {}
for root, dirs, files in os.walk("./results/sentiment_analysis/"):
    for fil in files:
        if "f1_macro.txt" in fil:
            if "test" in root:
                with open(os.path.join(root, fil)) as f:
                    score = float(f.read())
                task = (
                    root.replace("./results/sentiment_analysis/", "")
                    .replace("/preds", "")
                    .replace("/train", "")
                    .replace("/test", "")
                    .replace("f1_macro.txt", "")
                )
                if task not in name_to_info_dict:
                    name_to_info_dict[task] = {"test": score}
                else:
                    name_to_info_dict[task]["test"] = score
            elif "baseline" in fil:
                with open(os.path.join(root, fil)) as f:
                    score = float(f.read())
                task = (
                    root.replace("./results/sentiment_analysis/", "")
                    .replace("/preds", "")
                    .replace("/train", "")
                    .replace("/test", "")
                    .replace("f1_macro.txt", "")
                )
                if task not in name_to_info_dict:
                    name_to_info_dict[task] = {"test": score, "train": np.nan}
            else:
                with open(os.path.join(root, fil)) as f:
                    score = float(f.read())
                task = (
                    root.replace("./results/sentiment_analysis/", "")
                    .replace("/preds", "")
                    .replace("/train", "")
                    .replace("/test", "")
                    .replace("f1_macro.txt", "")
                )
                if task not in name_to_info_dict:
                    name_to_info_dict[task] = {"train": score}
                else:
                    name_to_info_dict[task]["train"] = score

df = pd.DataFrame(name_to_info_dict)
df = df.T
df = df.reset_index()
df.rename({"index": "Model", "test": "Test", "train": "Train"}, inplace=1, axis=1)
df["Model"] = df["Model"].apply(model_name_mapping.get)
df["Test"] = df["Test"].round(3)
df["Train"] = df["Train"].round(3)
df.to_csv("sentiment_analysis.csv", columns=["Model", "Train", "Test"], index=False)
