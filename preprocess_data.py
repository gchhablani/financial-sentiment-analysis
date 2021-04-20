import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# TO-DO: Column Renaming, NaN Handling, Normalizing/Scaling, Label Encoding


def convert_columns_to_labels(x):
    neg_words = eval(x[1]["Negative"])
    pos_words = eval(x[1]["Positive"])
    # POS
    if neg_words == [] and pos_words != []:
        return 0
    # NEG
    elif neg_words != [] and pos_words == []:
        return 1
    # NEU
    else:
        return 2


np.random.seed(42)

# Sentiment Data
print("# Sentiment Analysis")
df = pd.read_csv("../data/sentiment_labeled_clean_data.csv")
labels = []
for row in df.iterrows():
    labels.append(convert_columns_to_labels(row))

df["label"] = labels
df = df[["text", "label"]]
df.dropna(inplace=True)
print(df.shape)
print(np.unique(df["label"], return_counts=True))
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train.to_csv("../data/sentiment_analysis/train.csv", index=False)
test.to_csv("../data/sentiment_analysis/test.csv", index=False)


print("# Risk Profiling")
# Risk-Profile Data
df = pd.read_csv("../data/clean_data.csv")
df = df[["text", "label"]]
df.dropna(inplace=True)
df = df[~(df["label"] == "FALSE")]
df["label"] = df["label"].apply(int)
print(df.shape)

print(np.unique(df["label"], return_counts=True))
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train.to_csv("../data/risk_profiling/train.csv", index=False)
test.to_csv("../data/risk_profiling/test.csv", index=False)

# Pre-training
print("# Pre-training")
df = pd.read_csv("../data/clean_data.csv")
df = df[["text"]]
df.dropna(inplace=True)
print(df.shape)
df.to_csv("../data/pretraining/data.csv", index=False)
