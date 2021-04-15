import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# TO-DO: Column Renaming, NaN Handling, Normalizing/Scaling, Label Encoding


def convert_columns_to_labels(x):
    neg_words = eval(x[1]["Negative"])
    pos_words = eval(x[1]["Positive"])
    # POS
    if neg_words == [] and pos_words != []:
        return 2
    # NEG
    elif neg_words != [] and pos_words == []:
        return 0
    # NEU
    else:
        return 1


np.random.seed(42)

# Sentiment Data
df = pd.read_csv("../data/sentiment_labeled_clean_data.csv")
labels = []
for row in df.iterrows():
    labels.append(convert_columns_to_labels(row))

df["label"] = labels
df = df[["text", "label"]]
df.dropna(inplace=True)
print(df.shape)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train.to_csv("../sentiment_analysis/train.csv", index=False)
test.to_csv("../sentiment_analysis/test.csv", index=False)
