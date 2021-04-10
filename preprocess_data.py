import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

df = pd.read_csv("../data/text_label_data.csv")

df = df[["text", "label"]]
print(df.shape)
df = df[~(df["label"] == "FALSE")]
df.dropna(inplace=True)
print(df.shape)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify = df["label"])

# Column Renaming

# NaN Handling

# Normalizing/Scaling

# Label Encoding

train.to_csv("../train.csv", index=False)
test.to_csv("../test.csv", index=False)
