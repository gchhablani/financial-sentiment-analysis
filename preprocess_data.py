import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

df = pd.read_csv("../data.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42)

# Column Renaming

# NaN Handling

# Normalizing/Scaling

# Label Encoding

train.to_csv("../train.csv", index=False)
test.to_csv("../test.csv", index=False)
