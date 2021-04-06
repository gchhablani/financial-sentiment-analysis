import re

import pandas as pd
from tqdm.auto import tqdm

excel_file = "../data/word_sentiment_lists.xlsx"
data_file = "../data/clean_data.csv"


xls = pd.ExcelFile(excel_file)
negative_list = pd.read_excel(xls, "Negative", header=None).values
positive_list = pd.read_excel(xls, "Positive", header=None).values
uncertainty_list = pd.read_excel(xls, "Uncertainty", header=None).values
litigious_list = pd.read_excel(xls, "Litigious", header=None).values
strongmodal_list = pd.read_excel(xls, "StrongModal", header=None).values
weakmodal_list = pd.read_excel(xls, "WeakModal", header=None).values
contraining_list = pd.read_excel(xls, "Constraining", header=None).values

lists = [
    negative_list,
    positive_list,
    uncertainty_list,
    litigious_list,
    strongmodal_list,
    weakmodal_list,
    contraining_list,
]
data = pd.read_csv(data_file)
data = data.drop(["Cleaned"], axis=1)
data.fillna("", inplace=True)
data = data[
    data["If you won a lottery for Rs 1 crore tomorrow how would you spend it?"] != ""
]
columns = [
    "If you won a lottery for Rs 1 crore tomorrow how would you spend it?",
    "Unnamed: 3",
    "Unnamed: 4",
    "Unnamed: 5",
    "Unnamed: 6",
    "Unnamed: 7",
]
label_lists = [[], [], [], [], [], [], []]

texts = []
for value in tqdm(data[columns].values):
    text = "\n".join(value)
    texts.append(text.strip())
    for list_idx, lis in enumerate(lists):
        res = []
        for sub_str in lis:
            temp = [
                (" " + re.sub("[,.?;]", "", text.replace("\n", " ")) + " ")[
                    i.span()[0] + 1 : i.span()[1] - 1
                ]
                for i in re.finditer(
                    " " + sub_str[0].lower() + " ",
                    " " + re.sub("[,.?;]", "", text.lower().replace("\n", " ")) + " ",
                )
            ]
            if temp != []:
                res += temp
        label_lists[list_idx].append(res)

data["Text"] = texts
data["Negative"] = label_lists[0]
data["Positive"] = label_lists[1]
data["Uncertainty"] = label_lists[2]
data["Litigious"] = label_lists[3]
data["StrongModal"] = label_lists[4]
data["WeakModal"] = label_lists[5]
data["Constraining"] = label_lists[6]


data.to_csv(
    "sentiment_labeled_clean_data.csv",
    index=False,
    columns=[
        "Image File",
        "Text",
        "Negative",
        "Positive",
        "Uncertainty",
        "Litigious",
        "StrongModal",
        "WeakModal",
        "Constraining",
    ],
)
