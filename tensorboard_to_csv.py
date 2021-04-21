# https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

listOutput = glob.glob("./results/**/runs/*", recursive=True)
# print(listOutput)

listDF = []

for tb_output_folder in listOutput:
    # print(tb_output_folder)
    if "pretraining" in tb_output_folder:
        continue
    x = EventAccumulator(path=tb_output_folder)
    x.Reload()
    x.FirstEventTimestamp()
    keys = ["eval/f1", "eval/loss", "train/loss"]

    listValues = {}
    if x.Tags()["scalars"] == []:
        continue
    # else:
    #     print(x.Tags())
    steps = [e.step for e in x.Scalars(keys[0])]
    wall_time = [e.wall_time for e in x.Scalars(keys[0])]
    index = [e.index for e in x.Scalars(keys[0])]
    count = [e.count for e in x.Scalars(keys[0])]
    n_steps = len(steps)
    listRun = [tb_output_folder] * n_steps
    printOutDict = {}
    data = np.zeros((n_steps, len(keys)))
    for i in range(len(keys)):
        data[:, i] = [e.value for e in x.Scalars(keys[i])][-n_steps:]

    printOutDict = {
        keys[0]: data[:, 0],
        keys[1]: data[:, 1],
        keys[2]: data[:, 2],
    }

    printOutDict["Name"] = listRun

    DF = pd.DataFrame(data=printOutDict)

    listDF.append(DF)

df = pd.concat(listDF)
df.to_csv("Output.csv")
