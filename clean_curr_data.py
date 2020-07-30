# %% - imports
import pandas as pd
import numpy as np
import os
import fnmatch
from datetime import datetime
import pprint
import re

import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
# This is for multiple print statements per cell
InteractiveShell.ast_node_interactivity = "all"

# %% - get data


def getDataFiles():
    files_list = [f for f in os.listdir() if os.path.isfile(
        f) and fnmatch.fnmatch(f, '*historical-data-weekly*')]
    return files_list


# %%
files = getDataFiles()

# %% - turn files into dataset


def data2Df(files):
    dataframes = {}
    for dataset in files:
        name = str(dataset).lower()
        if "historical" in name:
            name = name.replace("historical", "")
        name_split = re.split(r"[ :\-]+", name)
        if "futures" in name:
            df_name = ""
            pos = name_split.index("futures")
            for i in range(pos+1):
                if i == pos:
                    df_name = df_name+name_split[i]
                else:
                    df_name = df_name+name_split[i]+"_"
        else:
            df_name = name_split[0]+"_"+name_split[1]
        print("creating df: "+df_name)
        dataframes.update({df_name: pd.read_csv(dataset)})

    return dataframes


# %% - get all data frames into a dictionary
dataframes = data2Df(files)
# %% - check start and finish dates of each dataframe
for k,v in dataframes.items():
    print(k,v)