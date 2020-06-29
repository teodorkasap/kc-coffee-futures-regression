# %% - imports and other settings
import pandas as pd
import numpy as np
import os
import fnmatch
from datetime import datetime
import pprint

import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" #This is for multiple print statements per cell


# %% function to retrieve dataframe and aggregate
def data2Df(files):
    df_aggregated = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        commodity_codes = df['CFTC_Commodity_Code'].unique()
        markets = []
        for code in commodity_codes:
            name = df[df['CFTC_Commodity_Code'] == code].iloc[0, 0]
            markets.append(name)
        coffee_markets = []
        for market in markets:
            if "coffee" in market.lower():
                coffee_markets.append(market)
        for coffee_market in coffee_markets:
            df = df[df['Market_and_Exchange_Names'] == coffee_market]
            df_aggregated = df_aggregated.append(df)
            print(df_aggregated)
    return df_aggregated

# %% function to get all txt files' names and append to list, then return list (current directory only, no subdirectories)
def getDataFiles():
    files_list = [f for f in os.listdir() if os.path.isfile(
        f) and fnmatch.fnmatch(f, 'f_year*.txt')]
    return files_list


# %% - get data from every file and append to a master dataframe
files = getDataFiles()
df_all = data2Df(files)
# %% - check data frame
df_all.shape
# %% - check data types
df_all.dtypes
# %% - convert column to datetime from object
df_all['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df_all['Report_Date_as_YYYY-MM-DD'])
# %% - check data types
df_all.dtypes
# %%
df_all['date'] = pd.to_datetime(df_all['Report_Date_as_YYYY-MM-DD'])
df_all = df_all.set_index('date')
df_all.drop(['Report_Date_as_YYYY-MM-DD'], axis=1, inplace=True)
# %% - sort according to new index and check
df_all = df_all.sort_index()
df_all.head()
# %% - add net position along with short / long
df_all['Net_Position']=df_all['Prod_Merc_Positions_Short_All']-df_all['Prod_Merc_Positions_Long_All']
df_all.tail()
# %% - visualize
plt.rcParams['figure.figsize'] = (10, 8)   # Increases the Plot Size
df_all['Prod_Merc_Positions_Short_All'].plot(grid = True,color='blue')
df_all['Prod_Merc_Positions_Long_All'].plot(grid = True,color='orange')
df_all['Net_Position'].plot(grid = True,color='red')
plt.legend()
# %%
list_of_columns = df_all.columns.tolist()
pprint.pprint(list_of_columns)
# %% - plot new column
