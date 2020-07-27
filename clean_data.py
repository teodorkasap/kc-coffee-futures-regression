# %% - imports and other settings
import pandas as pd
import numpy as np
import os
import fnmatch
from datetime import datetime
import pprint

import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
# This is for multiple print statements per cell
InteractiveShell.ast_node_interactivity = "all"


# %% function to retrieve dataframe and aggregate
def data2Df(files, commodity):
    df_aggregated = pd.DataFrame()
    for file in files:
        df = pd.read_excel(file, converters={'As_of_Date_In_Form_YYMMDD': str})
        commodity_codes = df['CFTC_Commodity_Code'].unique()
        markets = []
        for code in commodity_codes:
            name = df[df['CFTC_Commodity_Code'] == code].iloc[0, 0]
            markets.append(name)
        commodity_markets = []
        for market in markets:
            if commodity in market.lower():
                commodity_markets.append(market)
        for commodity_market in commodity_markets:
            df = df[df['Market_and_Exchange_Names'] == commodity_market]
            df_aggregated = df_aggregated.append(df)
            print(df_aggregated)
    return df_aggregated

# %% function to get all txt files' names and append to list, then return list (current directory only, no subdirectories)


def getDataFiles():
    files_list = [f for f in os.listdir() if os.path.isfile(
        f) and fnmatch.fnmatch(f, 'CIT*.xls')]
    return files_list


# %% - get data from every file and append to a master dataframe by specifying the commodity
commodity = "coffee"
files = getDataFiles()
df_all = data2Df(files, commodity)
# %% - check data frame
df_all.shape
# %% - check data types
df_all.dtypes
# %% - convert column to datetime from object
# df_all['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df_all['Report_Date_as_YYYY-MM-DD'])
# %% - check data types
# df_all.dtypes
# %%
df_all['date'] = pd.to_datetime(
    df_all['As_of_Date_In_Form_YYMMDD'], format='%y%m%d', errors='coerce')
df_all = df_all.set_index('date')
df_all.drop(['Report_Date_as_YYYY_MM_DD'], axis=1, inplace=True)
df_all.drop(['Report_Date_as_MM_DD_YYYY'], axis=1, inplace=True)
# %% - sort according to new index and check
df_all = df_all.sort_index()
df_all.head()
# %% - add net position along with short / long
df_all['Net_Position'] = df_all['NComm_Positions_Short_All_NoCIT'] - \
    df_all['NComm_Positions_Long_All_NoCIT']
df_all.tail()
# %% - visualize
plt.rcParams['figure.figsize'] = (10, 8)   # Increases the Plot Size
df_all['NComm_Positions_Short_All_NoCIT'].plot(grid=True, color='blue')
df_all['NComm_Positions_Long_All_NoCIT'].plot(grid=True, color='orange')
df_all['Net_Position'].plot(grid=True, color='red')
plt.legend()
# %%
list_of_columns = df_all.columns.tolist()
pprint.pprint(list_of_columns)
# %% - plot new column
# Todo: change name of df_all to suite future use
# %%
df_all.head()

# %% - get coffee weekly price data
df_coffee_price = pd.read_csv('coffee-futures-hist-data-weekly.csv')

# %% - change to datetime coffe price data frame
df_coffee_price['Date'] = pd.to_datetime(df_coffee_price['Date'], format='%b %d, %Y')
df_coffee_price
# Todo: fix exclusion of pre-ICE era data in the commitments data!!
# Todo: reindex coffee price data using datetime column as index!
