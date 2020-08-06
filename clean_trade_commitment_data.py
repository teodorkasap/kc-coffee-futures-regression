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
df_all_commit = data2Df(files, commodity)
# %% - check data frame
df_all_commit.shape
# %% - check data types
df_all_commit.dtypes
# %% - convert column to datetime from object
# df_all_commit['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df_all_commit['Report_Date_as_YYYY-MM-DD'])
# %% - check data types
# df_all_commit.dtypes
# %%
df_all_commit['date'] = pd.to_datetime(
    df_all_commit['As_of_Date_In_Form_YYMMDD'], format='%y%m%d', errors='coerce')
df_all_commit = df_all_commit.set_index('date')
df_all_commit.drop(['Report_Date_as_YYYY_MM_DD'], axis=1, inplace=True)
df_all_commit.drop(['Report_Date_as_MM_DD_YYYY'], axis=1, inplace=True)
# %% - sort according to new index and check
df_all_commit = df_all_commit.sort_index()
df_all_commit.head()
# %% - add net position along with short / long
df_all_commit['Net_Position'] = df_all_commit['NComm_Positions_Short_All_NoCIT'] - \
    df_all_commit['NComm_Positions_Long_All_NoCIT']
df_all_commit.tail()
# %% - visualize
plt.rcParams['figure.figsize'] = (10, 8)   # Increases the Plot Size
df_all_commit['NComm_Positions_Short_All_NoCIT'].plot(grid=True, color='blue')
df_all_commit['NComm_Positions_Long_All_NoCIT'].plot(grid=True, color='orange')
df_all_commit['Net_Position'].plot(grid=True, color='red')
plt.legend()
# %%
list_of_columns = df_all_commit.columns.tolist()
pprint.pprint(list_of_columns)

# %% - shift dates back two days to get same date as other data
df_all_commit['Prev Sunday'] = df_all_commit.index.shift(-2, freq='d')

# %% - plot new column
# Todo: change name of df_all_commit_commit to suite future use
# %%
df_all_commit.head()
df_all_commit.tail()


def convert_to_float(value):
    text = str(value)
    if "K" in text:
        text = text.strip()
        text = text.replace("K", "")
        try:
            number = float(text)
            number = number*1000
        except ValueError as e:
            print("encountered an error, leaving cell empty")
            number=0
    else:
        text = text.strip()
        text = text.replace("%", "")
        try:
            number = float(text)
            number = number/100
        except ValueError as e:
            print("encountered an error, leaving cell empty")
            number=0

    return number


# %% - get coffee weekly price data
df_coffee_price = pd.read_csv('coffee-futures-hist-data-weekly.csv')

# %% - change to datetime coffe price data frame
df_coffee_price['Date'] = pd.to_datetime(
    df_coffee_price['Date'], format='%b %d, %Y')
df_coffee_price = df_coffee_price.drop(
    ['Price', 'Open', 'High', 'Low'], axis=1)
columns = ['Coffee Date', 'Coffee Vol.', 'Coffee Price Change%']
df_coffee_price.columns = columns
df_coffee_price

# %% - merge coffee commitments and coffee price
df_coffee = df_all_commit.merge(
    df_coffee_price, left_on='Prev Sunday', right_on='Coffee Date')
df_coffee = df_coffee[['Coffee Date', 'Net_Position',
                       'Coffee Vol.', 'Coffee Price Change%']]
# df_coffee['Coffee Price Change%'] = df_coffee['Coffee Price Change%'].astype(float)
df_coffee['Coffee Price Change% shifted'] = df_coffee['Coffee Price Change%'].shift(
    periods=1)
df_coffee['Coffee Vol. shifted'] = df_coffee['Coffee Vol.'].shift(
    periods=1)
df_coffee['Coffee Price Change% shifted'] = df_coffee['Coffee Price Change% shifted'].apply(
    lambda x: convert_to_float(x))
df_coffee['Coffee Vol. shifted'] = df_coffee['Coffee Vol. shifted'].apply(
    lambda x: convert_to_float(x))

df_coffee = df_coffee.dropna(axis=0)

df_coffee.dtypes
df_coffee.describe()


# Todo: fix exclusion of pre-ICE era data in the commitments data!!
# Todo: reindex coffee price data using datetime column as index!

# %% - plot vol against price change
plt.scatter(df_coffee['Coffee Vol. shifted'],
            df_coffee['Coffee Price Change% shifted'])
plt.axhline(0)
plt.axvline(0)
plt.show()

# %% - scatter plot
plt.scatter(df_coffee['Net_Position'],
            df_coffee['Coffee Price Change% shifted'])
plt.axhline(0)
plt.axvline(0)
plt.show()

# %% - Run LSTM Model
