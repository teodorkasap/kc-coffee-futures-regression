# %% - imports and other settings
import pandas as pd
import numpy as np
import os
import fnmatch
from datetime import datetime
import pprint

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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


# %% - make a function to process dataframe for traders commitments

def process_commitment_data(df, commodity):
    df['date'] = pd.to_datetime(
        df['As_of_Date_In_Form_YYMMDD'], format='%y%m%d', errors='coerce')
    df.drop(['Report_Date_as_YYYY_MM_DD'], axis=1, inplace=True)
    df.drop(['Report_Date_as_MM_DD_YYYY'], axis=1, inplace=True)
    df['{} comm. Net_Position'.format(commodity)] = df['NComm_Positions_Short_All_NoCIT'] - \
        df['NComm_Positions_Long_All_NoCIT']
    print(df)
    df = df[['date', '{} comm. Net_Position'.format(commodity)]]
    print(df)
    df = df.set_index('date')
    df['Prev Sunday'] = df.index.shift(-2, freq='d')
    df = df.sort_index()
    return df


# %%
# df_all_commit['date'] = pd.to_datetime(
#     df_all_commit['As_of_Date_In_Form_YYMMDD'], format='%y%m%d', errors='coerce')
# df_all_commit = df_all_commit.set_index('date')
# df_all_commit.drop(['Report_Date_as_YYYY_MM_DD'], axis=1, inplace=True)
# df_all_commit.drop(['Report_Date_as_MM_DD_YYYY'], axis=1, inplace=True)
# # %% - sort according to new index and check
# df_all_commit = df_all_commit.sort_index()
# df_all_commit.head()
# # %% - add net position along with short / long
# df_all_commit['Net_Position'] = df_all_commit['NComm_Positions_Short_All_NoCIT'] - \
#     df_all_commit['NComm_Positions_Long_All_NoCIT']
# df_all_commit.tail()

# %% - get commitment data

commodity1 = "coffee"
commodity2 = "sugar"
commodity3 = "cocoa"
files = getDataFiles()
df_coffee_comm = data2Df(files, commodity1)
df_coffee_comm = process_commitment_data(df_coffee_comm, commodity1)
df_sugar_comm = data2Df(files, commodity2)
df_sugar_comm = process_commitment_data(df_sugar_comm, commodity2)
df_cocoa_comm = data2Df(files, commodity3)
df_cocoa_comm = process_commitment_data(df_cocoa_comm, commodity3)

df_coffee_comm
df_cocoa_comm
df_sugar_comm

# %% - merge dataframes
merged_inner = pd.merge(left=df_coffee_comm, right=df_cocoa_comm,
                        left_on='Prev Sunday', right_on='Prev Sunday')
merged_inner = pd.merge(left=merged_inner, right=df_sugar_comm,
                        left_on='Prev Sunday', right_on='Prev Sunday')


# %% - visualize
plt.rcParams['figure.figsize'] = (10, 8)   # Increases the Plot Size
df_cocoa_comm['cocoa comm. Net_Position'].plot(grid=True, color='blue')
df_coffee_comm['coffee comm. Net_Position'].plot(grid=True, color='orange')
df_sugar_comm['sugar comm. Net_Position'].plot(grid=True, color='red')
plt.legend()

# %% - shift dates back two days to get same date as other data
# df_all_commit['Prev Sunday'] = df_all_commit.index.shift(-2, freq='d')

# %% - plot new column
# Todo: change name of df_all_commit_commit to suite future use


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
            number = 0
    else:
        text = text.strip()
        text = text.replace("%", "")
        try:
            number = float(text)
            number = number/100
        except ValueError as e:
            print("encountered an error, leaving cell empty")
            number = 0

    return number


# %% - get coffee weekly price data
df_coffee_price = pd.read_csv('coffee-futures-hist-data-weekly.csv')

# %% - change to datetime coffe price data frame
df_coffee_price['Date'] = pd.to_datetime(
    df_coffee_price['Date'], format='%b %d, %Y')
df_coffee_price = df_coffee_price.drop(
    ['Open', 'High', 'Low'], axis=1)
df_coffee_price
# %%
columns = ['Coffee Date','Coffee Price', 'Coffee Vol.', 'Coffee Price Change%']
df_coffee_price.columns = columns
df_coffee_price

# %% - merge coffee commitments and coffee price
df_master = pd.merge(left=merged_inner, right=df_coffee_price,
                     left_on='Prev Sunday', right_on='Coffee Date')
df_master = df_master[['Coffee Date', 'Coffee Price', 'coffee comm. Net_Position', 'cocoa comm. Net_Position', 'sugar comm. Net_Position',
                       'Coffee Vol.', 'Coffee Price Change%']]


# %%
df_master = df_master.set_index('Coffee Date')
# %% - make new columns with shifted coffee price
# df_coffee['Coffee Price Change%'] = df_coffee['Coffee Price Change%'].astype(float)
df_master['Coffee Price Change% shifted'] = df_master['Coffee Price Change%'].shift(
    periods=1)
df_master['Coffee Price shifted'] = df_master['Coffee Price'].shift(
    periods=1)
# df_master['Coffee Vol. shifted'] = df_master['Coffee Vol.'].shift(
#     periods=1)

# %%
df_master

# %% - change type to float
df_master['Coffee Price Change% shifted'] = df_master['Coffee Price Change% shifted'].apply(
    lambda x: convert_to_float(x))

df_master['Coffee Price shifted'] = df_master['Coffee Price shifted'].apply(
    lambda x: convert_to_float(x)*100)

df_master['Coffee Price'] = df_master['Coffee Price'].apply(
    lambda x: convert_to_float(x)*100)

# df_master['Coffee Vol. shifted'] = df_master['Coffee Vol. shifted'].apply(
#     lambda x: convert_to_float(x))

df_master = df_master.dropna(axis=0)

df_master.dtypes
df_master.describe()
df_master

# %%

df_master.to_csv(r'coffee-cocoa-sugar-commitment-coffee-price-data-saved03082020.csv')


# %% - drop unused columns
df_master = df_master.drop(['Coffee Vol.', 'Coffee Price Change%'], axis=1)
df_master

# %% - train normalizer
values = df_master.values
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(values)
# %% - normalize
df_scaled = pd.DataFrame(x_scaled)

# %% inverse transform and print the first 5 rows
# inversed = scaler.inverse_transform(normalized)

# %% - standardized data frame
# df_standard = pd.DataFrame(inversed,index=df_master.index, columns=['coffee comm. Net_Position','cocoa comm. Net_Position','sugar comm. Net_Position','Coffee Price Change% shifted'])
# df_standard
df_scaled.columns = ['coffee comm. Net_Position', 'cocoa comm. Net_Position',
    'sugar comm. Net_Position', 'Coffee Price Change% shifted']
# %% - 
# Todo: get index of df_master to index df scaled
df_scaled.index = df_master.index

# %%
df_scaled

# %%
plt.rcParams['figure.figsize']=(10, 8)   # Increases the Plot Size
df_scaled['cocoa comm. Net_Position'].plot(grid = True, color = 'blue')
df_scaled['coffee comm. Net_Position'].plot(grid = True, color = 'orange')
df_scaled['sugar comm. Net_Position'].plot(grid = True, color = 'red')
df_scaled['Coffee Price Change% shifted'].plot(grid = True, color = 'pink')
plt.legend()



# Todo: fix exclusion of pre-ICE era data in the commitments data!!
# Todo: reindex coffee price data using datetime column as index!

# %% - plot vol against price change
# plt.scatter(df_coffee['Coffee Vol. shifted'],
#             df_coffee['Coffee Price Change% shifted'])
# plt.axhline(0)
# plt.axvline(0)
# plt.show()

# # %% - scatter plot
# plt.scatter(df_coffee['Net_Position'],
#             df_coffee['Coffee Price Change% shifted'])
# plt.axhline(0)
# plt.axvline(0)
# plt.show()

# %% - 
df_scaled.describe()
# %%
df_subset1 = df_scaled.iloc[100:150]
plt.rcParams['figure.figsize']=(10, 8)   # Increases the Plot Size
df_subset1['cocoa comm. Net_Position'].plot(grid = True, color = 'blue')
df_subset1['coffee comm. Net_Position'].plot(grid = True, color = 'orange')
df_subset1['sugar comm. Net_Position'].plot(grid = True, color = 'red')
df_subset1['Coffee Price Change% shifted'].plot(grid = True, color = 'pink')
plt.legend()

# %% - save df to 
df_scaled.to_csv(r'scaled-coffee-cocoa-sugar-commitment-coffee-price-data.csv')
df_master.to_csv(r'coffee-cocoa-sugar-commitment-coffee-price-data.csv')
