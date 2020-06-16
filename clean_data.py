# %%
import pandas as pd
import numpy as np
import os
import fnmatch

# %%
df = pd.read_csv("f_year2017.txt")
# %%
print(df.columns)
# %%
commodity_codes = df['CFTC_Commodity_Code'].unique()
# %%
markets = []
for code in commodity_codes:
    name = df[df['CFTC_Commodity_Code'] == code].iloc[0, 0]
    markets.append(name)
# %%
coffee_markets = []
for market in markets:
    if "coffee" in market.lower():
        coffee_markets.append(market)
print(coffee_markets)
# %%
df = df[df['Market_and_Exchange_Names'] == 'COFFEE C - ICE FUTURES U.S.']
df

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

# %% function to get all txt files' names and append to list, then return list


def getDataFiles():
    files_list = [f for f in os.listdir() if os.path.isfile(
        f) and fnmatch.fnmatch(f, 'f_year*.txt')]
    return files_list


# %%
files = getDataFiles()
df_all = data2Df(files)
# %%
df_all