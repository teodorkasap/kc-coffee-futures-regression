# %%
import pandas as pd
import numpy as np

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
df = df[df['Market_and_Exchange_Names']=='COFFEE C - ICE FUTURES U.S.']
df