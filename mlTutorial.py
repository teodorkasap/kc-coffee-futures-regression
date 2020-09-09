# %% - imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import talib
import talib.abstract as ta

# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - get data
df = pd.read_csv(
    'KCK21.NYB.csv')

# %% - drop nan
df = df.dropna()

# %% - get rid of rows with vol "0"
# df = df[df['Volume']!=0]

# %% - change date to datetime

df['Date'] = pd.to_datetime(
    df['Date'], format='%Y-%m-%d', errors='coerce')

# %% - set data column as index

df = df.set_index('Date')

# %% - plot closing price
df['Close'].plot(grid=True)
plt.title('KCK21 closing prices')
plt.ylabel('price $')
plt.show()


# %% - calculate Simple Moving Averages
def add_SMA(dataframe, colum_name,  period):
    dataframe['SMA_{}'.format(period)] = dataframe[colum_name].rolling(
        window=period).mean()


add_SMA(df, 'Close', 10)
add_SMA(df, 'Close', 20)
add_SMA(df, 'Close', 50)
add_SMA(df, 'Close', 100)
add_SMA(df, 'Close', 200)

# %% - calculate Exponential Moving Averages

def add_EMA(dataframe, colum_name,  period):
    dataframe['EMA_{}'.format(period)] = ta.EMA(dataframe, timeperiod=period, price=colum_name)

add_EMA(df, 'Close', 10)
add_EMA(df, 'Close', 20)
add_EMA(df, 'Close', 50)
add_EMA(df, 'Close', 100)
add_EMA(df, 'Close', 200)

# %% - calculate Average True Range

df['ATR'] = talib.ATR(df['High'].values,df['Low'].values,df['Close'].values,timeperiod=14)

# %% - calculate Average Directional Index
df['ADX'] = talib.ADX(df['High'].values,df['Low'].values,df['Close'].values,timeperiod=14)

df
