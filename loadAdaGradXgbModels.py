# %% - imports
import pickle
from pickle import ADDITEMS
import pandas as pd
import talib
import talib.abstract as ta

# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# %% - calculate Simple Moving Averages
def add_SMA(dataframe, colum_name,  period, commodity):
    dataframe['{}_SMA_{}'.format(commodity, period)] = dataframe[colum_name].rolling(
        window=period).mean()


# %% - calculate Exponential Moving Averages


def add_EMA(dataframe, colum_name,  period, commodity):
    dataframe['{}_EMA_{}'.format(commodity, period)] = ta.EMA(
        dataframe, timeperiod=period, price=colum_name)

# %% get usd data


def getUsdBrlData(filepath: str):

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.columns
    columns = ['Date', 'USD_Close',	'USD_Open',
               'USD_High', 'USD_Low', 'USD_Change %']
    df.columns = columns
    df['USD_Change %'] = df['USD_Change %'].str.replace(
        '%', '').astype('float') / 100.0
    return df


# %% - declare files
file_usd = "USD_BRL Historical Data-25092020.csv"
file_kc = 'KCK21.NYB-25092020.csv'


# %% - method for preparing final df

def prepareFinalKcData(df_kc, df_exch):
    df_kc = df_kc.dropna()
    columns = ["Date", "KC_Open", "KC_High", "KC_Low",
               "KC_Close", "KC_Adj_Close", "KC_Volume"]
    df_kc.columns = columns
    df = df_kc.drop(['KC_Volume'], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df = pd.merge(left=df, right=df_exch, left_on='Date', right_on='Date')
    df = df.set_index('Date')

    add_SMA(df, 'KC_Close', 10, "KC")
    add_SMA(df, 'KC_Close', 20, "KC")
    add_SMA(df, 'KC_Close', 50, "KC")
    add_SMA(df, 'KC_Close', 100, "KC")
    add_SMA(df, 'KC_Close', 200, "KC")
    add_EMA(df, 'KC_Close', 10, "KC")
    add_EMA(df, 'KC_Close', 20, "KC")
    add_EMA(df, 'KC_Close', 50, "KC")
    add_EMA(df, 'KC_Close', 100, "KC")
    add_EMA(df, 'KC_Close', 200, "KC")
    add_SMA(df, 'USD_Close', 5, "USD")
    add_SMA(df, 'USD_Close', 10, "USD")
    add_SMA(df, 'USD_Close', 25, "USD")
    add_SMA(df, 'USD_Close', 50, "USD")
    add_SMA(df, 'USD_Close', 100, "USD")
    add_EMA(df, 'USD_Close', 5, "USD")
    add_EMA(df, 'USD_Close', 10, "USD")
    add_EMA(df, 'USD_Close', 25, "USD")
    add_EMA(df, 'USD_Close', 50, "USD")
    add_EMA(df, 'USD_Close', 100, "USD")

    df['KC_ATR_14'] = talib.ATR(df['KC_High'].values, df['KC_Low'].values,
                            df['KC_Close'].values, timeperiod=14)
    df['KC_ADX_14'] = talib.ADX(df['KC_High'].values, df['KC_Low'].values,
                                df['KC_Close'].values, timeperiod=14)
    df['KC_CCI_14'] = talib.CCI(df['KC_High'].values, df['KC_Low'].values,
                                df['KC_Close'].values, timeperiod=14)
    df['KC_ROC_10'] = talib.ROC(df['KC_Close'], timeperiod=10)
    df['KC_RSI_14'] = talib.RSI(df['KC_Close'], timeperiod=14)
    df['KC_Williams_%R_14'] = talib.ATR(df['KC_High'].values, df['KC_Low'].values,
                                        df['KC_Close'].values, timeperiod=14)
    df['KC_Slowd'] = talib.STOCH(df['KC_High'].values,
                                df['KC_Low'].values,
                                df['KC_Close'].values,
                                fastk_period=5,
                                slowk_period=3,
                                slowk_matype=0,
                                slowd_period=3,
                                slowd_matype=0)[1]
    df['USD_ATR_14'] = talib.ATR(df['USD_High'].values, df['USD_Low'].values,
                                df['USD_Close'].values, timeperiod=14)
    df['USD_ATR_10'] = talib.ATR(df['USD_High'].values, df['USD_Low'].values,
                                df['USD_Close'].values, timeperiod=10)
    df['USD_ADX_14'] = talib.ADX(df['USD_High'].values, df['USD_Low'].values,
                                df['USD_Close'].values, timeperiod=14)
    df['USD_ADX_10'] = talib.ADX(df['USD_High'].values, df['USD_Low'].values,
                                df['USD_Close'].values, timeperiod=10)
    df['USD_CCI_14'] = talib.CCI(df['USD_High'].values, df['USD_Low'].values,
                                df['USD_Close'].values, timeperiod=14)
    df['USD_CCI_10'] = talib.CCI(df['USD_High'].values, df['USD_Low'].values,
                                df['USD_Close'].values, timeperiod=10)
    df['USD_ROC_10'] = talib.ROC(df['USD_Close'], timeperiod=10)
    df['USD_ROC_5'] = talib.ROC(df['USD_Close'], timeperiod=5)
    df['USD_RSI_14'] = talib.RSI(df['USD_Close'], timeperiod=14)
    df['USD_RSI_7'] = talib.RSI(df['USD_Close'], timeperiod=7)
    df['USD_Williams_%R_14'] = talib.ATR(df['USD_High'].values, df['USD_Low'].values,
                                        df['USD_Close'].values, timeperiod=14)
    df['USD_Williams_%R_7'] = talib.ATR(df['USD_High'].values, df['USD_Low'].values,
                                        df['USD_Close'].values, timeperiod=7)

    df = df.dropna()
    return df


# %% - get final dataset

df_exch = getUsdBrlData(file_usd)
df_kc = pd.read_csv(file_kc)


df = prepareFinalKcData(df_kc, df_exch)
# %%
X_input = df.drop(['KC_Close','KC_Adj_Close'],axis=1)
X_input =X_input.tail(20)

# %% - get models
file_ada = "final_ada_model.sav"
file_grad = "final_grad_model.sav"
file_xgb = "final_xgb_model.sav"

ada_reg = pickle.load(open(file_ada, 'rb'))
grad_reg = pickle.load(open(file_grad, 'rb'))
xgb_reg = pickle.load(open(file_xgb, 'rb'))

# %% - predictions
ada_predictions = ada_reg.predict(X_input)

# %%
ada_predictions