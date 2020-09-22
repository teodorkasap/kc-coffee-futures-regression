# %% - source
# https://www.datatechnotes.com/2019/07/regression-example-with.html

# %% - imports
from prepUsdBrlData import getUsdBrlData
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import talib
import talib.abstract as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error


# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - predict usd prices


# %% - get usd / brl data
df_exch = getUsdBrlData()
columns_shift = ['USD_Close',	"USD_Open",
                 "USD_High",	"USD_Low",	"USD_Change %"]
# df_exch['USD_Close',	"USD_Open",	"USD_High",	"USD_Low",	"USD_Change %"] = df_exch[['USD_Close',	"USD_Open",	"USD_High",	"USD_Low",	"USD_Change %"]].shift(-1)

for column in columns_shift:
    df_exch[column] = df_exch[column].shift(-1)

df_exch

# %% - get KC data
df = pd.read_csv(
    'KCK21.NYB.csv')

# %% - drop nan
df = df.dropna()
columns_shift = []
for column in columns_shift:
    df[column] = df[column].shift(-1)

df

# %% - get rid of rows with vol "0"
df = df.drop(['KC_Volume'], axis=1)

# %% - change date to datetime

df['Date'] = pd.to_datetime(
    df['Date'], format='%Y-%m-%d', errors='coerce')


# %% - merge two dataframes on KC data dates
df = pd.merge(left=df, right=df_exch, left_on='Date', right_on='Date')


# %%- get exch rate prediction


# %% - set data column as index

df = df.set_index('Date')

# %% - plot closing price
df['KC_Close'].plot(grid=True)

plt.title('KCK21 closing prices')
plt.ylabel('price $')
plt.show()

df['USD_Close'].plot(grid=True)
plt.title('USD/BRL closing prices')
plt.ylabel('BRL')
plt.show()


# %% - calculate Simple Moving Averages
def add_SMA(dataframe, colum_name,  period, commodity):
    dataframe['{}_SMA_{}'.format(commodity, period)] = dataframe[colum_name].rolling(
        window=period).mean()


add_SMA(df, 'KC_Close', 10, "KC")
add_SMA(df, 'KC_Close', 20, "KC")
add_SMA(df, 'KC_Close', 50, "KC")
add_SMA(df, 'KC_Close', 100, "KC")
add_SMA(df, 'KC_Close', 200, "KC")


# %% - calculate Exponential Moving Averages


def add_EMA(dataframe, colum_name,  period, commodity):
    dataframe['{}_EMA_{}'.format(commodity, period)] = ta.EMA(
        dataframe, timeperiod=period, price=colum_name)


add_EMA(df, 'KC_Close', 10, "KC")
add_EMA(df, 'KC_Close', 20, "KC")
add_EMA(df, 'KC_Close', 50, "KC")
add_EMA(df, 'KC_Close', 100, "KC")
add_EMA(df, 'KC_Close', 200, "KC")


# %% - calculate Average True Range

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

df['USD_Slowk'], df['USD_Slowd'] = talib.STOCH(df['USD_High'].values,
                                               df['USD_Low'].values,
                                               df['USD_Close'].values,
                                               fastk_period=5,
                                               slowk_period=3,
                                               slowk_matype=0,
                                               slowd_period=3,
                                               slowd_matype=0)

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

# %%- get rid of nan

df = df.dropna()
df

# %%
df.columns

# %% shift kc related columns

columns_shift = ['KC_Open', 'KC_High', 'KC_Low', 'KC_SMA_10',
                 'KC_SMA_20', 'KC_SMA_50', 'KC_SMA_100', 'KC_SMA_200', 'KC_EMA_10',
                 'KC_EMA_20', 'KC_EMA_50', 'KC_EMA_100', 'KC_EMA_200', 'KC_ADX_14',
                 'KC_CCI_14', 'KC_Slowd', 'KC_ROC_10', 'KC_RSI_14', 'KC_Williams_%R_14', ]
for column in columns_shift:
    df[column] = df[column].shift(1)

# %% - get shape
df = df.dropna()
df

# %% -corr. matrix

corrMatrix = df.corr()
print(corrMatrix)

# %% - eliminate low corellation features

df = df[['KC_Close',"KC_Open", "KC_High", "KC_Low", "KC_SMA_10", "KC_SMA_20", "KC_EMA_10", "KC_EMA_20", "KC_EMA_50",
         "USD_ATR_14", "USD_ATR_10", "USD_Williams_%R_14", "KC_RSI_14"]]

# %% - get train and test sets

cutoff = int(round((df.shape[0])*0.8))

df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

df_train.shape

X_train = df_train.drop(['KC_Close'], axis=1)
x_test = df_test.drop(['KC_Close'], axis=1)

y_train = df_train['KC_Close']
y_test = df_test['KC_Close']


# %% - Normalize data

# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(X_train)
# x_test_scaled = scaler.transform(x_test)

# %% - build model

ada_reg = AdaBoostRegressor(n_estimators=100)
ada_reg.fit(X_train, y_train)

# %% - validation
scores = cross_val_score(ada_reg, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(ada_reg, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# %% - predictions

ypred = ada_reg.predict(x_test)
mse = mean_squared_error(y_test,ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % np.sqrt(mse))

# %% - plot predictions vs actual

x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()