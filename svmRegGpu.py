from sklearn.metrics import mean_squared_error
from h2ogpu
from sklearn.model_selection import cross_val_score, KFold
from prepUsdBrlData import getUsdBrlData
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import talib
import talib.abstract as ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# %% - get usd / brl data
df_exch = getUsdBrlData()

# %% - get KC data
df = pd.read_csv(
    'KCK21.NYB.csv')

# %% - drop nan
df = df.dropna()


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

# df['KC_ATR_14'] = talib.ATR(df['KC_High'].values, df['KC_Low'].values,
#                             df['KC_Close'].values, timeperiod=14)

df['KC_ADX_14'] = talib.ADX(df['KC_High'].values, df['KC_Low'].values,
                            df['KC_Close'].values, timeperiod=14)

df['KC_CCI_14'] = talib.CCI(df['KC_High'].values, df['KC_Low'].values,
                            df['KC_Close'].values, timeperiod=14)

# df['KC_ROC_10'] = talib.ROC(df['KC_Close'], timeperiod=10)

# df['KC_RSI_14'] = talib.RSI(df['KC_Close'], timeperiod=14)

# df['KC_Williams_%R_14'] = talib.ATR(df['KC_High'].values, df['KC_Low'].values,
#                                     df['KC_Close'].values, timeperiod=14)

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

# %%
df = df.drop(['KC_Adj_Close'], axis=1)

# %%
df["target"] = df["KC_Close"]

columns_shift = ['KC_Open', 'KC_High', 'KC_Low', 'KC_Close', 'USD_Close', 'USD_Open',
                 'USD_High', 'USD_Low', 'USD_Change %', 'KC_SMA_10', 'KC_SMA_20',
                 'KC_SMA_50', 'KC_SMA_100', 'KC_SMA_200', 'KC_EMA_10', 'KC_EMA_20',
                 'KC_EMA_50', 'KC_EMA_100', 'KC_EMA_200', 'KC_ADX_14', 'KC_CCI_14',
                 'KC_Slowd', 'USD_ATR_14', 'USD_ATR_10', 'USD_ADX_14', 'USD_ADX_10',
                 'USD_CCI_14', 'USD_CCI_10', 'USD_ROC_10', 'USD_ROC_5', 'USD_RSI_14',
                 'USD_RSI_7', 'USD_Williams_%R_14', 'USD_Williams_%R_7', 'USD_Slowk',
                 'USD_Slowd', 'USD_SMA_5', 'USD_SMA_10', 'USD_SMA_25', 'USD_SMA_50',
                 'USD_SMA_100', 'USD_EMA_5', 'USD_EMA_10', 'USD_EMA_25', 'USD_EMA_50',
                 'USD_EMA_100']


for column in columns_shift:
    df[column] = df[column].shift(1)

# %%
df = df.dropna()
df


# %% - get train and test sets

cutoff = int(round((df.shape[0])*0.8))

df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]


X_train = df_train.drop(['target'], axis=1)
x_test = df_test.drop(['target'], axis=1)

y_train = df_train['target']
y_test = df_test['target']


X_train.shape, y_train.shape, x_test.shape, y_test.shape

# %% - scale dataset

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
x_test_sc = sc.transform(x_test)

X_train_sc
x_test_sc


# %% model

svm_reg = SVR(kernel='poly',gamma = 'default')
svm_reg.fit(X_train_sc, y_train)


# %% - validation


scores = cross_val_score(svm_reg, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(svm_reg, X_train, y_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# %% - predict


y_pred = svm_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % np.sqrt(mse))

# %% - plot predictions vs actual

x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# %%
preds_actual_df = pd.DataFrame()
preds_actual_df['Predicted'] = pd.Series(y_pred)
preds_actual_df

# %%
# y_test
pd.set_option("display.max_rows", None)
preds_actual_df['Actual'] = y_test.values
preds_actual_df
