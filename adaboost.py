# %% - source
# https://www.datatechnotes.com/2019/07/regression-example-with.html

# %% - imports

from operator import add
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import talib
import talib.abstract as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost

# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - get usd data method


def getUsdBrlData(filepath: str):

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.columns
    columns = ['Date', 'USD_Close',	'USD_Open',
               'USD_High', 'USD_Low', 'USD_Change %']
    df.columns = columns
    df['USD_Change %'] = df['USD_Change %'].str.replace(
        '%', '').astype('float') / 100.0
    df
    return df


def getOilPriceData(filepath: str):

    df = pd.read_csv(filepath)
    df = df.dropna()
    columns = ["Date", "CL_Open", "CL_High", "CL_Low",
               "CL_Close", "CL_Adj_Close", "CL_Volume"]
    df.columns = columns
    df = df.drop(['CL_Volume',"CL_Adj_Close"], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    return df


def getSugarPriceData(filepath: str):

    df = pd.read_csv(filepath)
    df = df.dropna()
    columns = ["Date", "SB_Open", "SB_High", "SB_Low",
               "SB_Close", "SB_Adj_Close", "SB_Volume"]
    df.columns = columns
    df = df.drop(['SB_Volume',"SB_Adj_Close"], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    return df

def getCornPriceData(filepath: str):

    df = pd.read_csv(filepath)
    df = df.dropna()
    columns = ["Date", "ZC_Open", "ZC_High", "ZC_Low",
               "ZC_Close", "ZC_Adj_Close", "ZC_Volume"]
    df.columns = columns
    df = df.drop(['ZC_Volume',"ZC_Adj_Close"], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    return df




# %% - data files
file_usd = "USD_BRL Historical Data-25092020.csv"
file_oil = "CLK21.NYM.csv"
file_sugar = "SBK21.NYB(1).csv"
file_corn = "ZCK21.CBT(1).csv"


# %% - get usd / brl data
df_exch = getUsdBrlData(file_usd)
columns_shift = ['USD_Close',	"USD_Open",
                 "USD_High",	"USD_Low",	"USD_Change %"]
# df_exch['USD_Close',	"USD_Open",	"USD_High",	"USD_Low",	"USD_Change %"] = df_exch[['USD_Close',	"USD_Open",	"USD_High",	"USD_Low",	"USD_Change %"]].shift(-1)

for column in columns_shift:
    df_exch[column] = df_exch[column].shift(-1)

df_exch

# %% - get oil contract prices
df_oil = getOilPriceData(file_oil)
columns_shift = ["CL_Open", "CL_High", "CL_Low",
                 "CL_Close", "CL_Adj_Close", "CL_Volume"]
for column in columns_shift:
    try:
        df_oil[column] = df_oil[column].shift(1)
    except KeyError as err:
        print("not found: ", column)

df_oil = df_oil.dropna()
df_oil

# %% - get sugar futures prices
df_sugar = getSugarPriceData(file_sugar)
columns_shift = ["SB_Open", "SB_High", "SB_Low",
                 "SB_Close", "SB_Adj_Close", "SB_Volume"]
for column in columns_shift:
    try:
        df_sugar[column] = df_sugar[column].shift(1)
    except KeyError as err:
        print("not found: ", column)

df_sugar = df_sugar.dropna()
df_sugar


# %% - get corn futures prices
df_corn = getCornPriceData(file_corn)
columns_shift = ["ZC_Open", "ZC_High", "ZC_Low",
                 "ZC_Close", "ZC_Adj_Close", "ZC_Volume"]
for column in columns_shift:
    try:
        df_corn[column] = df_corn[column].shift(1)
    except KeyError as err:
        print("not found: ", column)

df_corn = df_corn.dropna()
df_corn

# %% - get KC data
df = pd.read_csv(
    'KCK21.NYB-25092020.csv')

# %% - drop nan
df = df.dropna()
# columns_shift = []
# for column in columns_shift:
#     df[column] = df[column].shift(-1)

df

# %% - get rid of rows with vol "0"
columns = ["Date", "KC_Open", "KC_High", "KC_Low",
           "KC_Close", "KC_Adj_Close", "KC_Volume"]
df.columns = columns

df = df.drop(['KC_Volume'], axis=1)

# %% - change date to datetime

df['Date'] = pd.to_datetime(
    df['Date'], format='%Y-%m-%d', errors='coerce')


# %% - merge two dataframes on KC data dates
df = pd.merge(left=df, right=df_exch, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_oil, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_sugar, left_on='Date', right_on='Date')
# df = pd.merge(left=df, right=df_corn, left_on='Date', right_on='Date')




# %%- get exch rate prediction


# %% - set data column as index

df = df.set_index('Date')

df
# %% - plot closing price
df['KC_Close'].plot(grid=True)

plt.title('KCK21 closing prices')
plt.ylabel('price $')
plt.show()

df['USD_Close'].plot(grid=True)
plt.title('USD/BRL closing prices')
plt.ylabel('BRL')
plt.show()

df['CL_Close'].plot(grid=True)
plt.title('OIL/USD closing prices')
plt.ylabel('OIL')
plt.show()


# %% - plot on same graph

df['KC_Close'].plot(grid=True)
df['CL_Close'].plot(grid=True)
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

add_SMA(df, 'CL_Close', 10, "CL")
add_SMA(df, 'CL_Close', 20, "CL")
add_SMA(df, 'CL_Close', 50, "CL")
add_SMA(df, 'CL_Close', 100, "CL")
add_SMA(df, 'CL_Close', 200, "CL")

add_SMA(df, 'SB_Close', 10, "SB")
add_SMA(df, 'SB_Close', 20, "SB")
add_SMA(df, 'SB_Close', 50, "SB")
add_SMA(df, 'SB_Close', 100, "SB")
add_SMA(df, 'SB_Close', 200, "SB")

# add_SMA(df, 'ZC_Close', 10, "ZC")
# add_SMA(df, 'ZC_Close', 20, "ZC")
# add_SMA(df, 'ZC_Close', 50, "ZC")
# add_SMA(df, 'ZC_Close', 100, "ZC")
# add_SMA(df, 'ZC_Close', 200, "ZC")

# %% - calculate Exponential Moving Averages


def add_EMA(dataframe, colum_name,  period, commodity):
    dataframe['{}_EMA_{}'.format(commodity, period)] = ta.EMA(
        dataframe, timeperiod=period, price=colum_name)


add_EMA(df, 'KC_Close', 10, "KC")
add_EMA(df, 'KC_Close', 20, "KC")
add_EMA(df, 'KC_Close', 50, "KC")
add_EMA(df, 'KC_Close', 100, "KC")
add_EMA(df, 'KC_Close', 200, "KC")

add_EMA(df, 'CL_Close', 10, "CL")
add_EMA(df, 'CL_Close', 20, "CL")
add_EMA(df, 'CL_Close', 50, "CL")
add_EMA(df, 'CL_Close', 100, "CL")
add_EMA(df, 'CL_Close', 200, "CL")

add_EMA(df, 'SB_Close', 10, "SB")
add_EMA(df, 'SB_Close', 20, "SB")
add_EMA(df, 'SB_Close', 50, "SB")
add_EMA(df, 'SB_Close', 100, "SB")
add_EMA(df, 'SB_Close', 200, "SB")

# add_EMA(df, 'ZC_Close', 10, "ZC")
# add_EMA(df, 'ZC_Close', 20, "ZC")
# add_EMA(df, 'ZC_Close', 50, "ZC")
# add_EMA(df, 'ZC_Close', 100, "ZC")
# add_EMA(df, 'ZC_Close', 200, "ZC")


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

df['CL_ATR_14'] = talib.ATR(df['CL_High'].values, df['CL_Low'].values,
                            df['CL_Close'].values, timeperiod=14)
df['CL_ADX_14'] = talib.ADX(df['CL_High'].values, df['CL_Low'].values,
                            df['CL_Close'].values, timeperiod=14)
df['CL_CCI_14'] = talib.CCI(df['CL_High'].values, df['CL_Low'].values,
                            df['CL_Close'].values, timeperiod=14)
df['CL_ROC_10'] = talib.ROC(df['CL_Close'], timeperiod=10)
df['CL_RSI_14'] = talib.RSI(df['CL_Close'], timeperiod=14)
df['CL_Williams_%R_14'] = talib.ATR(df['CL_High'].values, df['CL_Low'].values,
                                    df['CL_Close'].values, timeperiod=14)
df['CL_Slowd'] = talib.STOCH(df['CL_High'].values,
                             df['CL_Low'].values,
                             df['CL_Close'].values,
                             fastk_period=5,
                             slowk_period=3,
                             slowk_matype=0,
                             slowd_period=3,
                             slowd_matype=0)[1]


df['SB_ATR_14'] = talib.ATR(df['SB_High'].values, df['SB_Low'].values,
                            df['SB_Close'].values, timeperiod=14)
df['SB_ADX_14'] = talib.ADX(df['SB_High'].values, df['SB_Low'].values,
                            df['SB_Close'].values, timeperiod=14)
df['SB_CCI_14'] = talib.CCI(df['SB_High'].values, df['SB_Low'].values,
                            df['SB_Close'].values, timeperiod=14)
df['SB_ROC_10'] = talib.ROC(df['SB_Close'], timeperiod=10)
df['SB_RSI_14'] = talib.RSI(df['SB_Close'], timeperiod=14)
df['SB_Williams_%R_14'] = talib.ATR(df['SB_High'].values, df['SB_Low'].values,
                                    df['SB_Close'].values, timeperiod=14)
df['SB_Slowd'] = talib.STOCH(df['SB_High'].values,
                             df['SB_Low'].values,
                             df['SB_Close'].values,
                             fastk_period=5,
                             slowk_period=3,
                             slowk_matype=0,
                             slowd_period=3,
                             slowd_matype=0)[1]

# df['ZC_ATR_14'] = talib.ATR(df['ZC_High'].values, df['ZC_Low'].values,
#                             df['ZC_Close'].values, timeperiod=14)
# df['ZC_ADX_14'] = talib.ADX(df['ZC_High'].values, df['ZC_Low'].values,
#                             df['ZC_Close'].values, timeperiod=14)
# df['ZC_CCI_14'] = talib.CCI(df['ZC_High'].values, df['ZC_Low'].values,
#                             df['ZC_Close'].values, timeperiod=14)
# df['ZC_ROC_10'] = talib.ROC(df['ZC_Close'], timeperiod=10)
# df['ZC_RSI_14'] = talib.RSI(df['ZC_Close'], timeperiod=14)
# df['ZC_Williams_%R_14'] = talib.ATR(df['ZC_High'].values, df['ZC_Low'].values,
#                                     df['ZC_Close'].values, timeperiod=14)
# df['ZC_Slowd'] = talib.STOCH(df['ZC_High'].values,
#                              df['ZC_Low'].values,
#                              df['ZC_Close'].values,
#                              fastk_period=5,
#                              slowk_period=3,
#                              slowk_matype=0,
#                              slowd_period=3,
#                              slowd_matype=0)[1]




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

df['target'] = df['KC_Close']

columns_shift = ['KC_Open', 'KC_High', 'KC_Low', 'KC_Close', 'KC_SMA_10',
                 'KC_SMA_20', 'KC_SMA_50', 'KC_SMA_100', 'KC_SMA_200', 'KC_EMA_10',
                 'KC_EMA_20', 'KC_EMA_50', 'KC_EMA_100', 'KC_EMA_200', 'KC_ADX_14',
                 'KC_CCI_14', 'KC_Slowd', 'KC_ROC_10', 'KC_RSI_14', 'KC_Williams_%R_14', ]
for column in columns_shift:
    df[column] = df[column].shift(1)

# %% - get shape
df = df.dropna()
df.columns

# %% -
df

# %% -corr. matrix

# corrMatrix = df.corr()
# print(corrMatrix)

# %% - eliminate low corellation features

# df = df[['KC_Close',"KC_Open", "KC_High", "KC_Low", "KC_SMA_10", "KC_SMA_20", "KC_EMA_10", "KC_EMA_20", "KC_EMA_50",
#          "USD_ATR_14", "USD_ATR_10", "USD_Williams_%R_14", "KC_RSI_14"]]

# %% - get train and test sets

cutoff = int(round((df.shape[0])*0.8))

df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

df_train.shape

X_train = df_train.drop(['target', "KC_Adj_Close"], axis=1)
x_test = df_test.drop(['target', "KC_Adj_Close"], axis=1)

y_train = df_train['target']
y_test = df_test['target']


# %% - Normalize data

# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(X_train)
# x_test_scaled = scaler.transform(x_test)

# %% - build model -adaboost

ada_reg = AdaBoostRegressor(n_estimators=100)
ada_reg.fit(X_train, y_train)

# %% - validation adaboost
scores = cross_val_score(ada_reg, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(ada_reg, X_train, y_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# %% - predictions

adaboost_y_pred = ada_reg.predict(x_test)
mse_test = mean_squared_error(y_test, adaboost_y_pred)
print("MSE: %.2f" % mse_test)
print("RMSE: %.2f" % np.sqrt(mse_test))


# %% - plot predictions vs actual

x_ax = range(len(y_test))
x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, adaboost_y_pred, lw=0.8,
         color="red", label="predicted (adaboost)")
plt.legend()
plt.show()


# %% - build model -grad. boost

grad_reg = GradientBoostingRegressor(
    n_estimators=100, subsample=0.8, max_depth=1, criterion="mse")
grad_reg.fit(X_train, y_train)

# %% - validation
scores = cross_val_score(grad_reg, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(grad_reg, X_train, y_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# %% - predictions


grad_boost_y_pred = grad_reg.predict(x_test)
mse_test = mean_squared_error(y_test, grad_boost_y_pred)
print("MSE: %.2f" % mse_test)
print("RMSE: %.2f" % np.sqrt(mse_test))

# %% - plot predictions vs actual

x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, grad_boost_y_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# %% - xgboost model

xgb_regressor = xgboost.XGBRegressor(
    n_estimators=35,
    reg_lambda=1,
    gamma=0.5,
    max_depth=3,
    subsample=0.5,
    tree_method='exact'
    # booster="gblinear"
)

xgb_regressor.fit(X_train, y_train)


# %% - validation
scores = cross_val_score(grad_reg, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgb_regressor, X_train, y_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())


# %% - predict
xgb_reg_y_pred = xgb_regressor.predict(x_test)
# %%  - MSE
mse_test = mean_squared_error(y_test, xgb_reg_y_pred)
print("MSE: %.2f" % mse_test)
print("RMSE: %.2f" % np.sqrt(mse_test))


# %%


x_ax = range(len(y_test))
x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, xgb_reg_y_pred, lw=0.8,
         color="red", label="predicted (xgboost)")
plt.legend()
plt.show()


# %%
preds_actual_df = pd.DataFrame()
preds_actual_df['adaboost'] = pd.Series(adaboost_y_pred)
preds_actual_df['grad boost'] = pd.Series(grad_boost_y_pred)
preds_actual_df['xgboost'] = pd.Series(xgb_reg_y_pred)
print("********")
print("predictions for the past 5 days:")
preds_actual_df.tail()
print("********")


# %% - plot predictions against preds_actual

x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, adaboost_y_pred, lw=0.8,
         color="red", label="predicted (adaboost)")
plt.plot(x_ax, grad_boost_y_pred, lw=0.8,
         color="green", label="predicted (gboost)")
plt.plot(x_ax, xgb_reg_y_pred, lw=0.8,
         color="purple", label="predicted (xgboost)")
plt.legend()
plt.show()
