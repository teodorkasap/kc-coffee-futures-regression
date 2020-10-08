# %% - imports

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost


# own modules
from dataLoader import getInvComData, getYahFinData
from dataPrep import shiftColumns, addSmaEma, addSinglePeriodFinFeat, \
    addStochFast, addStochSlow, addUltOsc


# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - get various datasets

file_usd_brl = "USD_BRL Historical Data-25092020.csv"
df_usd_brl = getInvComData(file_usd_brl, "USDBR")

# file_sugar = "SBK21.NYB(1).csv"
# df_sugar = getYahFinData(file_sugar, "SB")

file_oil = "CLK21.NYM.csv"
df_oil = getYahFinData(file_oil, "CL")

file_coffee = 'KCK21.NYB-25092020.csv'
df_coffe = getYahFinData(file_coffee, "KC")

file_usd_cop = "COP=X.csv"
df_usd_cop = getYahFinData(file_usd_cop, "USDCP")

US_T_note_2_year_file = "ZT=F.csv"
df_T_note_2y = getYahFinData(US_T_note_2_year_file, "ZT")

# %% - start merging data frames
df = pd.merge(left=df_coffe, right=df_usd_brl, left_on='Date', right_on='Date')
# df = pd.merge(left=df, right=df_sugar, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_oil, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_usd_cop, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_T_note_2y, left_on='Date', right_on='Date')


# %% - set index
df = df.set_index('Date')
df['target'] = df['KC_Close']
df.columns

# %% - shift columns
columns = ['KC_Open', 'KC_High', 'KC_Low', 'KC_Close', 'USDBR_Close', 'USDBR_Open',
           'USDBR_High', 'USDBR_Low', 'USDBR_Change %', 'CL_Open', 'CL_High',
           'CL_Low', 'CL_Close', 'USDCP_Open', 'USDCP_High', 'USDCP_Low',
           'USDCP_Close', 'ZT_Open', 'ZT_High', 'ZT_Low', 'ZT_Close', ]


shiftColumns(df, columns, 1)

# %% - drop nan
df = df.dropna()

# %% - add EMA & SMA
periods = [5, 10]

addSmaEma(df, "KC_Close", periods, "KC")
addSmaEma(df, "USDBR_Close", periods, "USDBR", sma=False)
addSmaEma(df, "USDCP_Close", periods, "USDCP", sma=False)
addSmaEma(df, "SB_Close", periods, "SB",sma=False)
addSmaEma(df, "CL_Close", periods, "CL", sma=False)
addSmaEma(df, "ZT_Close", periods, "ZT", sma=False)

# %% - add other financial features

periods = [7,14]

addSinglePeriodFinFeat(df, periods, "KC", trix=False, rocr=False,
                       willR=True, roc=True, atr=True, adx=True, cci=True, rsi=True)
addSinglePeriodFinFeat(df, periods, "USDBR", trix=False, rocr=False,
                       willR=True, roc=True, atr=True, adx=True, cci=True, rsi=True)
addSinglePeriodFinFeat(df, periods, "USDCP", trix=False, rocr=False,
                       willR=False, roc=True, atr=True, adx=True, cci=True, rsi=True)
# # addSinglePeriodFinFeat(df, periods, "SB", trix=True, rocr=True,
# #                        willR=True, roc=True, atr=True, adx=True, cci=True, rsi=True)
addSinglePeriodFinFeat(df, periods, "CL", trix=False, rocr=False,
                       willR=False, roc=True, atr=True, adx=True, cci=True, rsi=True)
addSinglePeriodFinFeat(df, periods, "ZT", trix=False, rocr=False,
                       willR=False, roc=True, atr=True, adx=True, cci=True, rsi=True)

# addStochFast(df, "KC")
# addStochFast(df, "USDBR")
# addStochFast(df, "USDCP")
# # addStochFast(df, "SB")
# addStochFast(df, "CL")
# addStochFast(df,"ZT")

addStochSlow(df, "KC")
addStochSlow(df, "USDBR")
addStochSlow(df, "USDCP")
# # addStochSlow(df, "SB")
addStochSlow(df, "CL")
addStochSlow(df,"ZT")

# addUltOsc(df, "KC")
# addUltOsc(df, "USDBR")
# addUltOsc(df, "USDCP")
# # addUltOsc(df, "SB")
# addUltOsc(df, "CL")
# addUltOsc(df,"ZT")

df.shape
df.dropna()


# %% - clear nan an infinite values
print("NaN values: ", np.any(np.isnan(df)))
print("infinite values: ", np.all(np.isfinite(df)))

df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# df = df[~np.isnan(df)]
# df = df[~np.isfinite(df)]
df.shape

for c in df.columns:
    print(c)

# %% - eliminate unnecessary cols
cols_eliminate = ['KC_High', 'KC_Low', 'USDBR_High', 'USDBR_Low', 'USDBR_Change %',
                  'CL_High', 'CL_Low', 'USDCP_High', 'USDCP_Low', 'ZT_High', 'ZT_Low']
for c in cols_eliminate:
    try:
        df = df.drop(c, axis=1)
    except KeyError:
        print("column {} not found, skipping".format(c))

# %% - train test split
cutoff = int(round((df.shape[0])*0.8))

df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]


X_train = df_train.drop(['target'], axis=1)
x_test = df_test.drop(['target'], axis=1)

y_train = df_train['target']
y_test = df_test['target']

df_train
df_test


# %% - define models

# Adaboost
ada_reg = AdaBoostRegressor(n_estimators=500)
ada_reg.fit(X_train, y_train)

# Grad.Boost
grad_reg = GradientBoostingRegressor(
    n_estimators=50, subsample=0.8, max_depth=2, criterion="mse", learning_rate=0.1)
grad_reg.fit(X_train, y_train)

# XGBoost
xgb_regressor = xgboost.XGBRegressor(
    n_estimators=35, reg_lambda=0.1,
    gamma=0.01, max_depth=2, subsample=0.9,
    # tree_method='exact',
)
xgb_regressor.fit(X_train, y_train)

# %% - Validation

# Adaboost
print("****************")
print("*** Adaboost ***")
print("****************")

scores = cross_val_score(ada_reg, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())


kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(ada_reg, X_train, y_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

adaboost_y_pred = ada_reg.predict(x_test)
mse_test = mean_squared_error(y_test, adaboost_y_pred)
print("MSE: %.2f" % mse_test)
print("RMSE: %.2f" % np.sqrt(mse_test))

print("****************")
print("****************")
print()

# Grad.Boost
print("****************")
print("** GradBoost ***")
print("****************")

scores = cross_val_score(grad_reg, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(grad_reg, X_train, y_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

grad_boost_y_pred = grad_reg.predict(x_test)
mse_test = mean_squared_error(y_test, grad_boost_y_pred)
print("MSE: %.2f" % mse_test)
print("RMSE: %.2f" % np.sqrt(mse_test))

print("****************")
print("****************")
print()

# XGBoost
print("****************")
print("*** XGBoost ****")
print("****************")

scores = cross_val_score(xgb_regressor, X_train, y_train, cv=5)
print("Mean cross-validataion score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgb_regressor, X_train, y_train, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

xgb_reg_y_pred = xgb_regressor.predict(x_test)
mse_test = mean_squared_error(y_test, xgb_reg_y_pred)
print("MSE: %.2f" % mse_test)
print("RMSE: %.2f" % np.sqrt(mse_test))

print("****************")
print("****************")
print()
