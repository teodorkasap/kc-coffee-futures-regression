# %% - imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost


# own modules
from dataLoader import getInvComData, getYahFinData
from dataPrep import shiftColumns, addSmaEmaWma, addSinglePeriodFinFeat, \
    addStochFast, addStochSlow, addUltOsc, addWeekDay, addBBands


# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - get various datasets

file_usd_brl = "USD_BRL Historical Data-25092020.csv"
df_usd_brl = getInvComData(file_usd_brl, "USDBR")

file_sugar = "SBK21.NYB(1).csv"
df_sugar = getYahFinData(file_sugar, "SB")

file_oil = "CLK21.NYM.csv"
df_oil = getYahFinData(file_oil, "CL")

file_coffee = 'KCK21.NYB-25092020.csv'
df_coffe = getYahFinData(file_coffee, "KC")

file_usd_cop = "COP=X.csv"
df_usd_cop = getYahFinData(file_usd_cop, "USDCP")

US_T_note_2_year_file = "ZT=F.csv"
df_T_note_2y = getYahFinData(US_T_note_2_year_file, "ZT")

US_T_note_5_year_file = 'ZF=F.csv'
df_T_note_5y = getYahFinData(US_T_note_2_year_file, "ZF")

file_corn = 'ZCK21.CBT(1).csv'
df_corn = getYahFinData(file_corn, 'ZC')

file_SP_500_fut = 'US 500 Futures Historical Data.csv'
df_sp500 = getInvComData(file_SP_500_fut, 'SP')


# %% - start merging data frames
df = pd.merge(left=df_coffe, right=df_usd_brl, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_sugar, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_oil, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_usd_cop, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_T_note_2y, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_T_note_5y, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_corn, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_sp500, left_on='Date', right_on='Date')

# %% - add weekday number

addWeekDay(df, 'Date')


# %% - set index
df = df.set_index('Date')
df['target'] = df['KC_Close']
df.columns

# %% - shift columns
columns = ['KC_Open', 'KC_High', 'KC_Low', 'KC_Close', 'USDBR_Close', 'USDBR_Open',
           'USDBR_High', 'USDBR_Low', 'USDBR_Change %', 'CL_Open', 'CL_High',
           'CL_Low', 'CL_Close', 'USDCP_Open', 'USDCP_High', 'USDCP_Low',
           'USDCP_Close', 'ZT_Open', 'ZT_High', 'ZT_Low', 'ZT_Close',
           'ZF_Open', 'ZF_High', 'ZF_Low', 'ZF_Close',
           'ZC_Open', 'ZC_High', 'ZC_Low', 'ZC_Close',
           'SP_Open', 'SP_High', 'SP_Low', 'SP_Close'
           ]


shiftColumns(df, columns, 1)

# %% - drop nan
df = df.dropna()

# %% - add EMA & SMA
periods = [5, 7, 10, 14]

addSmaEmaWma(dataframe=df, colum_name="KC_Close",
             periods=periods, commodity="KC")
addSmaEmaWma(df, "USDBR_Close", periods, "USDBR")
addSmaEmaWma(df, "USDCP_Close", periods, "USDCP")
addSmaEmaWma(df, "SB_Close", periods, "SB")
addSmaEmaWma(df, "CL_Close", periods, "CL")
addSmaEmaWma(df, "ZT_Close", periods, "ZT")
addSmaEmaWma(df, "ZF_Close", periods, "ZF")
addSmaEmaWma(df, "ZC_Close", periods, "ZC")
addSmaEmaWma(df, "SP_Close", periods, "SP")


# %% - add other financial features

periods = [7, 10, 14, 20]

addSinglePeriodFinFeat(df, periods, "KC", trix=True, rocr=True, rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)
addSinglePeriodFinFeat(df, periods, "USDBR", trix=True, rocr=True, rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)
addSinglePeriodFinFeat(df, periods, "USDCP", trix=True, rocr=True,  rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)
addSinglePeriodFinFeat(df, periods, "SB", trix=True, rocr=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True,
                       rsi=True)
addSinglePeriodFinFeat(df, periods, "CL", trix=True, rocr=True, rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)
addSinglePeriodFinFeat(df, periods, "ZT", trix=True, rocr=True, rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)
addSinglePeriodFinFeat(df, periods, "ZF", trix=True, rocr=True, rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)
addSinglePeriodFinFeat(df, periods, "ZC", trix=True, rocr=True, rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)
addSinglePeriodFinFeat(df, periods, "SP", trix=True, rocr=True, rsi=True,
                       willR=True, roc=True, atr=True, adx=True, cci=True)


addStochFast(df, "KC")
addStochFast(df, "USDBR")
addStochFast(df, "USDCP")
addStochFast(df, "SB")
addStochFast(df, "CL")
addStochFast(df, "ZT")
addStochFast(df, "ZF")
addStochFast(df, "ZC")
addStochFast(df, "SP")

addStochSlow(df, "KC")
addStochSlow(df, "USDBR")
addStochSlow(df, "USDCP")
addStochSlow(df, "SB")
addStochSlow(df, "CL")
addStochSlow(df, "ZT")
addStochSlow(df, "ZF")
addStochSlow(df, "ZC")
addStochSlow(df, "SP")

addUltOsc(df, "KC")
addUltOsc(df, "USDBR")
addUltOsc(df, "USDCP")
addUltOsc(df, "SB")
addUltOsc(df, "CL")
addUltOsc(df, "ZT")
addUltOsc(df, "ZF")
addUltOsc(df, "ZC")
addUltOsc(df, "SP")


# %% - adding BBands

addBBands(df, "KC", 5)

# %%
df.shape
df = df.dropna()


# %% - clear nan an infinite values
print("NaN values: ", np.any(np.isnan(df)))
print("infinite values: ", np.all(np.isfinite(df)))

df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# df = df[~np.isnan(df)]
# df = df[~np.isfinite(df)]
df.shape

for c in df.columns:
    print(c)

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


# %% - select features
sel_regressor = xgboost.XGBRegressor(
    gamma=0.0, n_estimators=150, base_score=0.7, colsample_bytree=1,
    learning_rate=0.05)


xgbModel = sel_regressor.fit(X_train, y_train,
                             eval_set=[(X_train, y_train), (x_test, y_test)],
                             verbose=False)

eval_result = sel_regressor.evals_result()

training_rounds = range(len(eval_result['validation_0']['rmse']))

# %% - plot training and validation errors

%matplotlib inline

plt.scatter(x=training_rounds,
            y=eval_result['validation_0']['rmse'], label='Training Error')
plt.scatter(x=training_rounds,
            y=eval_result['validation_1']['rmse'], label='Validation Error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()

# %% - plot feature importance
fig = plt.figure(figsize=(40, 10))
plt.xticks(rotation='vertical', fontsize=5)
plt.bar([i for i in range(len(xgbModel.feature_importances_))],
        xgbModel.feature_importances_.tolist(), tick_label=x_test.columns)
plt.title('Feature importance in comparison.')
plt.show()


# %%
importances = xgbModel.feature_importances_.tolist()
for i, l in zip(importances, x_test.columns):
    print(i, l)
# %% - eliminate unnecessary cols


cols_eliminate = ['USDBR_Open', 'SB_Close', 'SB_Open', 'CL_High', 'CL_Low',
                  'ZF_Open', 'ZF_High', 'ZF_Low', 'ZF_Close', 'ZC_Open',
                  'ZC_Low', 'ZC_Close']
for c in cols_eliminate:
    try:
        df = df.drop(c, axis=1)
    except KeyError:
        print("column {} not found, skipping".format(c))

# %% - select features
# df = df[['target', 'KC_High', 'KC_Low', 'KC_Close', 'KC_EMA_7', 'KC_EMA_10',
#          'KC_EMA_14', 'KC_WMA_5', 'KC_WMA_7', 'KC_WMA_14',
#          'USDBR_SMA_14',  'SB_SMA_5', 'SB_WMA_7', 'CL_WMA_10',
#          'CL_WMA_14', 'CL_SMA_10', 'ZT_EMA_10', 'KC_CCI_20', 'KC_RSI_20',
#          'KC_TRIX_14', 'SB_TRIX_10', 'ZT_ADX_20', 'ZT_TRIX_20']]

# %% - define models

# Adaboost
ada_reg = AdaBoostRegressor(n_estimators=500)
ada_reg.fit(X_train, y_train)

# Grad.Boost
grad_reg = GradientBoostingRegressor(
    n_estimators=50, subsample=0.8, max_depth=2, criterion="mse",
    learning_rate=0.1)
grad_reg.fit(X_train, y_train)


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

# %%

# XGBoost
xgb_regressor = xgboost.XGBRegressor(
    n_estimators=50, reg_lambda=0.1,
    gamma=0.5, max_depth=2, subsample=0.9,
    # tree_method='exact',
)
xgb_regressor.fit(X_train, y_train)


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


# %% - Predictions

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
