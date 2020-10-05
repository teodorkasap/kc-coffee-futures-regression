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

file_usd_brl= "USD_BRL Historical Data-25092020.csv"
df_usd_brl = getInvComData(file_usd_brl,"USD")

file_sugar = "SBK21.NYB(1).csv"
df_sugar = getYahFinData(file_sugar,"SB")

file_oil = "CLK21.NYM.csv"
df_oil = getYahFinData(file_oil,"CL")

file_coffee = 'KCK21.NYB-25092020.csv'
df_coffe = getYahFinData(file_coffee,"KC")

# %% - start merging data frames
df = pd.merge(left=df_coffe, right=df_usd_brl, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_sugar, left_on='Date', right_on='Date')
df = pd.merge(left=df, right=df_oil, left_on='Date', right_on='Date')

# %% - set index
df = df.set_index('Date')
df['target'] = df['KC_Close']
df.columns

# %% - shift columns
columns = ['KC_Open', 'KC_High', 'KC_Low', 'KC_Close', 'USD_Close', 'USD_Open',
       'USD_High', 'USD_Low', 'USD_Change %', 'SB_Open', 'SB_High', 'SB_Low',
       'SB_Close', 'CL_Open', 'CL_High', 'CL_Low', 'CL_Close',]

shiftColumns(df,columns,1)

# %% - drop nan
df.dropna()

# %% - add EMA & SMA
periods = [5,10,15,20]

addSmaEma(df,"KC_Close",periods,"KC")
addSmaEma(df,"USD_Close",periods,"USD")
addSmaEma(df,"SB_Close",periods,"SB")
addSmaEma(df,"CL_Close",periods,"CL")

# %% - add other financial features

addSinglePeriodFinFeat(df,periods,"KC")
addSinglePeriodFinFeat(df,periods,"USD")
addSinglePeriodFinFeat(df,periods,"SB")
addSinglePeriodFinFeat(df,periods,"CL")

addStochFast(df,"KC")
addStochFast(df,"USD")
addStochFast(df,"SB")
addStochFast(df,"CL")

addStochSlow(df,"KC")
addStochSlow(df,"USD")
addStochSlow(df,"SB")
addStochSlow(df,"CL")

addUltOsc(df,"KC")
addUltOsc(df,"USD")
addUltOsc(df,"SB")
addUltOsc(df,"CL")

df.shape
df.dropna()
# %% - train test split
cutoff = int(round((df.shape[0])*0.8))

df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

df_train.shape

X_train = df_train.drop(['target'], axis=1)
x_test = df_test.drop(['target'], axis=1)

y_train = df_train['target']
y_test = df_test['target']

# %% - 