# %% - imports
import pickle
from pickle import ADDITEMS
import pandas as pd

# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

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
    df
    return df


# %% - declare files
file_usd = "USD_BRL Historical Data-25092020.csv"
file_kc = 'KCK21.NYB-25092020.csv'

df_exch = getUsdBrlData(file_usd)
df = pd.read_csv(file_kc)


# %% - get models

file_ada = "final_ada_model.sav"
file_grad = "final_grad_model.sav"
file_xgb = "final_xgb_model.sav"

ada_reg = pickle.load(open(file_ada, 'rb'))
grad_reg = pickle.load(open(file_grad, 'rb'))
xgb_reg = pickle.load(open(file_xgb, 'rb'))

