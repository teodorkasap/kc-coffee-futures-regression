
import pandas as pd
import numpy as np
from datetime import datetime

# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


def getUsdBrlData():

    df = pd.read_csv(
        'USD_BRL Historical Data(1).csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.columns
    columns = ['Date', 'USD_Close',	'USD_Open_y', 'USD_High_y', 'USD_Low_y', 'USD_Change %']
    df.columns = columns
    df
    return df

