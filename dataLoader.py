import pandas as pd
import numpy as np


def getInvComData(filepath: str,
                  name: str) -> pd.DataFrame:

    df = pd.read_csv(filepath, decimal='.',thousands=',')
    try:
        df = df.drop(['Vol.'], axis=1)
    except KeyError:
        print('Vol. column not found, skipping')
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.columns
    columns = ['Date', '{}_Close'.format(name),	'{}_Open'.format(name),
               '{}_High'.format(name), '{}_Low'.format(name),
               '{}_Change %'.format(name)]
    df.columns = columns
    df['{}_Change %'.format(name)] = \
        df['{}_Change %'.format(name)].str.replace('%', '').\
        astype('float') / 100.0
    # df = df.replace(',', '', regex=True).values
    # for c in df.columns:
    #     if not c == 'Date':
    #         df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def getYahFinData(filepath: str,
                  name: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna()
    columns = ["Date", "{}_Open".format(name), "{}_High".format(name),
               "{}_Low".format(name), "{}_Close".format(name),
               "{}_Adj_Close".format(name), "{}_Volume".format(name)]
    df.columns = columns
    df = df.drop(['{}_Volume'.format(name),
                  "{}_Adj_Close".format(name)], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    return df
