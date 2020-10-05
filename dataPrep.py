from adaboost import add_EMA
from typing import List
import pandas as pd
import talib
import talib.abstract as ta


def shiftColumns(data_frame: pd.DataFrame,
                 columns_list: list,
                 num_of_rows: int) -> pd.DataFrame:
    for c in columns_list:
        try:
            data_frame[c] = data_frame[c].shift(num_of_rows)
        except KeyError:
            print('column {} not found, skipping'.format(c))
    data_frame = data_frame.dropna()
    return data_frame


def addSma(dataframe: pd.DataFrame,
           colum_name: str,
           period: int,
           commodity: str):
    dataframe['{}_SMA_{}'.format(commodity, period)] = dataframe[colum_name].rolling(
        window=period).mean()


def addEma(dataframe, colum_name,  period, commodity):
    dataframe['{}_EMA_{}'.format(commodity, period)] = ta.EMA(
        dataframe, timeperiod=period, price=colum_name)


def addSmaEma(dataframe: pd.DataFrame,
              colum_name: str,
              periods: List[int],
              commodity: str,
              ema: bool = True,
              sma: bool = False):
    if ema == True:
        for p in periods:
            addEma(dataframe, colum_name, p, commodity)
    if sma == True:
        for p in periods:
            addSma(dataframe, colum_name, p, commodity)


def addAtr(dataframe: pd.DataFrame,
           time_periods: List[int],
           commodity: str):
    for p in time_periods:
        dataframe['{}_ATR_{}'.format(commodity, p)] = \
            talib.ATR(dataframe['{}_High'.format(commodity)].values,
                      dataframe['{}_Low'.format(commodity)].values,
                      dataframe['{}_Close'.format(commodity)].values,
                      timeperiod=p)


def addAdx(dataframe: pd.DataFrame,
           time_periods: List[int],
           commodity: str):
    for p in time_periods:
        dataframe['{}_ADX_{}'.format(commodity, p)] = \
            talib.ADX(dataframe['{}_High'.format(commodity)].values,
                      dataframe['{}_Low'.format(commodity)].values,
                      dataframe['{}_Close'.format(commodity)].values,
                      timeperiod=p)


def addCci(dataframe: pd.DataFrame,
           time_periods: List[int],
           commodity: str):
    for p in time_periods:
        dataframe['{}_CCI_{}'.format(commodity, p)] = \
            talib.CCI(dataframe['{}_High'.format(commodity)].values,
                      dataframe['{}_Low'.format(commodity)].values,
                      dataframe['{}_Close'.format(commodity)].values,
                      timeperiod=p)


def addWillR(dataframe: pd.DataFrame,
             time_periods: List[int],
             commodity: str):
    for p in time_periods:
        dataframe['{}_Williams_%R_{}'.format(commodity, p)] = \
            talib.WILLR(dataframe['{}_High'.format(commodity)].values,
                        dataframe['{}_Low'.format(commodity)].values,
                        dataframe['{}_Close'.format(commodity)].values,
                        timeperiod=p)


def addRsi(dataframe: pd.DataFrame,
           time_periods: List[int],
           commodity: str):
    for p in time_periods:
        dataframe['{}_RSI_{}'.format(commodity, p)] = \
            talib.RSI(df['{}_Close'.format(commodity)], timeperiod=p)


def addRoc(dataframe: pd.DataFrame,
           time_periods: List[int],
           commodity: str):
    for p in time_periods:
        dataframe['{}_ROC_{}'.format(commodity, p)] = \
            talib.ROC(df['{}_Close'.format(commodity)], timeperiod=p)


def addSinglePeriodFinFeat(dataframe: pd.DataFrame,
                           time_periods: List[int],
                           commodity: str,
                           atr=True,
                           adx=True,
                           cci=True,
                           willR=True,
                           rsi=True,
                           roc=True):
                           if atr == True:
                               addAtr(dataframe, time_periods, commodity)
                           if adx == True:
                               addAdx(dataframe, time_periods, commodity)
                           if cci == True:
                               addCci(dataframe, time_periods, commodity)
                           if willR == True:
                               addWillR(dataframe, time_periods, commodity)
                           if rsi == True:
                               addRsi(dataframe, time_periods, commodity)
                           if roc == True:
                               addRoc(dataframe, time_periods, commodity)




