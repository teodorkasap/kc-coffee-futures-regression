import numpy as np
from typing import List
import pandas as pd
import talib
from talib import MA_Type


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


def addWma(dataframe, colum_name, period, commodity):
    weights = np.arange(1, period+1)
    dataframe['{}_WMA_{}'.format(commodity, period)] = \
        dataframe[colum_name].rolling(period).apply(
            lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)


def addSma(dataframe: pd.DataFrame,
           colum_name: str,
           period: int,
           commodity: str):
    dataframe['{}_SMA_{}'.format(commodity, period)] = dataframe[colum_name].\
        rolling(window=period).mean()


def addEma(dataframe, colum_name,  period, commodity):
    dataframe['{}_EMA_{}'.format(commodity, period)] = \
        dataframe[colum_name].ewm(span=period).mean()


def addSmaEmaWma(dataframe: pd.DataFrame,
                 colum_name: str,
                 periods: List[int],
                 commodity: str,
                 ema: bool = True,
                 sma: bool = True,
                 wma: bool = True):
    if ema is True:
        for p in periods:
            addEma(dataframe=dataframe, colum_name=colum_name,
                   period=p, commodity=commodity)
    if sma is True:
        for p in periods:
            addSma(dataframe, colum_name, p, commodity)
    if wma is True:
        for p in periods:
            addWma(dataframe, colum_name, p, commodity)


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
            talib.RSI(dataframe['{}_Close'.format(commodity)], timeperiod=p)


def addRoc(dataframe: pd.DataFrame,
           time_periods: List[int],
           commodity: str):
    for p in time_periods:
        dataframe['{}_ROC_{}'.format(commodity, p)] = \
            talib.ROC(dataframe['{}_Close'.format(commodity)], timeperiod=p)


def addTrix(dataframe: pd.DataFrame,
            time_periods: List[int],
            commodity: str):
    for p in time_periods:
        dataframe['{}_TRIX_{}'.format(commodity, p)] = \
            talib.TRIX(dataframe['{}_Close'.format(commodity)], timeperiod=p)


def addRocr(dataframe: pd.DataFrame,
            time_periods: List[int],
            commodity: str):
    for p in time_periods:
        dataframe['{}_ROCR_{}'.format(commodity, p)] = \
            talib.ROCR(dataframe['{}_Close'.format(commodity)], timeperiod=p)


def addSinglePeriodFinFeat(dataframe: pd.DataFrame,
                           time_periods: List[int],
                           commodity: str,
                           atr=True,
                           adx=True,
                           cci=True,
                           willR=True,
                           rsi=True,
                           roc=True,
                           trix=True,
                           rocr=True):
    if atr is True:
        addAtr(dataframe, time_periods, commodity)
    if adx is True:
        addAdx(dataframe, time_periods, commodity)
    if cci is True:
        addCci(dataframe, time_periods, commodity)
    if willR is True:
        addWillR(dataframe, time_periods, commodity)
    if rsi is True:
        addRsi(dataframe, time_periods, commodity)
    if roc is True:
        addRoc(dataframe, time_periods, commodity)
    if trix is True:
        addTrix(dataframe, time_periods, commodity)
    if rocr is True:
        addRocr(dataframe, time_periods, commodity)


def addStochSlow(dataframe: pd.DataFrame,
                 commodity: str,
                 fastk_period=5,
                 slowk_period=3,
                 slowk_matype=0,
                 slowd_period=3,
                 slowd_matype=0):
    dataframe['{}_Slowk'.format(commodity)], dataframe['{}_Slowd'.format(commodity)] = \
        talib.STOCH(dataframe['{}_High'.format(commodity)].values,
                    dataframe['{}_Low'.format(commodity)].values,
                    dataframe['{}_Close'.format(commodity)].values,
                    fastk_period, slowk_period, slowk_matype,
                    slowd_period, slowd_matype)


def addStochFast(dataframe: pd.DataFrame,
                 commodity: str,
                 fastk_period=5,
                 fastd_period=3,
                 fastd_matype=0):
    dataframe['{}_Fastk'.format(commodity)], dataframe['{}_Fastd'.format(commodity)] = \
        talib.STOCHF(dataframe['{}_High'.format(commodity)].values,
                     dataframe['{}_Low'.format(commodity)].values,
                     dataframe['{}_Close'.format(commodity)].values,
                     fastk_period, fastd_period, fastd_matype)


def addUltOsc(dataframe: pd.DataFrame,
              commodity: str,
              timeperiod1=7,
              timeperiod2=14,
              timeperiod3=28):
    """
    adds ultimate oscillator to dataset
    """
    dataframe['{}_ULTOSC'.format(commodity)] = talib.ULTOSC(
        dataframe['{}_High'.format(commodity)].values,
        dataframe['{}_Low'.format(commodity)].values,
        dataframe['{}_Close'.format(commodity)].values,
        timeperiod1, timeperiod2, timeperiod3
    )


def addWeekDay(df: pd.DataFrame, date_column: str):
    df['day_of_week'] = df.apply(lambda x: x[date_column].weekday(), axis=1)


# Todo - to be tested
def addBBands(df: pd.DataFrame,
              commodity_name: str,
              period: int,
              stdNbr: int = 2):
    try:
        close = df['{}_Close'.format(commodity_name)]
    except Exception:
        print('could not define column for calculating BBands for')
        return None

    # df['{}_upperBB'.format(commodity_name)],
    # df['{}_middleBB'.format(commodity_name)],
    # df['{}_lowerBB'.format(commodity_name)] = \
    #     talib.BBANDS(close.values, timeperiod=period)
    upper, middle, lower = talib.BBANDS(
        close.values,
        timeperiod=period,
        # number of non-biased standard deviations from the mean
        nbdevup=stdNbr,
        nbdevdn=stdNbr,
        # Moving average type: simple moving average here
        matype=0)
    df['{}_upperBB'.format(commodity_name)] = upper
    df['{}_middleBB'.format(commodity_name)] = middle
    df['{}_lowerBB'.format(commodity_name)] = lower


    # data = dict(upper=upper, middle=middle, lower=lower)
    # df_temp = pd.DataFrame(data, index=df.index, columns=[
    #     '{}_upperBB'.format(commodity_name),
    #     '{}_middleBB'.format(commodity_name),
    #     '{}_lowerBB'.format(commodity_name)]).dropna()

    # df = pd.concat([df, df_temp], axis=1)
