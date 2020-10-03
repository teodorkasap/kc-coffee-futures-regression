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
