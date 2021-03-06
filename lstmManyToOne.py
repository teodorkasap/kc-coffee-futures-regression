# %% - imports
from prepUsdBrlData import getUsdBrlData
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import talib
import talib.abstract as ta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - predict usd prices


# %% - get usd / brl data
df_exch = getUsdBrlData()

# %% - get KC data
df = pd.read_csv(
    'KCK21.NYB.csv')

# %% - drop nan
df = df.dropna()


# %% - get rid of rows with vol "0"
df = df.drop(['KC_Volume'], axis=1)

# %% - change date to datetime

df['Date'] = pd.to_datetime(
    df['Date'], format='%Y-%m-%d', errors='coerce')

# %% - merge two dataframes on KC data dates
df = pd.merge(left=df, right=df_exch, left_on='Date', right_on='Date')


# %%- get exch rate prediction


# %% - set data column as index

df = df.set_index('Date')

# %% - plot closing price
df['KC_Close'].plot(grid=True)

plt.title('KCK21 closing prices')
plt.ylabel('price $')
plt.show()

df['USD_Close'].plot(grid=True)
plt.title('USD/BRL closing prices')
plt.ylabel('BRL')
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


# %% - calculate Exponential Moving Averages


def add_EMA(dataframe, colum_name,  period, commodity):
    dataframe['{}_EMA_{}'.format(commodity, period)] = ta.EMA(
        dataframe, timeperiod=period, price=colum_name)


add_EMA(df, 'KC_Close', 10, "KC")
add_EMA(df, 'KC_Close', 20, "KC")
add_EMA(df, 'KC_Close', 50, "KC")
add_EMA(df, 'KC_Close', 100, "KC")
add_EMA(df, 'KC_Close', 200, "KC")


# %% - calculate Average True Range

# df['KC_ATR_14'] = talib.ATR(df['KC_High'].values, df['KC_Low'].values,
#                             df['KC_Close'].values, timeperiod=14)

df['KC_ADX_14'] = talib.ADX(df['KC_High'].values, df['KC_Low'].values,
                            df['KC_Close'].values, timeperiod=14)

df['KC_CCI_14'] = talib.CCI(df['KC_High'].values, df['KC_Low'].values,
                            df['KC_Close'].values, timeperiod=14)

# df['KC_ROC_10'] = talib.ROC(df['KC_Close'], timeperiod=10)

# df['KC_RSI_14'] = talib.RSI(df['KC_Close'], timeperiod=14)

# df['KC_Williams_%R_14'] = talib.ATR(df['KC_High'].values, df['KC_Low'].values,
#                                     df['KC_Close'].values, timeperiod=14)

df['KC_Slowd'] = talib.STOCH(df['KC_High'].values,
                             df['KC_Low'].values,
                             df['KC_Close'].values,
                             fastk_period=5,
                             slowk_period=3,
                             slowk_matype=0,
                             slowd_period=3,
                             slowd_matype=0)[1]

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


df['Prediction'] = np.where(df['KC_Close'].shift(-1) > df['KC_Close'], 1, 0)
df['Prediction']

# %% - generic function to shape input dataframe for lstm model


def lstm_data_transform(x_data, y_data, num_steps=5):

    # Prepare the list for the transformed data
    X, y = list(), list()

    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps

    # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        # Get only the last element of the sequence for y
        seq_y = y_data[end_ix]
        # append sequences
        X.append(seq_X)
        y.append(seq_y)

    # Make final arrays and return them
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array


# %% - train test split
train_ind = int(0.8 * df.shape[0])
df_train = df[:train_ind]
df_train.shape
val_ind = int(0.8 * df_train.shape[0])
val_ind
df_val = df_train[val_ind:]
df_train = df_train[:val_ind]
df_test = df[train_ind:]
df_train.shape
df_val.shape


df_train_X = df_train.drop(['Prediction'], axis=1)
df_val_X = df_val.drop(['Prediction'], axis=1)
df_test_X = df_test.drop(['Prediction'], axis=1)


df_train_y = df_train['Prediction']
df_test_y = df_test['Prediction']
df_val_y = df_val['Prediction']

# %% - scaling training and validation sets
sc = MinMaxScaler(feature_range=(0, 1))
X_training_set_scaled = sc.fit_transform(df_train_X)
x_val_set_scaled = sc.transform(df_val_X)
x_test_set_scaled = sc.transform(df_test_X)

# %% - reshape training data for keras lstm

num_steps = 1
# training set
(x_train_transformed,
 y_train_transformed) = lstm_data_transform(X_training_set_scaled, df_train_y, num_steps=num_steps)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]

# %% -reshape validation data for keras lstm
(x_val_transformed,
 y_val_transformed) = lstm_data_transform(x_val_set_scaled, df_val_y, num_steps=num_steps)
assert x_train_transformed.shape[0] == y_train_transformed.shape[0]

# %% - reshape test data for keras lstm

# test set
(x_test_transformed,
 y_test_transformed) = lstm_data_transform(x_test_set_scaled, df_test_y, num_steps=num_steps)
assert x_test_transformed.shape[0] == y_test_transformed.shape[0]


# %% check shapes of all transformed dfs
print(x_train_transformed.shape, y_train_transformed.shape,
      x_test_transformed.shape, y_test_transformed.shape)

# %% - build model

# model = Sequential()
# model.add(LSTM(256, activation='tanh', input_shape=(x_train_transformed.shape[1], x_train_transformed.shape[2]), return_sequences=False))
# # model.add(Dense(units=20, activation='relu'))
# model.add(Dense(units=1, activation='sigmoid'))
# adam = Adam(lr=0.001)
# model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])
# model.summary()


# %% - early stopping callback

# early_stop = EarlyStopping(monitor='val_loss', patience=2)

# %% - model checkpoint
# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# %% - train model
# # history = model.fit(x_train_transformed, y_train_transformed, epochs=10, validation_data=(x_val_transformed,y_val_transformed), batch_size=5, verbose=2, callbacks=[early_stop])
# history = model.fit(x_train_transformed, y_train_transformed, epochs=100, validation_data=(x_val_transformed,y_val_transformed), batch_size=1, verbose=2, callbacks=[checkpoint])


# %% - mehtods to re-create model and load weights

def create_model():
    model = Sequential()
    model.add(LSTM(256, activation='tanh', input_shape=(
        x_train_transformed.shape[1], x_train_transformed.shape[2]), return_sequences=False))
    # model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)

# %% - load model

# model_generic = create_model()

# returns a compiled model
# identical to the previous one

# model1= model_generic.load_weights('weights-improvement-55-0.62.hdf5')
# model2 = model_generic.load_weights('weights-improvement-59-0.64.hdf5')


model1 = load_model('weights-improvement-55-0.62.hdf5')
model2 = load_model('weights-improvement-59-0.64.hdf5')

model1.summary()
model2.summary()


# %% - predictions

predictions1 = model1.predict(x_test_transformed)
predictions2 = model2.predict(x_test_transformed)

# %% - check predictions against actual data
df_pred_comparison1 = pd.DataFrame.from_records(
    predictions1, columns=['Predictions1'])
df_pred_comparison2 = pd.DataFrame.from_records(
    predictions2, columns=['Predictions2'])
df_pred_comparison1
df_pred_comparison2

# %% - make preds binary
# df_test_y

df_pred_comparison1['Pred_bin'] = np.where(
    df_pred_comparison1['Predictions1'] > 0.5, 1, 0)
df_test_y = pd.Series.to_frame(df_test_y)
df_test_y


df_pred_comparison1.shape
df_test_y = df_test_y.iloc[1:]
df_test_y.shape
df_test_y=df_test_y.set_index(df_pred_comparison1.index)
df_pred_comparison1['Actual_bin'] = pd.Series(df_test_y['Prediction'])
df_pred_comparison1

# %% get number of true pred
seriesObj = df_pred_comparison1.apply(lambda x: True if x['Actual_bin'] == x['Pred_bin'] else False , axis=1)

# %%
numOfRows = len(seriesObj[seriesObj == True].index)
numOfRows
len(df_pred_comparison1.index)