# %% - imports
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - get data
df = pd.read_csv(
    'coffee-cocoa-sugar-commitment-coffee-price-data-saved03082020.csv')

df.dtypes
df.describe()

# %% - change date to datetime

df['Date'] = pd.to_datetime(
    df['Coffee Date'], format='%Y-%m-%d', errors='coerce')

# %% - set data column as index

df = df.set_index('Date')

# %% - drop unnecessary columns

df = df.drop(['Coffee Date',
              'coffee comm. Net_Position',
              'cocoa comm. Net_Position',
              'sugar comm. Net_Position',
              'Coffee Vol.'	,
              'Coffee Price Change%',
              'Coffee Price Change% shifted',
              'Coffee Price shifted'], axis=1)

# %% - check if any null values
print("checking if any null values are present\n", df.isna().sum())


# %% - plot df

plt.figure(figsize=(18, 9))
plt.plot(df['Coffee Price'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Coffee Price', fontsize=18)
plt.show()

# %% - train test split

test_data_perc = 1-0.2
cutoff = int(df.shape[0]*test_data_perc)

train_data = df[:cutoff]
test_data = df[cutoff:]

train_data.shape
test_data.shape

# %% - scale

scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)

# %% - generate time series
length = 1
generator = TimeseriesGenerator(
    scaled_train, scaled_train, length=length, batch_size=1)

# %%

X, y = generator[0]
X
y
scaled_train

# %% - import ts modules

# %% - create model
n_features = 1
model = Sequential()
model.add(LSTM(60, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# %% - create early stopping mechanism

early_stop = EarlyStopping(monitor='val_loss', patience=2)

# %% - create validation generator
validation_generator = TimeseriesGenerator(
    scaled_test, scaled_test, length=length, batch_size=1)

# %% - train model
# model.fit_generator(generator, epochs=20,
#                     validation_data=validation_generator, callbacks=[early_stop])
model.fit_generator(generator, epochs=4,
                    validation_data=validation_generator)

# %% - check out losses

losses = pd.DataFrame(model.history.history)
losses.plot()

# %% check against test set

test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test_data)):

    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [
                              [current_pred]], axis=1)

true_predictions = scaler.inverse_transform(test_predictions)

test_data['Predictions'] = true_predictions


# %% - 

test_data
test_data.plot(figsize =(12,8))
# %% - check out this tutorial: https://www.youtube.com/watch?v=6f67zrH-_IE
