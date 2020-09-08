
# %% imports
from sklearn import metrics
from os import posix_fallocate
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# %% - get data
df = pd.read_csv(
    'coffee-cocoa-sugar-commitment-coffee-price-data-saved03082020.csv')

df['Date'] = pd.to_datetime(
    df['Coffee Date'], format='%Y-%m-%d', errors='coerce')

# %% - check df
df.dtypes
df.describe()
df
# %% - training set
dataset = df.iloc[:, 1:2].values
training_set = dataset[:422]
test_set = dataset[528:]
val_set = dataset[422:528]

training_set
test_set
val_set


# %% - scale
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
val_set_scaled = sc.transform(val_set)
test_set_scaled = sc.transform(test_set)


# %% - create method to generate sequences

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-seq_size):
        # print(i)
        window = obs[i:(i+seq_size)]
        after_window = obs[i+seq_size]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        x = np.reshape(obs, (obs.shape[0], obs.shape[1], 1))

    return x, y

# %% - generate train / val / tes sequences


SEQUENCE_SIZE = 10
x_train, y_train = to_sequences(SEQUENCE_SIZE, training_set_scaled)
x_val, y_val = to_sequences(SEQUENCE_SIZE, val_set_scaled)
x_test, y_test = to_sequences(SEQUENCE_SIZE, test_set_scaled)

# %% - check all three sets
print("Shape of training set: {}".format(x_train.shape))
print("Shape of test set: {}".format(x_val.shape))
print("Shape of training set: {}".format(x_test.shape))

# %% - check train set
x_train

# %% - build and compile model

print('Build model...')
model = Sequential()
model.add(LSTM(64, dropout=0.0, recurrent_dropout=0.0, input_shape=(None, 1)))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5,
                        verbose=1, mode='auto', restore_best_weights=True)


# %% - train model
print('Train...')

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          callbacks=[monitor], verbose=2, epochs=1000)


# %% - evaluate model


pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred, y_test))
print("Score (RMSE): {}".format(score))

# %% - plot losses
losses = pd.DataFrame(model.history.history)
losses.plot()
