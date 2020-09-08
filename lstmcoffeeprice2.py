# %% imports
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

# %% - scale
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
val_set_scaled = sc.transform(val_set)


# %% data with timesteps
timestep = 20

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range (len(obs)-seq_size):
        past_interval = obs[i:(i+seq_size)]
        fut_value = obs[i+seq_size]
        # fut_value = obs.shift(-seq_size)
        print(past_interval)
        # past_interval = [[x] for x in past_interval]
        x.append(past_interval)
        y.append(fut_value)

    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x,y


X_train, y_train = to_sequences(timestep,training_set_scaled)
x_val, y_val = to_sequences(timestep,val_set_scaled)




# %%  - create model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
# model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))

model.add(LSTM(units=50))
# model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# %% - early stopping mechanism

early_stop = EarlyStopping(monitor='val_loss', patience=2)

# %% - scale test set
test_set_scaled = sc.fit_transform(test_set)

# %% - make X_test and y_test

x_test, y_test = to_sequences(timestep,test_set_scaled)


# %% - train model
model.fit(X_train, y_train, validation_data=(x_val, y_val),
          epochs=100, batch_size=1, callbacks=[early_stop])

# %% - training predicion
predicted_train_data = model.predict(X_train)
predicted_train_data = sc.inverse_transform(predicted_train_data)

# %% - plot

plt.plot(training_set, color='black', label='train data')
plt.plot(predicted_train_data, color='green', label='predicted train data')
plt.title('predicted train data')
plt.xlabel('Time')
plt.ylabel('Coffee Price')
plt.legend()
plt.show()


# %% - test data prediction
predicted_test_data = model.predict(x_test)
predicted_test_data = sc.inverse_transform(predicted_test_data)


# %% plot

plt.plot(test_set, color='black', label='test data')
plt.plot(predicted_test_data, color='green', label='predicted test data')
plt.title('predicted test data')
plt.xlabel('Time')
plt.ylabel('Coffee Price')
plt.legend()
plt.show()

# %% check out this tutorial https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_10_2_lstm.ipynb
# Todo: Make changes as necessary