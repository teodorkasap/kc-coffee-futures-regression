# %% -source
# tutorial link here: https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f

# %% - imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

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
test_set_scaled = sc.transform(test_set)

# %% data with timesteps
timestep = 1


def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(timestep, len(obs)):
        x.append(obs[i-timestep:i])
        y.append(obs[i])

    # for i in range (len(obs)-seq_size):
    #     past_interval = obs[i:(i+seq_size)]
    #     fut_value = obs[i+seq_size]
    #     # fut_value = obs.shift(-seq_size)
    #     print(past_interval)
    #     # past_interval = [[x] for x in past_interval]
    #     x.append(past_interval)
    #     y.append(fut_value)

    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y

# %% - x_train, y_train / val / test


X_train, y_train = to_sequences(timestep, training_set_scaled)
x_val, y_val = to_sequences(timestep, val_set_scaled)
x_test, y_test = to_sequences(timestep, test_set_scaled)


# %% - build model
model = Sequential()  # Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
# Adding a second LSTM layer and some Dropout regularisation
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
# Adding a third LSTM layer and some Dropout regularisation
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))  # Adding the output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')


# %% - early stopping mechanism

early_stop = EarlyStopping(monitor='val_loss', patience=2)

# %% - train model
model.fit(X_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val), callbacks=[early_stop])


# %% - test data prediction
predicted_test_data = model.predict(x_test)
predicted_test_data = sc.inverse_transform(predicted_test_data)


# %% plot

plt.plot(test_set[timestep:], color='black', label='test data')
plt.plot(predicted_test_data, color='green', label='predicted test data')
plt.title('predicted test data')
plt.xlabel('Time')
plt.ylabel('Coffee Price')
plt.legend()
plt.show()

# %% - compare actual and predicted price movements

test_set_comparison = test_set[timestep:]
test_set_comparison.shape
predicted_test_data.shape

accurate_prediction_count = 0

for i in range(1, len(test_set_comparison)):
    if (test_set_comparison[i-1]-test_set_comparison[i] < 0) and (predicted_test_data[i-1] - predicted_test_data[i] < 0):
        accurate_prediction_count += 1
    if (test_set_comparison[i-1]-test_set_comparison[i] > 0) and (predicted_test_data[i-1] - predicted_test_data[i] > 0):
        accurate_prediction_count += 1


pred_accuracy_percentage = accurate_prediction_count/len(predicted_test_data)
print(pred_accuracy_percentage)
