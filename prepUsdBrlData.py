# %% - imports
import time
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import talib.abstract as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression



# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# %% - methods

def getUsdBrlData(filepath:str):

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    df.columns
    columns = ['Date', 'USD_Close',	'USD_Open', 'USD_High', 'USD_Low', 'USD_Change %']
    df.columns = columns
    df
    return df


def add_SMA(dataframe, colum_name,  period, commodity):
    dataframe['{}_SMA_{}'.format(commodity,period)] = dataframe[colum_name].rolling(
        window=period).mean()

def add_EMA(dataframe, colum_name,  period,commodity):
    dataframe['{}_EMA_{}'.format(commodity,period)] = ta.EMA(
        dataframe, timeperiod=period, price=colum_name)



# %% - get data

df = getUsdBrlData(filepath)
df = df.dropna()

df['Date'] = pd.to_datetime(
    df['Date'], format='%Y-%m-%d', errors='coerce')

df = df.set_index('Date')

# %% - feature engineering

def getUsdPredictions(df):


    df = df.dropna()

    df['Date'] = pd.to_datetime(
        df['Date'], format='%Y-%m-%d', errors='coerce')

    df = df.set_index('Date')

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

    df = df.dropna()

    df['Prediction'] = np.where(df['USD_Close'].shift(-1) > df['USD_Close'], 1, 0)
    df['Prediction']

    df_train = df.iloc[:396]
    df_test = df.iloc[396:]

    df_train_X = df_train.drop(columns=['Prediction'])
    df_test_x = df_test.drop(columns=['Prediction'])

    df_train_y = df_train['Prediction']
    df_test_y = df_test['Prediction']

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(df_train_X)
    X_test_scaled = scaler.transform(df_test_x)

    log_model = LogisticRegression(solver='lbfgs', max_iter=5000)
    log_model.fit(X_train_scaled,df_train_y)
    predictions_train = log_model.predict(X_train_scaled)
    predictions_test = log_model.predict(X_test_scaled)
    predictions = np.concatenate((predictions_train,predictions_test),axis=0)
    return predictions



# %% - Classification models

# classification_models = {
#     "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=5000),
#     "Nearest Neighbors": KNeighborsClassifier(),
#     "Support Vector Machine": SVC(gamma="auto"),
#     "Gradient Boosting Classifier": GradientBoostingClassifier(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=100),
#     "Neural Net": MLPClassifier(solver='adam', alpha=0.001, learning_rate='constant', learning_rate_init=0.001),
#     "Naive Bayes": GaussianNB()
# }

# no_classifiers = len(classification_models.keys())

# log_model = LogisticRegression(solver='lbfgs', max_iter=5000)
# log_model.fit(X_train_scaled,df_train_y)



# # %% - batch training and results


# def batch_classify(df_train_scaled, df_test, verbose=True):
#     df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers, 3)),
#                               columns=['classfier', 'train_score', 'training_time'])
#     count = 0
#     for key, classifier in classification_models.items():
#         t_start = time.process_time()
#         classifier.fit(X_train_scaled, df_train_y)
#         t_end = time.process_time()
#         elapsed_time = t_end - t_start
#         train_score = classifier.score(X_train_scaled, df_train_y)
#         df_results.loc[count, 'classfier'] = key
#         df_results.loc[count, 'train_score'] = train_score
#         df_results.loc[count, 'training_time'] = elapsed_time
#         if verbose:
#             print("trained {c} in {f:.2f}s".format(c=key, f=elapsed_time))
#         count += 1
#     return df_results


# # %% - train models

# df_results = batch_classify(X_train_scaled, df_train_y)
# print(df_results.sort_values(by='train_score', ascending=True))

# # %% - Knn model
# # knn_model = KNeighborsClassifier()
# # knn_model.fit(X_train_scaled,df_train_y)

# # # predict on test set
# # predictions = knn_model.predict(X_test_scaled)
# # print("accuracy score: ")
# # print(accuracy_score(df_test_y,predictions))
# # print("confusion matrix: ")
# # print(confusion_matrix(df_test_y,predictions))


# # # %% - Neural Net
# # mlp_model = MLPClassifier(solver='adam', alpha=0.001, learning_rate='constant', learning_rate_init=0.001)
# # mlp_model.fit(X_train_scaled,df_train_y)

# # # predict on test set
# # predictions = mlp_model.predict(X_test_scaled)
# # print("accuracy score: ")
# # print(accuracy_score(df_test_y,predictions))
# # print("confusion matrix: ")
# # print(confusion_matrix(df_test_y,predictions))

# # %% - Logistic Regresssion
# log_model = LogisticRegression(solver='lbfgs', max_iter=5000)
# log_model.fit(X_train_scaled,df_train_y)

# # predict on test set
# predictions_train = log_model.predict(X_train_scaled)
# predictions_test = log_model.predict(X_test_scaled)

# print("accuracy score: ")
# print(accuracy_score(df_test_y,predictions_test))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y,predictions_test))

# # %%
# predictions_test