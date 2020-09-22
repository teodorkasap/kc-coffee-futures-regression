# %% - source
# https://medium.com/analytics-vidhya/stock-trend-prediction-with-technical-indicators-feature-engineering-and-python-code-1fa54d5806ba

# %% - imports
from prepUsdBrlData import getUsdBrlData
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import talib
import talib.abstract as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel


# This is for multiple print statements per cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% - predict usd prices


# %% - get usd / brl data
df_exch = getUsdBrlData()
columns_shift = ['USD_Close',	"USD_Open",	"USD_High",	"USD_Low",	"USD_Change %"]
# df_exch['USD_Close',	"USD_Open",	"USD_High",	"USD_Low",	"USD_Change %"] = df_exch[['USD_Close',	"USD_Open",	"USD_High",	"USD_Low",	"USD_Change %"]].shift(-1)

for column in columns_shift:
    df_exch[column] = df_exch[column].shift(-1)

df_exch

# %% - get KC data
df = pd.read_csv(
    'KCK21.NYB.csv')

# %% - drop nan
df = df.dropna()
columns_shift = []
for column in columns_shift:
    df[column] = df[column].shift(-1)

df

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


# %% -
# df['Prediction'] = np.where(df['USD_Close'].shift(-1) > df['USD_Close'], 1, 0)
# df['Prediction']

# %% - add prediction column

df['Prediction'] = np.where(df['KC_Close'].shift(-1) > df['KC_Close'], 1, 0)
df

# %%
df.columns

# %% shift kc related columns

columns_shift = ['KC_Open', 'KC_High', 'KC_Low', 'KC_Close', 'KC_Adj_Close','KC_SMA_10',
       'KC_SMA_20', 'KC_SMA_50', 'KC_SMA_100', 'KC_SMA_200', 'KC_EMA_10',
       'KC_EMA_20', 'KC_EMA_50', 'KC_EMA_100', 'KC_EMA_200', 'KC_ADX_14',
       'KC_CCI_14', 'KC_Slowd']
for column in columns_shift:
    df[column] = df[column].shift()

# %% - get shape
df=df.dropna()
df


# %% - get train and test sets

# cutoff = int(round((df.shape[0])*0.8))

# df_train = df.iloc[:cutoff]
# df_test = df.iloc[cutoff:]

# df_train.shape

# df_train_X = df_train.drop(['Prediction'], axis=1)
# df_test_x = df_test.drop(['Prediction'], axis=1)

# df_train_y = df_train['Prediction']
# df_test_y = df_test['Prediction']

# %% - Normalize data

# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(df_train_X)
# X_test_scaled = scaler.transform(df_test_x)


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

# %% - batch training and results


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


# %% - train models

# df_results = batch_classify(X_train_scaled, df_train_y)
# print(df_results.sort_values(by='train_score', ascending=True))


# %% - Naive Bayes

# model = GaussianNB()
# model.fit(X_train_scaled, df_train_y)
# model

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))


# %% - SVM Model
# model = SVC(kernel='linear', gamma='auto')
# model.fit(X_train_scaled, df_train_y)

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))


# %% - check df

corrMatrix = df.corr()
print (corrMatrix)

df.columns
# %% - check importance of features


# def f_importances(coef, names):
#     imp = coef
#     imp, names = zip(*sorted(zip(imp, names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.show()


# feature_names = ['KC_Open', 'KC_High', 'KC_Low', 'KC_Close', 'KC_Adj_Close', 'USD_Close',
#                  'USD_Open', 'USD_High', 'USD_Low', 'USD_Change %', 'KC_SMA_10',
#                  'KC_SMA_20', 'KC_SMA_50', 'KC_SMA_100', 'KC_SMA_200', 'KC_EMA_10',
#                  'KC_EMA_20', 'KC_EMA_50', 'KC_EMA_100', 'KC_EMA_200', 'KC_ADX_14',
#                  'KC_CCI_14', 'KC_Slowd', 'USD_ATR_14', 'USD_ATR_10', 'USD_ADX_14',
#                  'USD_ADX_10', 'USD_CCI_14', 'USD_CCI_10', 'USD_ROC_10', 'USD_ROC_5',
#                  'USD_RSI_14', 'USD_RSI_7', 'USD_Williams_%R_14', 'USD_Williams_%R_7',
#                  'USD_Slowk', 'USD_Slowd', 'USD_SMA_5', 'USD_SMA_10', 'USD_SMA_25',
#                  'USD_SMA_50', 'USD_SMA_100', 'USD_EMA_5', 'USD_EMA_10', 'USD_EMA_25',
#                  'USD_EMA_50', 'USD_EMA_100']
# f_importances(model.coef_[0], feature_names)


# %% - Logistic Reg.

# model = LogisticRegression(solver='liblinear', max_iter=5000, dual=True)
# model.fit(X_train_scaled, df_train_y)

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))


# %% - random forest

# model = RandomForestClassifier(n_estimators=1000, min_samples_leaf=1)
# model.fit(X_train_scaled, df_train_y)

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))
# print("classification report")
# print(classification_report(df_test_y, predictions))

# %% - ROC curve
# pred_prob = model.predict_proba(X_test_scaled)[:, 1]
# fpr, tpr, thresholds = roc_curve(df_test_y, pred_prob)
# roc_auc = auc(fpr, tpr)

# print("roc auc is:" + str(roc_auc))
# plt.plot([0, 1], [0, 1], "k--")
# plt.plot(fpr, tpr)
# plt.xlabel('False pos. rate')
# plt.ylabel('True pos. rate')
# plt.show()


# %% - gradient boosting

# model = GradientBoostingClassifier()
# model.fit(X_train_scaled, df_train_y)
# model

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))

# %% - naive bayes

# model = GaussianNB()
# model.fit(X_train_scaled, df_train_y)
# model

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))


# =======================================================
# %% - pick the most accurate and test
# knn_model = KNeighborsClassifier()
# knn_model.fit(X_train_scaled, df_train_y)

# # predict on test set
# predictions = knn_model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))


# %% -select best features for SVM
# df = df[['KC_Open', 'KC_CCI_14', 'USD_Slowk',
#          'KC_Low', 'KC_High', 'USD_CCI_14',
#          'USD_RSI_14', 'USD_ADX_14', 'USD_ADX_10',
#          'KC_ADX_14', 'USD_Open', 'USD_Williams_%R_14',
#          'USD_Williams_%R_7', 'USD_ATR_10', 'USD_ROC_10','Prediction',
#          'USD_ROC_5', 'USD_ATR_14']]


# %% - get train and test sets

# cutoff = int(round((df.shape[0])*0.8))

# df_train = df.iloc[:cutoff]
# df_test = df.iloc[cutoff:]

# df_train.shape

# df_train_X = df_train.drop(['Prediction'], axis=1)
# df_test_x = df_test.drop(['Prediction'], axis=1)

# df_train_y = df_train['Prediction']
# df_test_y = df_test['Prediction']

# # %% - Normalize data

# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(df_train_X)
# X_test_scaled = scaler.transform(df_test_x)


# # %% - build and train model
# model = SVC(kernel='linear', gamma='auto',probability=True)
# probas_ = model.fit(X_train_scaled, df_train_y).predict_proba(X_test_scaled)

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y, predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y, predictions))

# # %% - ROC curve
# pred_prob = model.predict_proba(X_test_scaled)[:, 1]
# fpr, tpr, thresholds = roc_curve(df_test_y, probas_[:, 1])
# roc_auc = auc(fpr, tpr)

# print("roc auc is:" + str(roc_auc))
# plt.plot([0, 1], [0, 1], "k--")
# plt.plot(fpr, tpr)
# plt.xlabel('False pos. rate')
# plt.ylabel('True pos. rate')
# plt.show()