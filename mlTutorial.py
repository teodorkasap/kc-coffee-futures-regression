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

# own modules
from prepUsdBrlData import getUsdBrlData

# %% - get usd / brl data
df_exch = getUsdBrlData()

# %% - get KC data
df = pd.read_csv(
    'KCK21.NYB.csv')

# %% - drop nan
df = df.dropna()



# %% - get rid of rows with vol "0"
# df = df[df['Volume']!=0]

# %% - change date to datetime

df['Date'] = pd.to_datetime(
    df['Date'], format='%Y-%m-%d', errors='coerce')

# %% - merge two dataframes on KC data dates
df = pd.merge(left=df, right=df_exch, left_on='Date', right_on='Date')
df

# %% - set data column as index

df = df.set_index('Date')

# %% - plot closing price
df['Close'].plot(grid=True)
plt.title('KCK21 closing prices')
plt.ylabel('price $')
plt.show()


# %% - calculate Simple Moving Averages
def add_SMA(dataframe, colum_name,  period):
    dataframe['SMA_{}'.format(period)] = dataframe[colum_name].rolling(
        window=period).mean()


add_SMA(df, 'Close', 10)
add_SMA(df, 'Close', 20)
add_SMA(df, 'Close', 50)
add_SMA(df, 'Close', 100)
add_SMA(df, 'Close', 200)

# %% - calculate Exponential Moving Averages


def add_EMA(dataframe, colum_name,  period):
    dataframe['EMA_{}'.format(period)] = ta.EMA(
        dataframe, timeperiod=period, price=colum_name)


add_EMA(df, 'Close', 10)
add_EMA(df, 'Close', 20)
add_EMA(df, 'Close', 50)
add_EMA(df, 'Close', 100)
add_EMA(df, 'Close', 200)

# %% - calculate Average True Range

df['ATR'] = talib.ATR(df['High'].values, df['Low'].values,
                      df['Close'].values, timeperiod=14)

# %% - calculate Average Directional Index
df['ADX'] = talib.ADX(df['High'].values, df['Low'].values,
                      df['Close'].values, timeperiod=14)

# %% - calculate Commodity Channel Index
df['CCI'] = talib.CCI(df['High'].values, df['Low'].values,
                      df['Close'].values, timeperiod=14)

# %% - calculate Rate Of Change
df['ROC'] = talib.ROC(df['Close'], timeperiod=10)

# %% - calculate RElative Strenght Index
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

# %% - calculate William's %R
df['Williams_%R'] = talib.ATR(df['High'].values, df['Low'].values,
                              df['Close'].values, timeperiod=14)

# %% - stochastic K%D
df['Slowk'], df['Slowd'] = talib.STOCH(df['High'].values,
                                       df['Low'].values,
                                       df['Close'].values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

# %%- get rid of nan

df = df.dropna()

# %% - add prediction column

df['Prediction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df['Prediction']

# %% - get shape
df.shape

# %% - get train and test sets

df_train = df.iloc[:314]
df_test = df.iloc[314:]

df_train_X = df_train.drop(columns=['Prediction'])
df_test_x = df_test.drop(columns=['Prediction'])

df_train_y = df_train['Prediction']
df_test_y = df_test['Prediction']

# %% - Normalize data

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(df_train_X)
X_test_scaled = scaler.transform(df_test_x)


# %% - Classification models

classification_models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=5000),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(gamma="auto"),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Neural Net": MLPClassifier(solver='adam', alpha=0.001, learning_rate='constant', learning_rate_init=0.001),
    "Naive Bayes": GaussianNB()
}

no_classifiers = len(classification_models.keys())

# %% - batch training and results


def batch_classify(df_train_scaled, df_test, verbose=True):
    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers, 3)),
                              columns=['classfier', 'train_score', 'training_time'])
    count = 0
    for key, classifier in classification_models.items():
        t_start = time.process_time()
        classifier.fit(X_train_scaled, df_train_y)
        t_end = time.process_time()
        elapsed_time = t_end - t_start
        train_score = classifier.score(X_train_scaled, df_train_y)
        df_results.loc[count, 'classfier'] = key
        df_results.loc[count, 'train_score'] = train_score
        df_results.loc[count, 'training_time'] = elapsed_time
        if verbose:
            print("trained {c} in {f:.2f}s".format(c=key, f=elapsed_time))
        count += 1
    return df_results


# %% - train models

df_results = batch_classify(X_train_scaled, df_train_y)
print(df_results.sort_values(by='train_score', ascending=True))


# %% - Logistic Reg.

# model = LogisticRegression(solver='liblinear', max_iter=5000)
# model.fit(X_train_scaled,df_train_y)
# model

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y,predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y,predictions))


# %% - random forest

model = RandomForestClassifier(n_estimators=1000, min_samples_leaf=1)
model.fit(X_train_scaled, df_train_y)

predictions = model.predict(X_test_scaled)
print("accuracy score: ")
print(accuracy_score(df_test_y, predictions))
print("confusion matrix: ")
print(confusion_matrix(df_test_y, predictions))
print("classification report")
print(classification_report(df_test_y, predictions))

# %% - ROC curve
pred_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(df_test_y, pred_prob)
roc_auc = auc(fpr, tpr)

print("roc auc is:" + str(roc_auc))
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr)
plt.xlabel('False pos. rate')
plt.ylabel('True pos. rate')
plt.show()


# %% - gradient boosting

# model = GradientBoostingClassifier()
# model.fit(X_train_scaled,df_train_y)
# model

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y,predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y,predictions))

# %% - naive bayes

# model = GaussianNB()
# model.fit(X_train_scaled,df_train_y)
# model

# predictions = model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y,predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y,predictions))


# =======================================================
# %% - pick the most accurate and test
# knn_model = KNeighborsClassifier()
# knn_model.fit(X_train_scaled,df_train_y)

# # %% - predict on test set
# predictions = knn_model.predict(X_test_scaled)
# print("accuracy score: ")
# print(accuracy_score(df_test_y,predictions))
# print("confusion matrix: ")
# print(confusion_matrix(df_test_y,predictions))
