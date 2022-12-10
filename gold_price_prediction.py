# -*- coding: utf-8 -*-
"""Gold_Price_Prediction.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # to plot the time series plot
from sklearn import metrics # for the evalution
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

data = pd.read_excel('goldpricee.xlsx')
data.head()

data.isna().sum()

data[data.isnull().any(axis=1)]

data.dropna(axis=0, inplace=True)
data.head()

data.info()

for col in data.columns:
  print("_"*40)
  print("Column Name:", col)
  print(data[col].value_counts())

data['Month'] = data['Date'].apply(lambda x: str(x).split(" ")[0])
data['Day'] = data['Date'].apply(lambda x: str(x).split(" ")[1][:-1]).astype(int)
data['Year'] = data['Date'].apply(lambda x: str(x).split(" ")[2]).astype(int)
data.head()

data[data.Month=='Jul']

months = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr':4,
         'May':5,
         'Jun':6,
         'Jul':7,
         'Aug':8,
         'Sep':9,
         'Oct':10,
         'Nov':11,
         'Dec':12
        }

data['Month'] = data['Month'].map(months)

data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.head()

data.set_index(['Date'], drop=False, inplace=True)

data.sort_index(ascending=True, axis=0, inplace=True)

data.reset_index(drop=True, inplace=True)
data.head()

"""## Data Visulization"""

# Plot: Heat Map

correlation = data.corr()             # pairwise correlation
sns.set(rc = {'figure.figsize':(10,10)})
sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot = True)

data.plot(x="Year", y=["Price"], figsize=(12,8))

"""### Stationarity Test : KPSS"""

from statsmodels.tsa.stattools import kpss

def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

kpss_test(data.set_index("Date")['Open'])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(15,8), 'figure.dpi':440})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)

axes[0, 0].plot(data['Open'])
axes[0, 0].set_title('Original Series')
plot_acf(data['Open'], ax=axes[0, 1])


# 1st Differencing
axes[1, 0].plot(data['Open'].diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(data['Open'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(data['Open'].diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(data['Open'].diff().diff().dropna(), ax=axes[2, 1])

plt.show()

data_subset = data[(data['Date']<='2016-12-31') & (data['Date']<='2013-01-01')]

from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(data_subset.set_index("Date")['Open'], period=1).plot()

seasonal_decompose(data_subset.set_index("Date")['Open'], period=2).plot()

plot_acf(seasonal_decompose(data.set_index("Date")['Open'], period=2).resid.fillna(method='ffill').fillna(method='bfill'))
plt.show()

plot_acf(seasonal_decompose(data.set_index("Date")['Open'], period=2).resid.fillna(method='ffill').fillna(method='bfill'), lags=7)
plt.show()

plot_pacf(seasonal_decompose(data.set_index("Date")['Open'], period=2).resid.fillna(method='ffill').fillna(method='bfill'))
plt.show()

plot_pacf(seasonal_decompose(data.set_index("Date")['Open'], period=2).resid.fillna(method='ffill').fillna(method='bfill'), lags=7)
plt.show()

data.tail(20)

df_train = data[data['Date']<"2022-10-19"]
df_test = data[data['Date']>="2022-10-19"]

from statsmodels.tsa.statespace.sarimax import SARIMAX
# 1,1,2 SARIMA Model
model = SARIMAX(df_train.set_index('Date')['Open'], order=(0,2,2), seasonal_order=(0,1,3,12))
model_fit_1 = model.fit()
print(model_fit_1.summary())

from statsmodels.tsa.statespace.sarimax import SARIMAX
# 1,1,2 SARIMA Model
model = SARIMAX(df_train.set_index('Date')['Chg%'], order=(0,2,3), seasonal_order=(1,1,3,12))
model_fit_2 = model.fit()
print(model_fit_2.summary())

model_fit_1.plot_diagnostics()
plt.show()

print(model_fit_1.forecast(df_test.shape[0]))

df_test['forecast_Open'] = model_fit_1.forecast(df_test.shape[0])
df_test.head()

plt.figure(figsize=(12,6))
plt.plot(df_test['Date'], df_test['Open'], 'b-', label='original')
plt.plot(df_test['Date'], df_test['forecast_Open'], 'g-', label='forecast')
plt.xlabel("Date")
plt.ylabel("Chg% Price")
plt.legend()
plt.show()

plt.figure(figsize=(16,9))
fig = px.line(data, x=data.Date, y='Open', title='Open')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

""## LSTM ##""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # to plot the time series plot
from sklearn import metrics # for the evalution
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

data = pd.read_excel('goldpricee.xlsx')
data.head()

data.dropna(axis=0, inplace=True)
data.head()

data['Month'] = data['Date'].apply(lambda x: str(x).split(" ")[0])
data['Day'] = data['Date'].apply(lambda x: str(x).split(" ")[1][:-1]).astype(int)
data['Year'] = data['Date'].apply(lambda x: str(x).split(" ")[2]).astype(int)
data.head()

months = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr':4,
         'May':5,
         'Jun':6,
         'Jul':7,
         'Aug':8,
         'Sep':9,
         'Oct':10,
         'Nov':11,
         'Dec':12
        }

data['Month'] = data['Month'].map(months)

data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.head()

data.set_index(['Date'], inplace=True)

data.sort_index(axis=0, ascending=True, inplace=True)
data.head()

data.reset_index(inplace=True)
data.head()

final_data = data[['Date', 'Price', 'Open', 'High', 'Low', 'Chg%']]
final_data.head()

df_for_training, df_for_testing = train_test_split(final_data, test_size=0.2, shuffle=False)

final_data.shape

train_dates = df_for_training['Date']
df_for_training = df_for_training.drop(['Date'], axis=1).astype(float)

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

df_for_training_scaled.shape

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
print(len(trainX), len(trainY))

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

final_data.shape[0]-df_for_training.shape[0]+1

n_past = 16
n_days_for_prediction=15 

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='1d').tolist()
print(predict_period_dates)

prediction = model.predict(trainX[-n_days_for_prediction:])

prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Price':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

original = final_data[['Date', 'Price']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[(original['Date'] >= '2020-7-22') &(original['Date'] <= '2020-08-05')]

sns.lineplot(original['Date'], original['Price'], label='Original')
sns.lineplot(df_forecast['Date'], df_forecast['Price'], label='Forecast')

inr = pd.read_excel('inr (1).xlsx')
inr.head()

inr.plot(x="Year", y=["USD INR"], figsize=(8,5))

inr.plot(x="Year", y=["Price1"], figsize=(8,5))

print(inr['Price1'].corr(inr['USD INR']))

