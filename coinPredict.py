import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
sns.set(style="darkgrid", font_scale=1.5)
%matplotlib inline
CoinName='BTC'

def get_crypto_price(symbol, exchange, start_date = None):
    api_key = 'YOUR API KEY'
    api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    df = df.rename(columns = {'1a. open (USD)': 'open', '2a. high (USD)': 'high', '3a. low (USD)': 'low', '4a. close (USD)': 'close', '5. volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(['1b. open (USD)', '2b. high (USD)', '3b. low (USD)', '4b. close (USD)', '6. market cap (USD)'], axis = 1)
    if start_date:
        df = df[df.index >= start_date]
    return df

df = get_crypto_price(symbol = CoinName, exchange = 'USD', start_date = '2020-01-01')

df = df[["open"]]

data = df.iloc[:, 0]
hist = []
target = []
length = 90
for i in range(len(data)-length+1):
    if (i==len(data)-length):
        x = data[i:i+length]
        hist.append(x)
        print('aaa')
    else:
        x = data[i:i+length]
        y = data[i+length]
        hist.append(x)
        target.append(y)
    
hist = np.array(hist)
target = np.array(target)
target = target.reshape(-1,1)
    
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
hist_scaled = sc.fit_transform(hist)
target_scaled = sc.fit_transform(target)
hist_scaled = hist_scaled.reshape((len(hist_scaled), length, 1))


X_train = hist_scaled
y_train = target_scaled

import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.LSTM(units=32, return_sequences=True,
                  input_shape=(90,1), dropout=0.2))
model.add(layers.LSTM(units=32, return_sequences=True,
                  dropout=0.2))
model.add(layers.LSTM(units=32, dropout=0.2))
model.add(layers.Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train[0:len(X_train)-1], y_train, epochs=10, batch_size=32)
#Tenho que revisar essa predição
pred = model.predict(X_train[len(X_train)-1:len(X_train)])

pred_transformed = sc.inverse_transform(pred)

print(pred_transformed)