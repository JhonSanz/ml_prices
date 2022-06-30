import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')
df = web.DataReader('^GSPC',data_source='yahoo',start='2012-01-01',end='2022-06-29')
data = df[["Close"]]
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
scaler = MinMaxScaler(feature_range=(0,1))


def get_x_train(train_data):
    """ Fills len(train_data) long arrays with 60 records 
        in each position
        example = [[1...60], [1...60], ...]
    """
    x_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
    return np.array(x_train)

def create_model(x_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    return model

scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]
x_train = get_x_train(train_data)
"""
    example = [[1...60], [1...60], ...]; shape = (len(), 60)
    reshaped = [[[1]...[60]], [[1]...[60]], ...]; shape = (len(), 60, 1)
"""
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.array(train_data[60:len(train_data), 0])

model = create_model(x_train)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

#create the testing data set
test_data = scaled_data[training_data_len-60:, :]
x_test = get_x_train(test_data)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = dataset[training_data_len:, :]
  

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt( np.mean((predictions - y_test)**2))


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()

df = web.DataReader('^GSPC',data_source='yahoo',start='2012-01-01',end='2022-06-29')
data = df[["Close"]]
last_60_days = data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

x_test_new = []
x_test_new.append(last_60_days_scaled)
x_test_new = np.array(x_test_new)
x_test_new = np.reshape(x_test_new, (x_test_new.shape[0], x_test_new.shape[1], 1))
prediction_new = model.predict(x_test_new)
prediction_new = scaler.inverse_transform(prediction_new)
print(prediction_new)

df = web.DataReader('^GSPC',data_source='yahoo',start='2022-06-29', end='2022-06-29')
print(df["Close"])