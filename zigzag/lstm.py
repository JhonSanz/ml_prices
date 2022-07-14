import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from prepare_data import get_data


def get_trends(df):
    trends = df.copy()
    trends["trend"] = df["trend"].diff()
    trends = trends[trends["trend"] != 0]
    trends.reset_index(inplace=True)
    trends = trends[["Date"]]
    return trends


def get_dataset(train_data, trends, scaled_data):
    X = []
    Y = []
    trends_ = list(trends["Date"])
    print("trends: ", len(trends_))
    latest_position = 0
    i = 0
    for start, end in list(zip(trends_, trends_[1:])):
        zigzag_trend = train_data[
            (train_data["Date"] >= start) &
            (train_data["Date"] < end)
        ]
        values = zigzag_trend.values
        X.append(
           scaled_data[latest_position:latest_position + len(values), :-1].tolist()
        )
        Y.append(scaled_data[latest_position, -1])
        latest_position += len(values)
        i += 1
        if i == 2:
            break
    print("slices: ", len(X))
    return X, Y


def create_model():
    model = Sequential()
    model.add(LSTM(
        50, return_sequences=True, 
        input_shape=(None, 4)
    ))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    return model


# try:
#     df = pd.read_csv("test.csv")
#     df = df.iloc[:, 1:]
# except FileNotFoundError:
#     print("Setting up data")
#     get_data()
#     df = pd.read_csv("test.csv")

# trends = get_trends(df)
# scaler = MinMaxScaler(feature_range=(0, 1))
# training_data_len = math.ceil(len(df.values) * .8)

# scaled_data = scaler.fit_transform(
#     df.iloc[:, 1:].values
# )  # shape (70718, 5)

# X, Y = get_dataset(df, trends, scaled_data)

# x_train = X[:training_data_len]
# y_train = Y[:training_data_len]
# x_train = np.array(x_train, dtype=object)
# # y_train = np.array(y_train, dtype=object)

# print(x_train)
# print(y_train)

x_train = [
    [ np.array([1,2,3]), np.array([4,5,6]) ],
    [ np.array([7,8,9]) ],
]
x_train = np.array(x_train, dtype=object)
y_train = np.array([1,0])
print(x_train)
print(y_train)

model = create_model()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing data set
# test_data = scaled_data[training_data_len-60:, :]
# x_test = get_dataset(test_data)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# y_test = dataset[training_data_len:, :]


# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)
# rmse = np.sqrt(np.mean((predictions - y_test)**2))


# train = data[:training_data_len]
# valid = data[training_data_len:]
# valid['Predictions'] = predictions

# plt.figure(figsize=(16, 8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

# df = web.DataReader('^GSPC', data_source='yahoo',
#                     start='2022-04-01', end='2022-07-11')
# data = df[["Close"]]
# last_60_days = data[-60:].values
# last_60_days_scaled = scaler.transform(last_60_days)

# x_test_new = []
# x_test_new.append(last_60_days_scaled)
# x_test_new = np.array(x_test_new)
# x_test_new = np.reshape(
#     x_test_new, (x_test_new.shape[0], x_test_new.shape[1], 1))
# prediction_new = model.predict(x_test_new)
# prediction_new = scaler.inverse_transform(prediction_new)
# print(prediction_new)

# df = web.DataReader('^GSPC', data_source='yahoo',
#                     start='2022-07-11', end='2022-07-11')
# print(df["Close"])
