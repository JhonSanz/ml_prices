import math
from termcolor import colored
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, scale
import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking
from keras.layers import Dropout
from keras.models import load_model
import matplotlib.pyplot as plt
from prepare_data import get_data

MASK_VALUE = 666


def get_trends(df):
    trends = df.copy()
    trends["trend"] = df["trend"].diff()
    trends = trends[trends["trend"] != 0]
    trends.reset_index(inplace=True)
    trends = trends[["Date"]]
    return trends


def get_dataset(train_data, trends, scaled_data):
    print(colored('Generating data set.', 'yellow'))

    X = []
    Y = []
    highest_value = 0
    for val in train_data["trend"].unique().tolist():
        s = df['trend'].eq(val)
        r = (~s).cumsum()[s].value_counts().max()
        highest_value = r if r > highest_value else highest_value

    trends_ = list(trends["Date"])
    print("trends: ", len(trends_))
    latest_position = 0
    for start, end in list(zip(trends_, trends_[1:])):
        zigzag_trend = train_data[
            (train_data["Date"] >= start) &
            (train_data["Date"] < end)
        ]
        values = zigzag_trend.values
        batch = scaled_data[
            latest_position:latest_position +
            len(values), :-1
        ].tolist()
        batch.extend(
            [[MASK_VALUE] * len(batch[0])]
            * (highest_value - len(batch))
        )
        X.append(batch)
        Y.append(scaled_data[latest_position, -1])
        latest_position += len(values)
    print("slices: ", len(X))
    print(colored('Data set done.', 'green'))
    return X, Y


def create_model(shape_examples, shape_features):
    print(colored('Generating LSTM model.', 'yellow'))
    print(colored(f'Shape {shape_examples}x{shape_features}', 'cyan'))

    model = Sequential()
    model.add(Masking(
        mask_value=MASK_VALUE,
        input_shape=(shape_examples, shape_features)
    ))
    model.add(LSTM(
        units=50,
        return_sequences=True,
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(units=50))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


try:
    X = np.load('resources/X.npy')
    Y = np.load('resources/Y.npy')
    scaler = joblib.load("resources/scaler.save")
    print(colored('Loaded storaged data.', 'green'))
except FileNotFoundError:
    try:
        df = pd.read_csv("resources/test.csv")
        df = df.iloc[:, 1:]
    except FileNotFoundError:
        print(colored('Creating missing file.', 'red'))
        get_data()
        df = pd.read_csv("resources/test.csv")

    trends = get_trends(df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        df.iloc[:, 1:].values
    )  # shape (70718, 5)
    joblib.dump(scaler, "resources/scaler.save")
    X, Y = get_dataset(df, trends, scaled_data)
    np.save("resources/X.npy", X)
    np.save("resources/Y.npy", Y)

training_data_len = math.ceil(len(Y) * .8)
x_train = X[:training_data_len]
y_train = Y[:training_data_len]
x_train = np.array(x_train)
y_train = np.array(y_train)

try:
    model = load_model('resources/my_model.h5')
    print(colored('Model loaded successfully', 'green'))
except IOError:
    print(colored('Training...', 'yellow'))
    model = create_model(X.shape[1], X.shape[-1])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=100)
    model.save('resources/my_model.h5')


# predictions = model.predict(x_train)
# # predictions = scaler.inverse_transform(predictions)
# print("y_train: ", y_train, len(y_train))
# print("predictions: ", predictions, len(predictions))

data = [
    [12054.96,12020.91,12036.7,12014.96,248.9479399999982,11786.90934,50.513533759766055,-1],
    [12044.91,12036.08,12034.15,12026.22,249.09552000000076,11787.3453,49.70259454415664,-1],
    [12049.02,12034.02,12031.69,12028.75,251.9762099999989,11787.784479999998,48.887223039359164,-1],
    [12036.75,12031.32,12013.87,12007.42,234.76174999999967,11788.198699999999,43.34059984386315,-1],
    [12039.05,12014.49,12032.36,12008.79,236.16675000000032,11788.63108,49.715603419268476,-1],
    [12041.89,12031.87,12039.15,12013.96,239.71710999999777,11789.13384,51.857765137955425,-1],
    [12046.27,12038.91,12019.45,12017.83,243.35965000000033,11789.598659999998,45.76600060112602,-1],
]
some = scaler.transform(data)
some = list(map(lambda x: list(x[:-1]), some))
some.extend([[MASK_VALUE] * 7] * (135 - len(data)))
some = [some]

prediction_2 = model.predict(some)
print(prediction_2)


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
