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
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error


def get_dataset(train_data):
    """ Fills len(train_data) long arrays with 60 records 
        in each position
        example = [[1...60], [1...60], ...]
    """
    x_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, :-1])
    x_train = np.array(x_train)
    return np.array(x_train)


def create_model(shape_examples, shape_features):
    print(colored('Generating LSTM model.', 'yellow'))
    print(colored(f'Shape {shape_examples}x{shape_features}', 'cyan'))

    model = Sequential()
    model.add(LSTM(
        units=50,
        return_sequences=True,
        input_shape=(shape_examples, shape_features)
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
    X = np.load('resources/X_2.npy')
    Y = np.load('resources/Y_2.npy')
    scaler = joblib.load("resources/scaler_2.save")
    print(colored('Loaded storaged data.', 'green'))
except FileNotFoundError:
    try:
        df = pd.read_csv("resources/test.csv")
        df = df.iloc[:, 1:]
    except FileNotFoundError:
        print(colored('Creating missing file.', 'red'))
        get_data()
        df = pd.read_csv("resources/test.csv")

    print(colored('Setting up parameters.', 'yellow'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(
        df.iloc[:, 1:].values
    )  # shape (70718, 8)
    joblib.dump(scaler, "resources/scaler_2.save")
    X = get_dataset(scaled_data)
    Y = np.array(scaled_data[60:len(scaled_data), -1])
    np.save("resources/X_2.npy", X)
    np.save("resources/Y_2.npy", Y)

print(colored(f'X shape: {X.shape}', 'cyan'))
print(colored(f'Y shape: {Y.shape}', 'cyan'))

training_data_len = math.ceil(len(Y) * .8)
x_train = X[:training_data_len]
y_train = Y[:training_data_len]
x_train = np.array(x_train)
y_train = np.array(y_train)

try:
    model = load_model('resources/my_model_2.h5')
    print(colored('Model loaded successfully', 'green'))
except IOError:
    print(colored('Training...', 'yellow'))
    model = create_model(X.shape[1], X.shape[-1])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=64, epochs=5)
    model.save('resources/my_model_2.h5')

# sample = X[26584:26584+60][0][0]
# sample = [0.30617711, 0.3072357, 0.30659798, 0.30818803, 0.58209172, 0.3317571, 0.53299394, 0]
# sample = np.array(sample).reshape(1, -1)
# print(sample)
# data = scaler.inverse_transform(sample)
# print(data)

# sample = []
# for item in X[0]:
#     aux = list(item)
#     aux.append(0)
#     sample.append(aux)
# data = scaler.inverse_transform(sample)
# print(data)

sample = X[28]
sample = [sample]
sample = np.array(sample)
print(sample.shape)
prediction = model.predict(sample)
print(prediction)
# print("train: ", mean_squared_error(y_train, prediction))
