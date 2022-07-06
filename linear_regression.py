import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.neural_network import MLPClassifier
import joblib


class Solver:
    SLOW = 500
    FAST = 1
    AO = f'AO_{FAST}_{SLOW}'

    def __init__(self):
        self.scaler = None

    def get_data(self, file_name, train_mode):
        if train_mode:
            df = pd.read_csv(file_name)
            df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)
            df = df[["close", "high", "low", "type"]]
        else:
            df = pd.read_csv(file_name, sep="\t")
            df["<DATE>"] = df["<DATE>"] + " " + df["<TIME>"]
            df.rename(columns={
                "<DATE>": "date",
                "<OPEN>": "open",
                "<HIGH>": "high",
                "<LOW>": "low",
                "<CLOSE>": "close",
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d %H:%M")
            df.set_index(pd.DatetimeIndex(df["date"]), inplace=True)
            df = df[["high", "low", "close"]]
        df.ta.ao(
            high="high", low="low", slow=self.SLOW,
            fast=1, append=True
        )
        df.ta.sma(length=500, append=True)
        df.ta.rsi(close="close", append=True)
        df = df.iloc[self.SLOW:, :]
        return df

    def define_sets(self):
        df = self.get_data("us100_labeled.csv", True)
        df["bad_col"] = -100
        X_train, X_test, y_train, y_test = train_test_split(
            # df[[self.AO, "close", "SMA_500", "RSI_14"]],
            df[["bad_col"]],
            df[["type"]], test_size=0.2
        )
        X_train.to_csv("train_data/X_train_bad.csv")
        X_test.to_csv("train_data/X_test_bad.csv")
        y_train.to_csv("train_data/y_train_bad.csv")
        y_test.to_csv("train_data/y_test_bad.csv")

    def get_train_data(self):
        X_train = pd.read_csv("train_data/X_train_bad.csv")
        X_test = pd.read_csv("train_data/X_test_bad.csv")
        y_train = pd.read_csv("train_data/y_train_bad.csv")
        y_test = pd.read_csv("train_data/y_test_bad.csv")
        X_train.set_index(pd.DatetimeIndex(X_train["date"]), inplace=True)
        X_test.set_index(pd.DatetimeIndex(X_test["date"]), inplace=True)
        y_train.set_index(pd.DatetimeIndex(y_train["date"]), inplace=True)
        y_test.set_index(pd.DatetimeIndex(y_test["date"]), inplace=True)
        X_train = X_train.iloc[:, 1:]
        X_test = X_test.iloc[:, 1:]
        y_train = y_train.iloc[:, 1:]
        y_test = y_test.iloc[:, 1:]
        return X_train, X_test, y_train, y_test

    def get_model(self):
        try:
            X_train, X_test, y_train, y_test = self.get_train_data()
        except:
            self.define_sets()
            X_train, X_test, y_train, y_test = self.get_train_data()
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        try:
            model = joblib.load("model.pkl")
        except FileNotFoundError:
            model = MLPClassifier(
                hidden_layer_sizes=(100, 100, 100),
                activation='relu',
                solver='adam',
                max_iter=10000
            )
            model.fit(X_train, y_train.values.ravel())
            # joblib.dump(model, "model.pkl")

        predictions_train = model.predict(X_train)
        print("model: 100, 100, 100")
        print("train: ", mean_squared_error(y_train[["type"]], predictions_train))
        print(classification_report(y_train, predictions_train))

        predictions_test = model.predict(X_test)
        print("test: ", mean_squared_error(y_test[["type"]], predictions_test))
        print(classification_report(y_test, predictions_test))
        return model

    def calculate(self):
        model = self.get_model()
        
        # df = self.get_data("us100_30M.csv", False)
        # # df = self.get_data("US100Cash_M30_2020_01_01_2020_12_31.csv", False)
        # df = df[[self.AO, "close", "SMA_500", "RSI_14"]]
        # train_data = self.scaler.transform(df)
        # prediction = model.predict(train_data)
        # df["result"] = prediction
        # df.to_csv("results.csv")


Solver().calculate()
