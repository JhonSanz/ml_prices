import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
        df = df.iloc[self.SLOW:, :]
        return df

    def get_model(self):
        try:
            model = joblib.load("model.pkl")
        except FileNotFoundError:
            df = self.get_data("us100_labeled.csv", True)
            self.scaler = StandardScaler()
            X_train, X_test, y_train, y_test = train_test_split(
                (df[[self.AO, "close"]]), df[["type"]], test_size=0.2
            )
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)

            model = MLPClassifier(
                hidden_layer_sizes=(10, 8, 10),
                activation='relu',
                solver='adam',
                max_iter=1000
            )
            model.fit(X_train, y_train.values.ravel())
            predictions = model.predict(X_test)

            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))
            # joblib.dump(model, "model.pkl") 
        return model

    def calculate(self):
        model = self.get_model()
        df = self.get_data("us100_30M.csv", False)
        df = df[[self.AO, "close"]]
        train_data = self.scaler.transform(df)
        prediction = model.predict(train_data)
        print(prediction)
        df["result"] = prediction
        df.to_csv("results.csv")

Solver().calculate()