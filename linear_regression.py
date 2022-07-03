import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import joblib

SHOW_PLOT = False
SLOW = 500
FAST = 1
AO = f'AO_{FAST}_{SLOW}'

def get_data(file_name, train_mode):
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
        high="high", low="low", slow=SLOW,
        fast=1, append=True
    )
    df = df.iloc[SLOW:, :]
    return df

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    df = get_data("us100_labeled.csv", True)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        (df[[AO, "close"]]), df[["type"]], test_size=0.2
    )
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

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
    joblib.dump(model, "model.pkl") 

df = get_data("us100_30M_07.csv", False)
print(df[305:])
df = df[[AO, "close"]]
print(model.predict([[330.26310, 11920.67]]))
