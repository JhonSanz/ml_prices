from termcolor import colored
import pandas as pd
import pandas_ta as ta


CANDLES_PREDICTION = 1

def get_x_train(train_data, trends):
    print(colored('Generating trends', 'yellow'))
    aux = train_data.copy()
    aux["trend"] = None
    trends_ = list(trends["Date"])
    for start, end in list(zip(trends_, trends_[1:])):
        mask = (aux["Date"] >= start) & (aux["Date"] <= end)
        aux.loc[mask, "trend"] = 1 if (
            list(trends[trends["Date"] == start]["Zigzag"])[0] <
            list(trends[trends["Date"] == end]["Zigzag"])[0]
        ) else -1
    aux = aux[aux["Date"] > trends_[0]]
    aux = aux[aux["Date"] < trends_[-1]]
    return aux


def get_data():
    df = pd.read_csv(
        'data.csv',
        names=[
            "Date", "Time", "High", "Open", "Close",
            "Low", "ZigzagMax", "ZigzagMin"
        ]
    )
    print(colored('Adding indicators', 'yellow'))
    df.ta.ao(
        high="high", low="low", slow=500,
        fast=1, append=True
    )
    df.ta.sma(length=500, append=True)
    df.ta.rsi(close="close", append=True)
    df["Date"] = df["Date"] + " " + df["Time"]
    df["Zigzag"] = df["ZigzagMax"] + df["ZigzagMin"]
    df = df.iloc[500:, :]
    trends = df[df["Zigzag"] > 0]
    df = df[[
        "Date", "High", "Open", "Close", "Low",
        "AO_1_500", "SMA_500", "RSI_14"
    ]]
    trends.reset_index(inplace=True)
    print(colored(f'Predict {CANDLES_PREDICTION} candles in the future', 'yellow'))
    df = get_x_train(df, trends)
    df["trend"] = df["trend"].shift(-CANDLES_PREDICTION)
    df = df.iloc[:-1, :]
    df.reset_index(inplace=True, drop=True)
    print(df.tail())
    trends = trends[["Date"]]
    df.to_csv("resources/test.csv")
    print(colored('Data created successfully', 'green'))