import pandas as pd


def get_x_train(train_data, trends):
    aux = train_data.copy()
    aux["trend"] = None
    trends_ = list(trends["Date"])
    # aux.loc[
    #     ((aux["Date"] >= "2022.06.29 15:00") & (aux["Date"] <= "2022.07.13 22:30")),
    #     ["trend"]
    # ] = 70
    for start, end in list(zip(trends_, trends_[1:]))[::2]:
        # print("start  ", start, "  end: ", end)
        aux.loc[
            ((aux["Date"] >= start) & (aux["Date"] <= end)),
            ["trend"]
        ] = 1 if (
            list(trends[trends["Date"] == start]["Zigzag"])[0] <
            list(trends[trends["Date"] == end]["Zigzag"])[0]
        ) else -1
    print(aux)
    return aux


def get_data():
    df = pd.read_csv(
        'data.csv',
        names=[
            "Date", "Time", "High", "Open", "Close",
            "Low", "ZigzagMax", "ZigzagMin"
        ]
    )
    df["Date"] = df["Date"] + " " + df["Time"]
    # df.set_index(pd.DatetimeIndex(df["Date"]), inplace=True)
    df["Zigzag"] = df["ZigzagMax"] + df["ZigzagMin"]
    trends = df[df["Zigzag"] > 0]
    df = df[["Date", "High", "Open", "Close", "Low"]]
    trends.reset_index(inplace=True)
    df = get_x_train(df, trends)
    trends = trends[["Date"]]
    print(df)
    raise Exception
    return df, trends
