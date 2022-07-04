import pandas as pd

df = pd.read_csv("labeled_data.csv")
data = pd.read_csv("us100_30M.csv", sep="\t")

data["<DATE>"] = data["<DATE>"] + " " + data["<TIME>"]
data.rename(columns={
    "<DATE>": "date",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
}, inplace=True)

data["date"] = pd.to_datetime(data["date"], format="%Y.%m.%d %H:%M")
data = data[["date", "open", "high", "low", "close"]]
data["type"] = None

df["inicio"] = pd.to_datetime(df["inicio"], format="%Y.%m.%d %H:%M")
df["fin"] = pd.to_datetime(df["fin"], format="%Y.%m.%d %H:%M")

for start, end, type in zip(df["inicio"], df["fin"], df["tipo"]):
    mask = (data["date"] >= start) & (data["date"] < end)
    data.loc[mask, ["type"]] = type

data.to_csv("us100_labeled.csv")