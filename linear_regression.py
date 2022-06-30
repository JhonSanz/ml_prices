import pandas as pd
import pandas_datareader as web
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas_ta as ta
import matplotlib.pyplot as plt


SHOW_PLOT = False
EMA_LENGTH = 500
EMA_IND = f'EMA_{EMA_LENGTH}'

df = web.DataReader(
    '^GSPC', data_source='yahoo', start='2012-01-01', end='2022-06-28'
)
df = df[["Close"]]
df.ta.ema(close="Close", length=EMA_LENGTH, append=True)
df = df.iloc[EMA_LENGTH:, :]

X_train, X_test, y_train, y_test = train_test_split(
    df[[EMA_IND]], df[["Close"]], test_size=.2
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# another test :D
df = web.DataReader(
    '^GSPC', data_source='yahoo', start='2012-01-01', end='2022-06-29'
)
df = df[["Close"]]
df.ta.ema(close="Close", length=EMA_LENGTH, append=True)
last_60_days = df[-500:]
print(last_60_days)
y_pred = model.predict([[4038.759228]])
print(y_pred)

if SHOW_PLOT:
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD($)',fontsize=18)
    plt.plot(df[EMA_IND])
    plt.plot(df['Close'])
    plt.legend(['Train','Val','Predictions'],loc='lower right')
    plt.show()