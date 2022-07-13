import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

X_train = [x for x in range(0, 30)]
Y_train = [
    3, 4, 5, 7, 10, 8, 9, 10, 10, 23, 27, 44, 50,
    63, 67, 60, 62, 70, 75, 88, 81, 87,
    95, 100, 108, 135, 151, 160, 169, 179
]

features = PolynomialFeatures(degree=10, include_bias=False)
poly_features = features.fit_transform(np.array(X_train).reshape(-1, 1))
model = LinearRegression()
model.fit(poly_features, Y_train)

pred_train = model.predict(np.array(poly_features))
print("mse_train: ", mean_squared_error(Y_train, pred_train))

X_test = [7, 8, 15, 20]
Y_test = [25, 27, 48, 72]
poly_features = features.fit_transform(np.array(X_test).reshape(-1, 1))
pred_test = model.predict(np.array(poly_features))
print("mse_test: ", mean_squared_error(Y_test, pred_test))

plt.figure()
plt.title("test")
plt.scatter(X_train, Y_train)
plt.scatter(X_test, Y_test)
plt.plot(X_train, pred_train, c="red")
plt.show()

