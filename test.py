import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# X_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# # y = 1 * x_0 + 2 * x_1 + 3
# Y_train = np.dot(X_train, np.array([1, 2])) + 3


X_train = np.arange(0, 30)
Y_train = [
    3, 4, 5, 7, 10, 8, 9, 10, 10, 23, 27, 44, 50,
    63, 67, 60, 62, 70, 75, 88, 81, 87,
    95, 100, 108, 135, 151, 160, 169, 179
]
features = PolynomialFeatures(degree=2, include_bias=False)
poly_features = features.fit_transform(X_train.reshape(-1, 1))
model = LinearRegression()
model.fit(poly_features, Y_train)


pred_train = model.predict(np.array(poly_features))  # (1 * 1) + (2 * 4) + 3 = 12
print("Y_train", Y_train)
print("prediction", pred_train)
print(mean_squared_error(Y_train, pred_train))


plt.figure()
plt.title("test")
plt.scatter(X_train, pred_train)
plt.plot(X_train, pred_train, c="red")
plt.show()

# print("-" * 50)

# x_0 = -6400000.5415
# x_1 = 1.651651
# test = [x_0, x_1]
# x_test = np.array([test])
# y_test = np.array([
#     (1 * x_0 + 2 * x_1 + 3)
# ])
# pred_test = model.predict(np.array([test]))
# print("y_test", y_test)
# print("prediction", pred_test)
# print(mean_squared_error(y_test, pred_test))
