import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

print("X", X)
print("-"*50)
print("y", y)
print("-"*50)

reg = LinearRegression().fit(X, y)
result = reg.predict(np.array([[1, 4]])) # (1 * 1) + (2 * 4) + 3 = 12
print(np.array([[1, 4]]))
print("result", result)
