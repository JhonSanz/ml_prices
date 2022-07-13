import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1
)
clf = MLPClassifier(
    random_state=1,
    hidden_layer_sizes=(2),
    max_iter=1000000000000000
).fit(X_train, y_train)
pred_train = clf.predict(X_train)
print(clf.score(X_test, y_test))

df = pd.DataFrame(X, columns=["x", "y"])
df["label"] = y
groups = df.groupby('label')

result = pd.DataFrame(X_train, columns=["x", "y"])
result['label'] = pred_train
groups_pred = result.groupby('label')

fig, ax = plt.subplots(1, 2)
for name, group in groups:
    ax[0].plot(
        group.x, group.y, marker='o',
        color='blue' if name == 0 else "red",
        linestyle='', ms=10, label=name
    )
    ax[1].plot(
        group.x, group.y, marker='o',
        color='blue' if name == 0 else "red",
        linestyle='', ms=10, label=name
    )
for name, group in groups_pred:
    ax[0].plot(
        group.x, group.y, marker='^',
        color='black' if name == 0 else "cyan",
        linestyle='', ms=10, label=name
    )
# for name, group in groups_test:
#     ax[1].plot(
#         group.x, group.y, marker='^',
#         color='black' if name == 0 else "cyan",
#         linestyle='', ms=10, label=name
#     )
ax[0].legend()
ax[1].legend()
plt.show()

