from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("two_circles.csv")
X = df.loc[:, ["x_0", "x_1"]].to_numpy()
y = df.loc[:, "label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

clf = SVC(kernel="rbf", gamma=2.0)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))

plt.style.use('dark_background')
plt.scatter(X_test[(y_pred == 0) * (y_test == 0), 0], X_test[(y_pred == 0) * (y_test == 0), 1], c='r', s=1)
plt.scatter(X_test[(y_pred == 1) * (y_test == 1), 0], X_test[(y_pred == 1) * (y_test == 1), 1], c='w', s=1)
plt.scatter(X_test[y_pred != y_test, 0], X_test[y_pred != y_test, 1], c='b', s=5)
plt.gca().set_aspect('equal')
plt.show()
