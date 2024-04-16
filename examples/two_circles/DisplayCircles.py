import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("two_circles.csv")
X = df.loc[:, ["x_0", "x_1"]].to_numpy()
y = df.loc[:, "label"].to_numpy()

plt.style.use('dark_background')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', s=1)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='w', s=1)
plt.gca().set_aspect('equal')
plt.show()
