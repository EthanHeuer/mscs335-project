import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class functionData(Dataset):
    def __init__(self, f, a, b, num=1000, normalize=False):
        self.X = np.linspace(a, b, num=num).astype(np.float32)
        self.X.shape = (-1, 1)
        if normalize:
            self.y = f(self.X.astype(np.float32))
            self.X -= np.average(self.X)
            self.X /= np.std(self.X)
            self.X = torch.from_numpy(self.X)
            self.y -= np.average(self.y)
            self.y /= np.std(self.y)
            self.y = torch.from_numpy(self.y)
        else:
            self.y = torch.from_numpy(f(self.X).astype(np.float32))
            self.X = torch.from_numpy(self.X)
        self.len = num

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


model = nn.Sequential(
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.Sigmoid(),
    nn.Linear(100, 100),
    nn.Sigmoid(),
    nn.Linear(100, 100),
    nn.Sigmoid(),
    nn.Linear(100, 1)
)

data_train = functionData(lambda x: np.sin(3*x), -np.pi*2, np.pi*2, 16000, normalize=False)

trainLoader = DataLoader(data_train, batch_size=32, shuffle=True)

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Set up the plot
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(111)


def ANN_animate(i):
    ax.clear()
    for j, data in enumerate(trainLoader, 0):
        optimizer.zero_grad()
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y = model(data_train.X)
    print(f"Epoch: {i}")
    y = y.numpy()
    y.shape = (-1,)
    x = data_train.X.numpy()
    x.shape = (-1,)
    ax.plot(x, y, c='r')


ani = FuncAnimation(fig, ANN_animate, frames=100, interval=0, repeat=False)
plt.show()
