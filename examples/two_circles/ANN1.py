import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


# Dataset class for use in the DataLoader
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


# Fully connected ANN with 2 hidden layers
model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )


# Read in data for the ANN
df = pd.read_csv("two_circles.csv")
X_numpy = df.loc[:, ["x_0", "x_1"]].to_numpy()
y_numpy = df.loc[:, ["label"]].to_numpy()  # label has to be a column vector

# Create a train/test split for cross-validation
X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.20)

# Create Dataset objects
data_train = Data(X_train, y_train)
data_test = Data(X_test, y_test)

# Create the DataLoader with randomization for each epoch
trainLoader = DataLoader(data_train, batch_size=32, shuffle=True)

# Use Binary Cross-Entropy Loss
loss_fn = nn.BCELoss()

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# lr = learning rate, the higher, the more persuasive the model is to change

# Loop of the data a set number of times
for epoch in range(5):
    for i, data in enumerate(trainLoader, 0):
        # Zero out the stored gradient values
        optimizer.zero_grad()
        # Get the input and labels
        inputs, labels = data
        # Run the model on the input
        outputs = model(inputs)
        # Compute the loss function
        loss = loss_fn(outputs, labels)
        # Perform back propagation to calculate the gradient
        loss.backward()
        # Have the optimizer update the neural network based on the gradients
        optimizer.step()
    print(f"Epoch: {epoch}")
    with torch.no_grad():
        y_pred = model(data_test.X)
    print(f"Test Accuracy: {(y_pred.round() == data_test.y).float().mean().item()}")
    print("Confusion Matrix:")
    print(confusion_matrix(data_test.y, y_pred.round()))


# Plot the data
with torch.no_grad():
    y_pred = model(data_test.X).round().numpy()
y_pred.shape = (-1,)
y_test.shape = (-1,)
plt.style.use('dark_background')
plt.scatter(X_test[(y_pred == 0) * (y_test == 0), 0], X_test[(y_pred == 0) * (y_test == 0), 1], c='r', s=1)
plt.scatter(X_test[(y_pred == 1) * (y_test == 1), 0], X_test[(y_pred == 1) * (y_test == 1), 1], c='w', s=1)
plt.scatter(X_test[y_pred != y_test, 0], X_test[y_pred != y_test, 1], c='b', s=5)
plt.gca().set_aspect('equal')
plt.show()
