import _pickle as cPickle
import gzip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv

# taken from http://www.deeplearning.net/tutorial/gettingstarted.html
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()


class Data(Dataset):
    def __init__(self, X):
        self.X = torch.Tensor(X * 2.0 - 1.0).view(-1, 1, 28, 28)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.len


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 640),
            nn.ReLU(),
            nn.Linear(640, 16)
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 640),
            nn.ReLU(),
            nn.Linear(640, 784),
            nn.Tanh(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


ae = AutoEncoder()
ae.cuda()

image_data = Data(train_set[0])
data_loader = DataLoader(image_data, batch_size=10, shuffle=True)

ae_cost = nn.MSELoss()

optimizer = torch.optim.Adam(ae.parameters(), lr=0.0005)

EPOCHS = 100

running_loss = 0.0
canvas = np.zeros((28, 56))
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    for i, data in enumerate(data_loader, 0):
        data = data.cuda()
        optimizer.zero_grad()
        outputs = ae(data)
        loss = ae_cost(outputs, data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f"Loss: {running_loss / 1000.0}")
            running_loss = 0.0
            with torch.no_grad():
                d = image_data.X[np.random.randint(50000)].cuda()
                out = ae(d.view(1, 1, 28, 28)).cpu().numpy()
                out.shape = (28, 28)
                out = (out + 1) / 2
                d = d.cpu().numpy()
                d.shape = (28, 28)
                d = (d + 1) / 2
                canvas[:, :28] = d
                canvas[:, 28:] = out
                cv.imshow("Original / Output", np.repeat(np.repeat(canvas, 20, axis=0), 20, axis=1))
                cv.waitKey(1)

cv.destroyAllWindows()
