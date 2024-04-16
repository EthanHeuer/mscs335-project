import _pickle as cPickle
import gzip
import torch
import torch.nn as nn
import numpy as np
import cv2

SET_SIZE = 50000
BATCH_SIZE = 10
IMG_SIZE = 28
NUM_BATCHES = SET_SIZE // BATCH_SIZE

# taken from http://www.deeplearning.net/tutorial/gettingstarted.html
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

images = torch.Tensor(train_set[0] * 2.0 - 1.0).view(50000, 1, 28, 28)
ONES = torch.ones([BATCH_SIZE, 1]).cuda()
ZEROS = torch.zeros([BATCH_SIZE, 1]).cuda()


class NumberDiscriminator(nn.Module):

    def __init__(self):
        super(NumberDiscriminator, self).__init__()
        self.i_to_h1 = nn.Sequential(
            nn.Conv2d(1, 32, 4),  # 25
            nn.LeakyReLU(0.2)
        )
        self.h1_to_h2 = nn.Sequential(
            nn.Conv2d(32, 32, 4),  # 22
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2)  # 11
        )
        self.h2_to_h3 = nn.Sequential(
            nn.Conv2d(32, 32, 5),  # 7
            nn.LeakyReLU(0.2)
        )
        self.h3_to_h4 = nn.Sequential(
            nn.Conv2d(32, 32, 4),  # 4
            nn.LeakyReLU(0.2)
        )
        self.h4_to_o = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1)
        )

    def forward(self, x):
        x = self.i_to_h1(x)
        x = self.h1_to_h2(x)
        x = self.h2_to_h3(x)
        x = self.h3_to_h4(x)
        x = self.h4_to_o(x.view(-1, 32 * 4 * 4))
        return x


def train_disc(disc, optimizer, real, fake, cost):
    # Clear the gradient in case forward propagation was used while training the generator
    optimizer.zero_grad()
    # Cross entropy loss averaged over the training batch
    loss = cost(disc(real), ONES) + cost(disc(fake), ZEROS)
    # Back propagation
    loss.backward()
    # Update weights
    optimizer.step()
    return loss


class NumberGenerator(nn.Module):

    def __init__(self):
        super(NumberGenerator, self).__init__()
        self.i_to_h1 = nn.Sequential(
            nn.Linear(20, 64 * 5 * 5),  # Create 64 5x5 features
            nn.LeakyReLU(0.2),
        )
        self.h1_to_h2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2),  # 64 11x11 features
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(64)
        )
        self.h2_to_h3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2),  # 64 23x23 features
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(64)
        )
        self.h3_to_h4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4),  # 64 26x26 features
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(64)
        )
        self.h4_to_o = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3),  # 64 28x28 features
            nn.Tanh()
        )

    def forward(self, x):
        x = self.i_to_h1(x)
        x = self.h1_to_h2(x.view(-1, 64, 5, 5))
        x = self.h2_to_h3(x)
        x = self.h3_to_h4(x)
        x = self.h4_to_o(x)
        return x


def train_gen(gen, optimizer, disc, seed, cost):
    # Clear the gradient in case forward propagation was used while training the discriminator
    optimizer.zero_grad()
    # Cross entropy loss averaged over the training batch
    loss = cost(disc(gen(seed)), ONES)
    # Back propagation
    loss.backward()
    # Update weights
    optimizer.step()
    # Return loss
    return loss


nd = NumberDiscriminator()
ng = NumberGenerator()
nd.cuda()
ng.cuda()

nd_cost = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
ng_cost = nn.BCEWithLogitsLoss(reduction='elementwise_mean')

EPOCHS = 1000

# Optimizer for discriminator
d_optimizer = torch.optim.Adam(nd.parameters(), lr=0.0002)
# Optimizer for generator
g_optimizer = torch.optim.Adam(ng.parameters(), lr=0.0002)

canvas = np.zeros((28 * 10, 28 * 10), dtype=np.float64)

for epoch in range(EPOCHS):
    print("---- Epoch: ", epoch)
    indices = np.arange(SET_SIZE)
    np.random.shuffle(indices)
    for batch in range(NUM_BATCHES):
        batch_seeds = torch.randn(BATCH_SIZE, 20).cuda()
        disc_error = train_disc(nd, d_optimizer, images[indices[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]].cuda(),
                                ng(batch_seeds), nd_cost)
        gen_error = train_gen(ng, g_optimizer, nd, batch_seeds, ng_cost)
        if batch % 100 == 0:
            with torch.no_grad():
                output = ng(torch.randn(100, 20).cuda()).cpu()
                img = (output.detach().numpy()[:, 0] + 1.0) / 2.0
                for row in range(10):
                    for col in range(10):
                        canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = img[row * 10 + col]
                cv2.imshow("Generated Number", canvas)
                cv2.waitKey(1)
            print("Epoch : ", epoch + batch / NUM_BATCHES, ", D_error: ", disc_error.item(), ", G_error: ",
                  gen_error.item())
cv2.destroyAllWindows()
