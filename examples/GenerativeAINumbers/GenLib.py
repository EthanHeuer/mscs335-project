import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


# Typed MNIST dataset
class TMNISTData(Dataset):
    def __init__(self):
        # Load the typed MNIST dataset
        df = pd.read_csv("TMNIST_Data.csv")

        # Extract the numbers and scale to values 0 to 255
        numbers = df.loc[:, "1":"784"].to_numpy() / 255.0

        # Create a one-hot encoding of the labels
        labels = df["labels"].to_numpy()
        labels_one_hot = np.zeros((numbers.shape[0], 10))
        labels_one_hot[np.arange(numbers.shape[0]), labels] = 1.0

        # Convert to torch tensor
        self.X = torch.tensor(numbers).view(-1, 1, 28, 28).to(torch.float32)  # single channel 28x28 images
        self.y = torch.tensor(labels_one_hot).to(torch.float32)

        # store the size of the dataset
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len


# Convolutional encoder which keeps track of KL loss
class Encoder(nn.Module):
    def __init__(self, latent_dims=16):
        super(Encoder, self).__init__()
        self.latent_dims = latent_dims

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),  # 32 26x26 images
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),  # 64 24x24 images
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 12x12 images
            nn.Conv2d(64, 128, (3, 3)),  # 128 10x10 images
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 5x5 images
            nn.Flatten()  # 3200 length Tensor
        )

        # mean and log variance
        self.fc_mean = nn.Linear(128 * 5 * 5, self.latent_dims)
        self.fc_log_var = nn.Linear(128 * 5 * 5, self.latent_dims)

        # KL Loss
        self.kl = 0.0

    def forward(self, x):
        x = self.conv_layers(x)  # Apply CNN part of the network

        z_mean = self.fc_mean(x)  # mean
        z_log_var = self.fc_log_var(x)  # log of the variance
        z_sigma = torch.exp(0.5 * self.fc_log_var(x))  # Standard deviation

        self.kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_sigma.pow(2)) / x.shape[0]  # sum KL over all dimensions and average the batch

        return z_mean + z_sigma * torch.randn_like(z_mean)  # return vector from a probability distribution


# Transpose convolutional decoder
class Decoder(nn.Module):
    def __init__(self, latent_dims=16):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.transpose_conv = nn.Sequential(
            nn.Linear(latent_dims, 128 * 5 * 5),  # Upscale back to 128 5x5 images and reshape
            nn.Unflatten(1, (128, 5, 5)),
            nn.ConvTranspose2d(128, 64, (3, 3), stride=(2, 2), output_padding=(1, 1)),  # Transpose Convolution to 64 12x12 images
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), output_padding=(1, 1)),  # Transpose Convolution to 32 26x26 images
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (3, 3)),  # Transpose Convolution to 1 28x28 image
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.transpose_conv(x)


# Variational Autoencoder
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims=16):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Network to remove noise from latents
class LatentDenoise(nn.Module):
    def __init__(self, latent_dims=16):
        super(LatentDenoise, self).__init__()
        self.latent_lin1 = nn.Linear(latent_dims, 256)
        self.latent_lin2 = nn.Linear(256, 128)
        self.latent_lin3 = nn.Linear(128, 16)
        self.category_lin = nn.Linear(10, 256)

    def forward(self, x, y):
        x = self.latent_lin1(x)
        y = self.category_lin(y)
        x = F.relu(x + y)
        x = F.relu(self.latent_lin2(x))
        x = self.latent_lin3(x)
        return x


# Creates images of specified numbers
class CreateNumbers:
    def __init__(self, decoder_file="decoder.pt", denoise_file="denoise.pt", statistics_file="statistics.pt", device="cuda"):
        # Select cpu if requested or if CUDA is unavailable
        self.device = device
        if not ((self.device == 'cuda') and torch.cuda.is_available()):
            self.device = 'cpu'

        # Load the decoder
        self.decoder = Decoder()
        self.decoder.load_state_dict(torch.load(decoder_file, map_location=torch.device(self.device)))
        self.decoder.eval()
        self.decoder.to(self.device)

        # Load the denoise network
        self.ld = LatentDenoise()
        self.ld.load_state_dict(torch.load(denoise_file, map_location=torch.device(self.device)))
        self.ld.eval()
        self.ld.to(self.device)

        # Load the mean and standard deviation
        self.statistics = torch.load(statistics_file, map_location=torch.device(self.device)).to(self.device)

    def getNumbers(self, number=0, count=100, std_scale=1.0):
        # Make one-hot array of specified size
        number_one_hot = torch.zeros(count, 10)
        number_one_hot[:, number] = 1.0
        number_one_hot = number_one_hot.to(self.device)  # put on GPU if possible

        # Sample the latent space with a noraml distribution
        random_latents = torch.normal(0, std_scale, size=(count, self.decoder.latent_dims)).to(self.device)
        # Transform to the actual computed latent space since it might not be perfectly N(0,1) in every variable
        random_latents *= self.statistics[1, :]
        random_latents += self.statistics[0, :]

        with torch.no_grad():
            # Run through denoise process
            noise = self.ld(random_latents, number_one_hot)
            # Decode into images after substracting noise
            decoded = self.decoder(random_latents-noise)

        # return as numpy array and eliminate channel dimension
        return decoded.view(-1, 28, 28).cpu().numpy()

