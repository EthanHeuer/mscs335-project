import torch
from torch.utils.data import DataLoader
from torch import nn
import GenLib
import time


def trainAE(epochs=50, batch_size=16, MSEScaling=1600.0, device='cuda', encoder_file="encoder.pt", decoder_file="decoder.pt", statistics_file="statistics.pt", latent_dims=16):
    # Select cpu if requested or if CUDA is unavailable
    if not ((device == 'cuda') and torch.cuda.is_available()):
        device = 'cpu'

    # Create the DataLoader
    TMNISTData = GenLib.TMNISTData()
    TMNIST_loader = DataLoader(TMNISTData, batch_size=batch_size, shuffle=True)

    # Create an instance of the VAE Autoencoder and load it on the specified device
    vae = GenLib.VariationalAutoencoder(latent_dims=latent_dims)
    vae.to(device)

    # Mean Square Error loss function for the output of the autoencoder
    vae_MSE = nn.MSELoss()

    # Use Adam optimizer with lr=0.001
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    running_loss = 0.0
    start = time.time()
    for epoch in range(epochs):
        for i, data in enumerate(TMNIST_loader, 0):
            image, _ = data  # get the image data but ignore the labels
            image = image.to(device)  # put the data on the gpu if available

            optimizer.zero_grad()  # 0 the gradients

            output = vae(image)  # Get the output of the network
            loss = vae.encoder.kl + vae_MSE(output, image) * MSEScaling  # Compute the weighted loss function

            loss.backward()  # Compute gradient

            optimizer.step()  # Update weights in the network

            running_loss += loss.item() * len(image)
        # Scaled loss so that it is roughly independent of batch size
        print(f"Loss for epoch {epoch + 1} of {epochs}: {running_loss / len(TMNISTData)}.")
        running_loss = 0.0
    print(f"Seconds to train VAE: {time.time() - start}.")

    latents = torch.zeros((len(TMNISTData) // batch_size) * batch_size, latent_dims).to(device)  # For simplicity/speed doing batch sizes of 100
    with torch.no_grad():
        for i, data in enumerate(TMNIST_loader, 0):
            data, _ = data  # get the image data but ignore the labels
            if len(data) == batch_size:  # Ignore remainder batch at the end
                data = data.to(device)  # put the data on the gpu if available
                output = vae.encoder(data)  # Convert to numpy array
                latents[i * batch_size:(i + 1) * batch_size] = output
    statistics = torch.zeros(2, latent_dims).to(device)
    statistics[0, :] = torch.mean(latents, dim=0)
    statistics[1, :] = torch.std(latents, dim=0)
    print(f"Tensor of means: {statistics[0].cpu()}")
    print(f"Tensor of standard deviations: {statistics[1].cpu()}")

    # Save encoder and decoder from
    torch.save(vae.encoder.state_dict(), encoder_file)
    torch.save(vae.decoder.state_dict(), decoder_file)
    torch.save(statistics, statistics_file)
    print("Encoder and decoder saved.")


def trainDenoise(epochs=10, batch_size=16, noise_std=1.0, encoder_file="encoder.pt", denoise_file="denoise.pt", device="cuda"):
    # Select cpu if requested or if CUDA is unavailable
    if not ((device == 'cuda') and torch.cuda.is_available()):
        device = 'cpu'

    # Create the DataLoader
    TMNISTData = GenLib.TMNISTData()
    TMNIST_loader = DataLoader(TMNISTData, batch_size=batch_size, shuffle=True)

    # Load the encoder portion of the VAE
    encoder = GenLib.Encoder()
    encoder.load_state_dict(torch.load(encoder_file, map_location=torch.device(device)))
    encoder.eval()
    encoder.to(device)

    # Create an instance of the LatentDenoise network
    ld = GenLib.LatentDenoise()
    ld.to(device)

    # Mean square error for loss function
    denoise_MSE = nn.MSELoss()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(ld.parameters(), lr=0.0001)

    # Train the network
    running_loss = 0.0
    start = time.time()
    for epoch in range(epochs):
        for i, data in enumerate(TMNIST_loader, 0):
            image, label = data
            image, label = image.to(device), label.to(device)

            latent = encoder(image)  # latent from the image

            noise = torch.randn_like(latent) * noise_std  # Noise to be added to the image
            latent += noise

            optimizer.zero_grad()
            outputs = ld(latent, label)  # try to identify the noise given the label

            loss = denoise_MSE(outputs, noise)  # MSE loss
            loss.backward()  # compute gradient
            optimizer.step()  # update weights based on gradient
            running_loss += loss.item() * len(image)
        # Scaled loss so that it is roughly independent of batch size
        print(f"Loss for epoch {epoch + 1} of {epochs}: {running_loss / len(TMNISTData)}.")
        running_loss = 0.0
    print(f"Seconds to train LatentDenoise: {time.time() - start}.")
    # Save encoder and decoder from
    torch.save(ld.state_dict(), denoise_file)
    print(f"LatentDenoise saved.")


def default_build_all(vae_epochs=50, denoise_epochs=10, latent_dims=16):
    print("========================================= Training VAE =========================================")
    trainAE(epochs=vae_epochs, latent_dims=latent_dims)
    print("========================================= Training Latent Denoiser =============================")
    trainDenoise(epochs=denoise_epochs)

