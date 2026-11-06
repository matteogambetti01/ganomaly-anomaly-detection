import torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.discriminator import Discriminator


class GANomaly(nn.Module):
    """ GANomaly model for anomaly detection using an encoder-decoder architecture with a discriminator. """
    def __init__(self, latent_dim=512):

        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim)

        self.encoder_hat = Encoder(latent_dim=latent_dim)

        self.decoder = Decoder(latent_dim=latent_dim)

        self.discriminator = Discriminator()

    def forward(self, x):
        """ Forward pass through the GANomaly model. """
        # First latent space
        z = self.encoder(x)

        # Reconstruction
        x_hat = self.decoder(z)

        # Second latent space
        z_hat = self.encoder_hat(x_hat)

        # Discriminator
        disc_real = self.discriminator(x)

        disc_fake = self.discriminator(x_hat)

        return z, x_hat, z_hat, disc_real, disc_fake
