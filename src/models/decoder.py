import torch.nn as nn


class Decoder(nn.Module):
    """ Decoder model for reconstructing images from latent representations. """
    def __init__(self, latent_dim=512):

        super().__init__()

        self.decoder = nn.Sequential(

            # (512, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(
                latent_dim, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # (128, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # (64, 32, 32) -> (32, 64, 64)
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # (32, 64, 64) -> (3, 128, 128)
            nn.ConvTranspose2d(
                32, 3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, z):
        """ Reconstructs images from latent representations. """
        x_hat = self.decoder(z)

        return x_hat
