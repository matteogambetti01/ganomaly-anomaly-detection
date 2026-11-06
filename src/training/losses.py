import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    """
    Computes the GANomaly generator loss.

    The total loss is composed of:
        - Reconstruction loss (L1)
        - Latent loss (MSE)
        - Adversarial loss (BCE)
    """

    def __init__(
        self,
        lambda_rec: float,
        lambda_lat: float,
        lambda_adv: float,
    ):
        super().__init__()

        self.lambda_rec = lambda_rec
        self.lambda_lat = lambda_lat
        self.lambda_adv = lambda_adv

        self.reconstruction_criterion = nn.L1Loss()
        self.latent_criterion = nn.MSELoss()
        self.adversarial_criterion = nn.BCELoss()

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        z_hat: torch.Tensor,
        disc_fake: torch.Tensor,
    ) -> dict:
        # Reconstruction loss
        loss_rec = self.reconstruction_criterion(x_hat, x)

        # Latent loss
        loss_lat = self.latent_criterion(z_hat, z)

        # Generator tries to fool the discriminator:
        # fake images should be classified as real.
        real_labels = torch.ones_like(disc_fake)

        loss_adv = self.adversarial_criterion(
            disc_fake,
            real_labels,
        )

        # Total generator loss
        loss_total = (
            self.lambda_rec * loss_rec
            + self.lambda_lat * loss_lat
            + self.lambda_adv * loss_adv
        )

        return {
            "total": loss_total,
            "reconstruction": loss_rec,
            "latent": loss_lat,
            "adversarial": loss_adv,
        }


class DiscriminatorLoss(nn.Module):
    """
    Computes the GANomaly discriminator loss.

    Real samples are assigned a smoothed label,
    while fake samples are assigned label 0.
    """

    def __init__(self, real_label_smoothing: float):
        super().__init__()

        if not 0.0 < real_label_smoothing <= 1.0:
            raise ValueError(
                "real_label_smoothing must be in the interval (0, 1]."
            )

        self.real_label_smoothing = real_label_smoothing
        self.criterion = nn.BCELoss()

    def forward(
        self,
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor,
    ) -> dict:

        # Smoothed labels for real samples
        real_labels = torch.full_like(
            disc_real,
            self.real_label_smoothing,
        )

        # Labels for fake samples
        fake_labels = torch.zeros_like(disc_fake)

        loss_real = self.criterion(disc_real, real_labels)
        loss_fake = self.criterion(disc_fake, fake_labels)

        # Preserve the original formulation:
        # discriminator loss = real loss + fake loss
        loss_total = loss_real + loss_fake

        return {
            "total": loss_total,
            "real": loss_real,
            "fake": loss_fake,
        }


def build_losses(config: dict):
    """
    Builds the generator and discriminator losses
    using the YAML configuration.
    """

    loss_config = config["training"]["loss"]

    generator_loss = GeneratorLoss(
        lambda_rec=loss_config["lambda_rec"],
        lambda_lat=loss_config["lambda_lat"],
        lambda_adv=loss_config["lambda_adv"],
    )

    discriminator_loss = DiscriminatorLoss(
        real_label_smoothing=loss_config["real_label_smoothing"],
    )

    return generator_loss, discriminator_loss
