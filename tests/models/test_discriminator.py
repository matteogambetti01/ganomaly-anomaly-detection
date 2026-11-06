import torch
from src.models.discriminator import Discriminator


def test_discriminator_output_shape():
    """ Test the output shape of the Discriminator model. """
    discriminator = Discriminator()

    x = torch.randn(2, 3, 128, 128)

    validity = discriminator(x)

    assert validity.shape == (2, 1)

    assert validity.min() >= 0.0
    assert validity.max() <= 1.0
