import torch
from src.models.ganomaly import GANomaly


def test_ganomaly_forward():
    """ Test the forward pass of the GANomaly model. """
    model = GANomaly()

    x = torch.randn(2, 3, 128, 128)

    z, x_hat, z_hat, disc_real, disc_fake = model(x)

    assert z.shape == (2, 512, 4, 4)

    assert x_hat.shape == (2, 3, 128, 128)

    assert z_hat.shape == (2, 512, 4, 4)

    assert disc_real.shape == (2, 1)

    assert disc_fake.shape == (2, 1)


def test_encoders_do_not_share_parameters():
    """ Test that the encoder and encoder_hat don't share parameters. """
    model = GANomaly()

    encoder_params = list(model.encoder.parameters())

    encoder_hat_params = list(model.encoder_hat.parameters())

    for p1, p2 in zip(encoder_params, encoder_hat_params):

        assert p1 is not p2
