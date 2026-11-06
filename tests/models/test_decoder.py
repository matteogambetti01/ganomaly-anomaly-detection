import torch
from src.models.decoder import Decoder


def test_decoder_output_shape():
    """ Test the output shape of the Decoder model. """
    decoder = Decoder()

    z = torch.randn(2, 512, 4, 4)

    x_hat = decoder(z)

    assert x_hat.shape == (2, 3, 128, 128)

    assert x_hat.min() >= -1.0
    assert x_hat.max() <= 1.0