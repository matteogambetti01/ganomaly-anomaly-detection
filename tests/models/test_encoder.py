import torch
from src.models.encoder import Encoder


def test_encoder_output_shape():
    """ Test the output shape of the Encoder model. """
    encoder = Encoder()

    x = torch.randn(2, 3, 128, 128)
    z = encoder(x)

    assert z.shape == (2, 512, 4, 4)


def test_encoder_trainable_parameters():
    """ Test that the last three layers of the Encoder model are trainable and the rest are frozen. """
    encoder = Encoder()

    children = list(encoder.features.children())

    frozen_layers = children[:-3]
    trainable_layers = children[-3:]
    # Check that the frozen layers have 'requires_grad' set to False
    assert all(
        not param.requires_grad
        for layer in frozen_layers
        for param in layer.parameters()
    )
    # Check that the trainable layers have 'requires_grad' set to True
    assert all(
        param.requires_grad
        for layer in trainable_layers
        for param in layer.parameters()
    )
