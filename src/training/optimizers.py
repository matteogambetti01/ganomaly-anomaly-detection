import torch


def build_optimizers(model, config: dict):
    """
    Builds the generator and discriminator optimizers
    using the YAML configuration.

    Generator optimizer:
        - Encoder
        - Encoder_hat
        - Decoder

    Discriminator optimizer:
        - Discriminator
    """

    optimizer_config = config["training"]["optimizer"]

    optimizer_name = optimizer_config["name"]

    if optimizer_name != "Adam":
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    betas = tuple(optimizer_config["betas"])

    # Collect only trainable parameters
    encoder_params = [
        p for p in model.encoder.parameters()
        if p.requires_grad
    ]

    encoder_hat_params = [
        p for p in model.encoder_hat.parameters()
        if p.requires_grad
    ]

    decoder_params = [
        p for p in model.decoder.parameters()
        if p.requires_grad
    ]

    discriminator_params = [
        p for p in model.discriminator.parameters()
        if p.requires_grad
    ]

    # Generator optimizer
    optimizer_G = torch.optim.Adam(
        [
            {
                "params": encoder_params,
                "lr": optimizer_config["encoder_lr"],
            },
            {
                "params": encoder_hat_params,
                "lr": optimizer_config["encoder_hat_lr"],
            },
            {
                "params": decoder_params,
                "lr": optimizer_config["decoder_lr"],
            },
        ],
        betas=betas,
    )

    # Discriminator optimizer
    optimizer_D = torch.optim.Adam(
        discriminator_params,
        lr=optimizer_config["discriminator_lr"],
        betas=betas,
    )

    return optimizer_G, optimizer_D