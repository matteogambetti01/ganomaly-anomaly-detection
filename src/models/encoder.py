import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Encoder(nn.Module):
    """ Encoder model for extracting features from images using a pretrained ResNet18 model."""
    def __init__(self, latent_dim=512):

        super().__init__()

        # ResNet18 pretrained on ImageNet
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove avgpool and fully connected layer
        self.features = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # Freeze all layers
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze the last three layers of the ResNet18 model
        for child in list(self.features.children())[-3:]:

            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x):

        z = self.features(x)

        return z
