import torch
from PIL import Image
from src.data.transforms import get_transforms


def test_image_transforms():
    """ Test the image transformations applied to the images."""
    transform = get_transforms(image_size=128)

    image = Image.new("RGB", (256, 256), color="white")

    transformed_image = transform(image)

    assert isinstance(transformed_image, torch.Tensor)

    assert transformed_image.shape == (3, 128, 128)

    assert transformed_image.min() >= -1.0

    assert transformed_image.max() <= 1.0
