from torchvision import transforms


def get_transforms(image_size=128):
    """ Get the transformations for the dataset images. """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return transform
