import pytest
from PIL import Image

@pytest.fixture
def fake_mvtec_dataset(tmp_path):
    """ Create a temporary directory structure that mimics the MVTec dataset for testing purposes."""
    root = tmp_path / "bottle"

    train_good = root / "train" / "good"
    test_good = root / "test" / "good"
    test_broken = root / "test" / "broken_small"

    train_good.mkdir(parents=True)
    test_good.mkdir(parents=True)
    test_broken.mkdir(parents=True)

    # Creating 20 good training images, 4 good test images and 4 broken test images
    for i in range(20):

        image = Image.new("RGB", (128, 128), color="white")

        image.save(train_good / f"train_{i}.png")

    for i in range(4):

        image = Image.new("RGB", (128, 128), color="white")

        image.save(test_good / f"good_{i}.png")

    for i in range(4):

        image = Image.new("RGB", (128, 128), color="red")

        image.save(test_broken / f"broken_{i}.png")

    return root
