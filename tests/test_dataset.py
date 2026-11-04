import pytest
from PIL import Image
from src.data.dataset import MVTECDataset


def test_train_dataset(fake_mvtec_dataset):
    """ Test the MVTecDataset class for the training split. """
    dataset = MVTECDataset(
        root_dir=fake_mvtec_dataset,
        split="train"
    )

    assert len(dataset) == 20

    sample = dataset[0]

    assert sample["label"] == 0
    assert isinstance(sample["image"], Image.Image)


def test_test_dataset(fake_mvtec_dataset):
    """ Test the MVTecDataset class for the test split."""
    dataset = MVTECDataset(
        root_dir=fake_mvtec_dataset,
        split="test"
    )

    assert len(dataset) == 8

    labels = [
        dataset[i]["label"]
        for i in range(len(dataset))
    ]

    assert labels.count(0) == 4
    assert labels.count(1) == 4


def test_invalid_split(fake_mvtec_dataset):
    """ Test that providing an invalid split raises a ValueError."""
    with pytest.raises(ValueError):

        MVTECDataset(
            root_dir=fake_mvtec_dataset,
            split="validation"
        )
