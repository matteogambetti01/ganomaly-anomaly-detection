from src.data.transforms import get_transforms
from src.data.dataloaders import create_dataloaders

def test_dataloaders(fake_mvtec_dataset):
    """ Test the dataloaders created by 'create_dataloaders' function. """
    loaders = create_dataloaders(
        root_dir=fake_mvtec_dataset,
        transform=get_transforms(),
        batch_size=4,
        val_ratio=0.2,
        num_workers=0,
        seed=42
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Check that the number of samples in each dataloader is as expected
    assert len(train_loader.dataset) == 16
    assert len(val_loader.dataset) == 4
    assert len(test_loader.dataset) == 8

    # Check the shape of a batch from the training dataloader
    batch = next(iter(train_loader))

    assert batch["image"].shape == (4, 3, 128, 128)

    # Training and validation datasets should be disjoint
    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)

    assert train_indices.isdisjoint(val_indices)

    assert len(train_indices | val_indices) == 20


    # Training and validation datasets should contain only normal images
    assert all(
        train_loader.dataset.dataset[i]["label"] == 0
        for i in train_loader.dataset.indices
    )

    assert all(
        val_loader.dataset.dataset[i]["label"] == 0
        for i in val_loader.dataset.indices
    )
