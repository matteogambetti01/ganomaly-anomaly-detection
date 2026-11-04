import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset import MVTECDataset


def create_dataloaders(
    root_dir,
    transform,
    batch_size=16,
    val_ratio=0.2,
    num_workers=2,
    seed=42
):
    """ Create DataLoaders for trainig, validation and testing. """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    # Original dataset containing only normal images
    full_train_dataset = MVTECDataset(
        root_dir=root_dir,
        split="train",
        transform=transform
    )

    # Calculate the sizes of the splits
    n_val = int(val_ratio * len(full_train_dataset))
    n_train = len(full_train_dataset) - n_val

    if n_train == 0 or n_val == 0:
        raise ValueError("Dataset too small to create train/validation split")

    # Deterministic split using a fixed seed for reproducibility
    split_generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [n_train, n_val],
        generator=split_generator
    )

    # Test dataset containing both normal and anomalous images
    test_dataset = MVTECDataset(
        root_dir=root_dir,
        split="test",
        transform=transform
    )

    # Generator for shuffling the training data
    train_generator = torch.Generator().manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
