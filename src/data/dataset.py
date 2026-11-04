from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class MVTECDataset(Dataset):
    """ Custom Dataset for the MVTec Anomaly detection dataset. """
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, root_dir, split, transform=None):

        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform

        if self.split not in {"train", "test"}:
            raise ValueError(
                "split must be 'train' or 'test'"
            )

        self.image_paths = []

        if self.split == "train":

            img_dir = self.root_dir / "train" / "good"

            if not img_dir.exists():
                raise FileNotFoundError(
                    f"Directory not found: {img_dir}"
                )

            self.image_paths = sorted(
                [
                    path for path in img_dir.iterdir()
                    if path.suffix.lower() in self.VALID_EXTENSIONS
                ]
            )

        else:

            test_dir = self.root_dir / "test"

            if not test_dir.exists():
                raise FileNotFoundError(
                    f"Directory not found: {test_dir}"
                )

            for category_dir in sorted(test_dir.iterdir()):

                if category_dir.is_dir():

                    images = [
                        path for path in category_dir.iterdir()
                        if path.suffix.lower() in self.VALID_EXTENSIONS
                    ]

                    self.image_paths.extend(sorted(images))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found for split '{self.split}'"
            )

    def __len__(self):
        """ Return the number of images in the dataset. """
        return len(self.image_paths)

    def __getitem__(self, index):
        """ Get an image and its label. """
        img_path = self.image_paths[index]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.split == "train":
            label = 0
        else:
            label = int(img_path.parent.name != "good")

        return {
            "image": image,
            "label": label,
            "image_path": str(img_path)
        }
