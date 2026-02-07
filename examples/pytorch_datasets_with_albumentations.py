"""An example of creating PyTorch datasets that use Albumentations for data
augmentation and preprocessing.
"""

import json
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


class JSON_Dataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None) -> None:
        """Initialize the JSON_Dataset.

        Args:
            root_dir: Directory with all the images.
            split: The split to use, i.e. "train", "validation" or
                "test".
            transform: Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(root_dir, f"{split}.json"), "r") as f:
            self.metadata = json.load(f)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method.")


class ClassificationDataset(JSON_Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform=None,
        image_key: str = "image",
        label_key: str = "label",
        scan_mask_key: str = "scan_mask",
    ) -> None:
        """Initialize the ClassificationDataset.

        Args:
            root_dir: Directory with all the images.
            split: The split to use, i.e. "train", "validation" or "test".
            transform: Optional transform to be applied on a sample.
            image_key: The key for the image path in the metadata.
            label_key: The key for the label in the metadata.
            scan_mask_key: The key for the scan mask path in the metadata.
        """
        super().__init__(root_dir, split, transform)
        self.image_key = image_key
        self.label_key = label_key
        self.scan_mask_key = scan_mask_key

    def __getitem__(self, idx: int) -> tuple[np.ndarray | dict, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, self.metadata[idx][self.image_key])
        image = io.imread(image_path)
        if image.ndim != 3:
            # Convert single channel images to three channel images
            image = np.stack([image] * 3, axis=-1)

        label = self.metadata[idx][self.label_key]

        scan_mask_path = os.path.join(
            self.root_dir, self.metadata[idx][self.scan_mask_key]
        )
        scan_mask = io.imread(scan_mask_path).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, scan_mask=scan_mask)
            return transformed["image"], label
        else:
            return image, label


class SegmentationDataset(JSON_Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform=None,
        image_key: str = "image",
        mask_key: str = "mask",
        scan_mask_key: str = "scan_mask",
    ) -> None:
        """Initialize the SegmentationDataset.

        Args:
            root_dir: Directory with all the images.
            split: The split to use ("train", "validation" or "test").
            transform: Optional transform to be applied on a sample.
            image_key: The key for the image path in the metadata.
            mask_key: The key for the mask path in the metadata.
            scan_mask_key: The key for the scan mask path in the metadata.
        """
        super().__init__(root_dir, split, transform)
        self.image_key = image_key
        self.mask_key = mask_key
        self.scan_mask_key = scan_mask_key

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, self.metadata[idx][self.image_key])
        image = io.imread(image_path)
        if image.ndim != 3:
            # Convert single channel images to three channel images
            image = np.stack([image] * 3, axis=-1)

        mask_path = os.path.join(self.root_dir, self.metadata[idx][self.mask_key])
        mask = io.imread(mask_path).astype(np.uint8)
        if mask.ndim != 3:
            # Labels should have a channel with length equal to one
            mask = np.expand_dims(mask, axis=-1)

        scan_mask_path = os.path.join(
            self.root_dir, self.metadata[idx][self.scan_mask_key]
        )
        scan_mask = io.imread(scan_mask_path).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask, scan_mask=scan_mask)
            return transformed["image"], transformed["mask"]
        else:
            return image, mask
