"""An example of creating PyTorch datasets that use Albumentations for data
augmentation and preprocessing.
"""

import glob
import json
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


class JSON_Dataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        """Initialize the JSON_Dataset.

        Args:
            root_dir: Path to the split directory (e.g.,
                ``dataset_v1/train_val/`` or ``dataset_v1/test_256/``).
                Must contain exactly one JSON metadata file.
            transform: Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        json_files = glob.glob(os.path.join(root_dir, "*.json"))
        if len(json_files) != 1:
            raise ValueError(f"More than one JSON file found in {root_dir}.")
        with open(json_files[0]) as f:
            self.metadata = json.load(f)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        raise NotImplementedError("Subclasses must implement this method.")


class ClassificationDataset(JSON_Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        image_key: str = "image",
        label_key: str = "label",
        scan_mask_key: str = "scan_mask",
    ) -> None:
        """Initialize the ClassificationDataset.

        Args:
            root_dir: Path to the split directory (e.g.,
                ``dataset_v1/train_val/`` or ``dataset_v1/test_256/``).
                Must contain exactly one JSON metadata file.
            transform: Optional transform to be applied on a sample.
            image_key: The key for the image path in the metadata.
            label_key: The key for the label in the metadata.
            scan_mask_key: The key for the scan mask path in the metadata.
        """
        super().__init__(root_dir, transform)
        self.image_key = image_key
        self.label_key = label_key
        self.scan_mask_key = scan_mask_key

    def __getitem__(self, idx: int) -> tuple[np.ndarray | dict, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()  # ty:ignore[invalid-assignment]

        image_path = os.path.join(self.root_dir, self.metadata[idx][self.image_key])
        image = io.imread(image_path)

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
        transform=None,
        image_key: str = "image",
        mask_key: str = "mask",
        scan_mask_key: str = "scan_mask",
    ) -> None:
        """Initialize the SegmentationDataset.

        Args:
            root_dir: Path to the split directory (e.g.,
                ``dataset_v1/train_val/`` or ``dataset_v1/test_256/``).
                Must contain exactly one JSON metadata file.
            transform: Optional transform to be applied on a sample.
            image_key: The key for the image path in the metadata.
            mask_key: The key for the mask path in the metadata.
            scan_mask_key: The key for the scan mask path in the metadata.
        """
        super().__init__(root_dir, transform)
        self.image_key = image_key
        self.mask_key = mask_key
        self.scan_mask_key = scan_mask_key

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()  # ty:ignore[invalid-assignment]

        image_path = os.path.join(self.root_dir, self.metadata[idx][self.image_key])
        image = io.imread(image_path)

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
