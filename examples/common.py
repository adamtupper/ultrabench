"""Helpers shared by the example training scripts.

Everything here is plain PyTorch on top of the `ultrabench` package:
loading a task's splits, turning them into tensors, building a backbone
from its class path, and scoring a predictions CSV with the evaluation
layer.
"""

import argparse
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset as TorchDataset

from ultrabench import load_dataset
from ultrabench.datasets.builders.utils import resize_and_pad
from ultrabench.evaluate import ClassificationEvaluator, SegmentationEvaluator
from ultrabench.evaluate.protocol import GROUND_TRUTH_FILES, ground_truth_path
from ultrabench.evaluate.tasks import TaskSpec, get_task
from ultrabench.models.backbone import Backbone

TEST_SPLIT = "test.256"
"""The resized test split, which the published ground truth describes."""


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments both example scripts accept.

    Args:
        parser: The parser to add the arguments to.
    """
    parser.add_argument("--task", required=True, help="e.g. tn3k_segmentation")
    parser.add_argument(
        "--backbone",
        required=True,
        help="Import path of a Backbone subclass, e.g. "
        "ultrabench.models.dinov3.dinov3_backbone.DINOv3ViTBackbone",
    )
    parser.add_argument(
        "--backbone-kwargs",
        type=json.loads,
        default={},
        help="JSON object of keyword arguments for the backbone constructor.",
    )
    parser.add_argument(
        "--path", help="Raw data directory, for local datasets (GBCU, OpenKidney)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--max-steps", type=int, help="Stop after this many steps (for smoke tests)."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_backbone(
    class_path: str, kwargs: dict, output_intermediate_features: bool = False
) -> Backbone:
    """Instantiate a backbone from its import path.

    Args:
        class_path: The dotted import path of a `Backbone` subclass.
        kwargs: Keyword arguments for the backbone's constructor.
        output_intermediate_features: Whether the backbone should also
            return the features of its intermediate blocks or stages.

    Returns:
        The backbone, in evaluation mode with its parameters frozen.

    Raises:
        TypeError: If the class is not a `Backbone` subclass.
    """
    module_name, _, class_name = class_path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Backbone)):
        raise TypeError(f"{class_path} is not a subclass of Backbone.")

    backbone = cls(**kwargs, output_intermediate_features=output_intermediate_features)
    backbone.eval()
    backbone.requires_grad_(False)

    return backbone


def load_splits(
    task: TaskSpec, path: str | None, val_size: float, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    """Load the train, validation and test splits of a task.

    The validation set is carved out of the published `train` split,
    keeping each group (e.g., patient) on one side when the task names
    a `group_by` column.

    Args:
        task: The task to load.
        path: The raw data directory, for local datasets.
        val_size: The proportion of the `train` split to validate on.
        seed: The random seed for the train/validation split.

    Returns:
        The train, validation and test splits.

    Raises:
        ValueError: If the task's config does not name its dataset.
    """
    name = task.dataset_name
    if name is None:
        raise ValueError(f"The config of {task.name} does not name its dataset.")

    def load(split: str) -> Dataset:
        data = load_dataset(
            name,
            path=path,
            subset=task.subset,
            split=split,
            revision=task.revision,
        )
        assert isinstance(data, Dataset), "a single split loads as a Dataset"
        return data

    train_val, test = load("train"), load(TEST_SPLIT)

    if task.filter_by is not None:
        train_val = train_val.filter(
            lambda value: value is not None, input_columns=task.filter_by
        )

    if task.group_by is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        groups = train_val[task.group_by]
        train_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
        return train_val.select(train_idx), train_val.select(val_idx), test

    splits = train_val.train_test_split(
        test_size=val_size, stratify_by_column=task.stratify_by, seed=seed
    )

    return splits["train"], splits["test"], test


class TaskDataset(TorchDataset):
    """Serve a task's images and targets as tensors.

    Images are resized and padded to the size of the test split with
    `resize_and_pad`, the function that built the published `test.256`
    split. Resizing training images any other way (e.g., with OpenCV)
    leaves the test set looking like a different domain.

    Args:
        data: A split returned by `load_splits`.
        task: The task the split belongs to.
    """

    def __init__(self, data: Dataset, task: TaskSpec) -> None:
        self.data = data.select_columns(["filename", task.input_key, task.target_key])
        self.task = task

    def __len__(self) -> int:
        """Return the number of examples.

        Returns:
            The number of examples in the split.
        """
        return len(self.data)

    def _resize(self, array: np.ndarray, order: int) -> np.ndarray:
        """Resize and pad an array to the test image size.

        Args:
            array: An image or mask.
            order: The interpolation order, 1 for images and 0 for masks.

        Returns:
            The array, resized and padded unless it already has the size.
        """
        size = self.task.image_size
        if array.shape[:2] == (size, size):
            return array

        return resize_and_pad(array, size, order=order)

    def __getitem__(self, index: int) -> dict:
        """Load one example.

        Args:
            index: The index of the example.

        Returns:
            A dict with a uint8 "image" of shape [3, H, W], the "target"
            (a class index, or a mask of class indices of shape [H, W])
            and the "image_id".
        """
        row = self.data[index]
        image = self._resize(np.array(row[self.task.input_key].convert("RGB")), 1)
        target = row[self.task.target_key]
        if self.task.task_type == "segmentation":
            target = torch.from_numpy(self._resize(np.array(target), 0)).long()

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "target": target,
            "image_id": row["filename"],
        }


def score(task_name: str, predictions_path: Path, output_dir: Path) -> None:
    """Score a predictions CSV against the task's published ground truth.

    Args:
        task_name: The name of the task.
        predictions_path: The predictions CSV.
        output_dir: The directory to write the metrics to.
    """
    if task_name not in GROUND_TRUTH_FILES:
        print(f"{task_name} has no published ground truth; skipping scoring.")
        return

    task = get_task(task_name)
    if task.task_type == "classification":
        evaluator = ClassificationEvaluator(
            ground_truth_path(task_name),
            predictions_path,
            output_dir,
            aggregate_by="sequence_id" if task.sequence_by else None,
        )
    else:
        evaluator = SegmentationEvaluator(
            ground_truth_path(task_name),
            predictions_path,
            output_dir,
            image_height=task.image_size,
            image_width=task.image_size,
        )

    for metric, value in evaluator.evaluate().items():
        print(f"{metric}: {value:.4f}")
