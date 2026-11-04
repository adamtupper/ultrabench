"""Unit tests for the dataset loading the training layer does."""

import albumentations as A
import pytest
from datasets import Dataset

from ultrabench.training.data import utils
from ultrabench.training.data.utils import TEST_SPLIT, get_splits, resolve_task_path


@pytest.fixture
def loader(monkeypatch):
    """Replace the dataset loader with a recorder.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list the keyword arguments of each call are appended to.
    """
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append({"name": name, **kwargs})

        return Dataset.from_dict(
            {
                "image": [[[0]]] * 4,
                "target": [0, 1, 0, 1],
                "scan_mask": [[[1]]] * 4,
                "filename": ["a", "b", "c", "d"],
            }
        )

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    return calls


@pytest.fixture
def splits(loader):
    """Call get_splits with the minimum arguments it needs.

    Args:
        loader: The stubbed dataset loader.

    Returns:
        The recorded calls.
    """
    transform = A.Compose([A.NoOp()])
    get_splits(
        name="GBCU",
        input_key="image",
        target_key="target",
        train_transform=transform,
        test_transform=transform,
        path="/raw/GBCU",
        subset="classification",
        revision="0.4",
        cache_dir="/cache",
    )

    return loader


class TestGetSplits:
    """Tests for get_splits."""

    def test_loads_exactly_two_splits(self, splits):
        """The train and test splits are each fetched once."""
        assert len(splits) == 2

    def test_loads_the_train_and_resized_test_splits(self, splits):
        """Training uses the resized test split, as evaluation does."""
        assert [call["split"] for call in splits] == ["train", TEST_SPLIT]

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("name", "GBCU"),
            ("path", "/raw/GBCU"),
            ("subset", "classification"),
            ("revision", "0.4"),
            ("cache_dir", "/cache"),
        ],
    )
    def test_forwards_the_dataset_arguments(self, splits, key, expected):
        """Every argument identifying the dataset reaches the loader."""
        assert all(call[key] == expected for call in splits)


class TestResolveTaskPath:
    """Tests for resolve_task_path."""

    def test_a_task_without_a_path_needs_no_raw_data(self):
        """A hosted dataset's task config names no path."""
        assert resolve_task_path(None, "/data") is None

    def test_resolves_a_relative_path_against_the_data_directory(self):
        """Raw data is staged under the run's data directory."""
        assert resolve_task_path("raw/GBCU-Shared", "/data") == "/data/raw/GBCU-Shared"

    def test_leaves_an_absolute_path_alone(self):
        """A command-line override points wherever it likes."""
        assert resolve_task_path("/scratch/gbcu", "/data") == "/scratch/gbcu"
