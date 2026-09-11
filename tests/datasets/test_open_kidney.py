"""Unit tests for ultrabench2.datasets.open_kidney."""

import csv

import numpy as np
import pytest

from ultrabench2.datasets.open_kidney import load_open_kidney

GROUND_TRUTH_FILES = (
    "ground_truth_segmentation_capsule.csv",
    "ground_truth_segmentation_regions.csv",
)
"""The CSVs each test split carries, one per segmentation task."""


class TestLoadOpenKidney:
    """Tests for load_open_kidney."""

    @pytest.fixture
    def dataset(self, open_kidney_raw, tmp_path):
        """Load the fake raw dataset through the loader.

        Args:
            open_kidney_raw: The fake raw data directory.
            tmp_path: The pytest temporary directory fixture.

        Returns:
            A tuple of the loaded dataset and its cache entry.
        """
        cache = tmp_path / "cache"
        data = load_open_kidney(open_kidney_raw, cache_dir=cache)

        return data, next(cache.glob("OpenKidney/*"))

    def test_returns_the_published_splits(self, dataset):
        """The loader builds the same three splits as the CLI."""
        data, _ = dataset

        assert set(data) == {"train", "test", "test.256"}

    def test_partitions_by_the_manifests(self, dataset):
        """The manifests decide the splits."""
        data, _ = dataset

        assert len(data["train"]) == 3
        assert len(data["test"]) == 2

    def test_drops_the_duplicate_image(self, dataset):
        """A repeated file_id is removed before partitioning."""
        data, _ = dataset
        names = [*data["train"]["filename"], *data["test"]["filename"]]

        assert "9_IM-0001_anon" not in names

    def test_carries_both_masks(self, dataset):
        """Each example holds a capsule and a regions mask."""
        data, _ = dataset

        assert {"capsule_mask", "regions_mask"} <= set(data["train"].column_names)

    def test_keeps_the_regions_mask_multi_class(self, dataset):
        """Resizing the regions mask does not blend its classes."""
        data, _ = dataset
        values = set()
        for example in data["test.256"]:
            values |= set(np.unique(np.asarray(example["regions_mask"])).tolist())

        assert values <= {0, 1, 2, 3}

    @pytest.mark.parametrize("split", ["test", "test.256"])
    def test_writes_both_ground_truth_files(self, dataset, split):
        """Each test split carries one CSV per segmentation task."""
        _, entry = dataset

        for name in GROUND_TRUTH_FILES:
            assert (entry / split / name).is_file()

    def test_ground_truth_covers_every_test_example(self, dataset):
        """The CSVs describe exactly the examples in their split."""
        data, entry = dataset

        for name in GROUND_TRUTH_FILES:
            with (entry / "test" / name).open(newline="") as f:
                rows = list(csv.DictReader(f))

            assert {r["image_id"] for r in rows} == set(data["test"]["filename"])

    def test_caches_the_processed_dataset(self, open_kidney_raw, tmp_path, monkeypatch):
        """A second call reads the cache instead of reprocessing."""
        cache = tmp_path / "cache"
        first = load_open_kidney(open_kidney_raw, cache_dir=cache)

        def fail(*args, **kwargs):
            raise AssertionError("the dataset was reprocessed")

        monkeypatch.setattr(
            "ultrabench2.process.open_kidney.build_open_kidney_dataset", fail
        )
        second = load_open_kidney(open_kidney_raw, cache_dir=cache)

        assert second["train"]["filename"] == first["train"]["filename"]

    def test_rejects_a_missing_raw_directory(self, tmp_path):
        """A path that does not exist is reported."""
        with pytest.raises(FileNotFoundError, match="No raw data for OpenKidney"):
            load_open_kidney(tmp_path / "missing", cache_dir=tmp_path / "cache")


class TestParityWithTheCli:
    """Tests that the CLI and the loader share one implementation."""

    def test_produces_the_same_dataset(self, open_kidney_raw, tmp_path):
        """`ultrabench2 process open-kidney` and the loader agree."""
        from datasets import load_from_disk

        from ultrabench2.process.open_kidney import OUTPUT_NAME, process_open_kidney

        output = tmp_path / "out"
        output.mkdir()
        process_open_kidney(str(open_kidney_raw), str(output))
        processed = load_from_disk(str(output / OUTPUT_NAME))

        loaded = load_open_kidney(open_kidney_raw, cache_dir=tmp_path / "cache")

        assert set(processed) == set(loaded)
        for split in processed:
            assert processed[split]["filename"] == loaded[split]["filename"]

    def test_writes_identical_ground_truth(self, open_kidney_raw, tmp_path):
        """Both paths write byte-identical ground-truth CSVs."""
        from ultrabench2.process.open_kidney import OUTPUT_NAME, process_open_kidney

        output = tmp_path / "out"
        output.mkdir()
        process_open_kidney(str(open_kidney_raw), str(output))

        cache = tmp_path / "cache"
        load_open_kidney(open_kidney_raw, cache_dir=cache)
        entry = next(cache.glob("OpenKidney/*"))

        for split in ("test", "test.256"):
            for name in GROUND_TRUTH_FILES:
                assert (output / OUTPUT_NAME / split / name).read_bytes() == (
                    entry / split / name
                ).read_bytes()
