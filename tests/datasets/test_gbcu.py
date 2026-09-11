"""Unit tests for ultrabench2.datasets.gbcu."""

import csv

import pytest

from ultrabench2.datasets.gbcu import load_gbcu


class TestLoadGbcu:
    """Tests for load_gbcu."""

    @pytest.fixture
    def dataset(self, gbcu_raw, tmp_path):
        """Load the fake raw dataset through the loader.

        Args:
            gbcu_raw: The fake raw data directory.
            tmp_path: The pytest temporary directory fixture.

        Returns:
            A tuple of the loaded dataset and the cache directory.
        """
        cache = tmp_path / "cache"

        return load_gbcu(gbcu_raw, cache_dir=cache), cache

    def test_returns_the_published_splits(self, dataset):
        """The loader builds the same three splits as the CLI."""
        data, _ = dataset

        assert set(data) == {"train", "test", "test.256"}

    def test_partitions_by_the_manifests(self, dataset):
        """The manifests decide the splits, dropping the surplus."""
        data, _ = dataset

        assert len(data["train"]) == 4
        assert len(data["test"]) == 2
        assert "im00099" not in data["train"]["filename"]

    def test_builds_a_non_empty_scan_mask(self, dataset):
        """The scan mask marks the region the image holds signal in."""
        import numpy as np

        data, _ = dataset

        assert np.asarray(data["train"][0]["scan_mask"]).any()

    def test_resizes_the_resized_test_split(self, dataset):
        """The test.256 split is square at the standard size."""
        data, _ = dataset

        assert data["test.256"][0]["image"].size == (256, 256)

    def test_leaves_the_test_split_at_its_original_size(self, dataset):
        """The test split keeps the original resolution."""
        data, _ = dataset

        assert data["test"][0]["image"].size == (300, 220)

    def test_writes_ground_truth_for_both_test_splits(self, dataset):
        """Each test split gets a ground-truth CSV in the cache."""
        data, cache = dataset
        entry = next(cache.glob("GBCU/*"))

        for split in ("test", "test.256"):
            path = entry / split / "ground_truth_classification.csv"
            assert path.is_file()

            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))

            assert [r["image_id"] for r in rows] == data[split]["filename"]

    def test_caches_the_processed_dataset(self, gbcu_raw, tmp_path, monkeypatch):
        """A second call reads the cache instead of reprocessing."""
        cache = tmp_path / "cache"
        first = load_gbcu(gbcu_raw, cache_dir=cache)

        def fail(*args, **kwargs):
            raise AssertionError("the dataset was reprocessed")

        monkeypatch.setattr("ultrabench2.process.gbcu.build_gbcu_dataset", fail)
        second = load_gbcu(gbcu_raw, cache_dir=cache)

        assert second["train"]["filename"] == first["train"]["filename"]

    def test_different_raw_data_gets_its_own_cache_entry(self, gbcu_raw, tmp_path):
        """Two raw directories do not share a cache entry."""
        import shutil

        cache = tmp_path / "cache"
        load_gbcu(gbcu_raw, cache_dir=cache)

        other = tmp_path / "other"
        shutil.copytree(gbcu_raw, other)
        load_gbcu(other, cache_dir=cache)

        assert len(list((cache / "GBCU").iterdir())) == 2

    def test_the_same_raw_data_shares_a_cache_entry(self, gbcu_raw, tmp_path):
        """Two spellings of one directory resolve to one entry."""
        cache = tmp_path / "cache"
        load_gbcu(gbcu_raw, cache_dir=cache)
        load_gbcu(gbcu_raw / "imgs" / "..", cache_dir=cache)

        assert len(list((cache / "GBCU").iterdir())) == 1

    def test_leaves_no_staging_directory_behind(self, dataset):
        """A completed build cleans up after itself."""
        _, cache = dataset

        assert not list((cache / "GBCU").glob(".tmp-*"))

    def test_rejects_a_missing_raw_directory(self, tmp_path):
        """A path that does not exist is reported."""
        with pytest.raises(FileNotFoundError, match="No raw data for GBCU"):
            load_gbcu(tmp_path / "missing", cache_dir=tmp_path / "cache")

    def test_reports_raw_data_missing_a_listed_example(self, gbcu_raw, tmp_path):
        """Raw data that lost an example the manifest lists is an error."""
        (gbcu_raw / "train.txt").write_text("im00000.jpg,0\n")

        with pytest.raises(ValueError, match=r"missing 3 example\(s\)"):
            load_gbcu(gbcu_raw, cache_dir=tmp_path / "cache")


class TestParityWithTheCli:
    """Tests that the CLI and the loader share one implementation."""

    def test_produces_the_same_dataset(self, gbcu_raw, tmp_path):
        """`ultrabench2 process gbcu` and load_gbcu agree exactly."""
        from datasets import load_from_disk

        from ultrabench2.process.gbcu import OUTPUT_NAME, process_gbcu

        output = tmp_path / "out"
        output.mkdir()
        process_gbcu(str(gbcu_raw), str(output))
        processed = load_from_disk(str(output / OUTPUT_NAME))

        loaded = load_gbcu(gbcu_raw, cache_dir=tmp_path / "cache")

        assert set(processed) == set(loaded)
        for split in processed:
            assert processed[split]["filename"] == loaded[split]["filename"]
            assert processed[split]["label"] == loaded[split]["label"]

    def test_writes_identical_ground_truth(self, gbcu_raw, tmp_path):
        """Both paths write byte-identical ground-truth CSVs."""
        from ultrabench2.process.gbcu import OUTPUT_NAME, process_gbcu

        output = tmp_path / "out"
        output.mkdir()
        process_gbcu(str(gbcu_raw), str(output))

        cache = tmp_path / "cache"
        load_gbcu(gbcu_raw, cache_dir=cache)
        entry = next(cache.glob("GBCU/*"))

        for split in ("test", "test.256"):
            name = "ground_truth_classification.csv"
            assert (output / OUTPUT_NAME / split / name).read_bytes() == (
                entry / split / name
            ).read_bytes()
