"""Unit tests for ultrabench2.datasets.mmotu."""

import pytest

from ultrabench2.datasets.mmotu import NAME, load_mmotu

GROUND_TRUTH_FILES = (
    "ground_truth_classification.csv",
    "ground_truth_segmentation.csv",
)
"""The CSVs each test split carries, one per task."""


@pytest.fixture
def snapshot(mmotu_raw, monkeypatch):
    """Serve the fake raw tree in place of a Hub download.

    Args:
        mmotu_raw: The fake raw data directory.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list of the arguments each download was called with.
    """
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})

        return str(mmotu_raw)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    return calls


class TestLoadMmotu:
    """Tests for load_mmotu."""

    @pytest.fixture
    def dataset(self, snapshot, tmp_path):
        """Load the fake raw dataset through the loader.

        Args:
            snapshot: The stubbed Hub download.
            tmp_path: The pytest temporary directory fixture.

        Returns:
            A tuple of the loaded dataset and its cache entry.
        """
        cache = tmp_path / "cache"
        data = load_mmotu("UltraBench/MMOTU-raw", cache_dir=cache)

        return data, next(cache.glob(f"{NAME}/*"))

    def test_returns_the_published_splits(self, dataset):
        """The loader builds the same three splits as the CLI."""
        data, _ = dataset

        assert set(data) == {"train", "test", "test.256"}

    def test_partitions_by_the_manifests(self, dataset):
        """The manifests decide the splits."""
        data, _ = dataset

        assert len(data["train"]) == 3
        assert len(data["test"]) == 2

    def test_reads_the_two_space_separated_labels(self, dataset):
        """The class label files parse rather than collapsing to one column."""
        data, _ = dataset

        assert set(data["train"]["tumor_type"]) == {0, 1, 2}

    @pytest.mark.parametrize("split", ["test", "test.256"])
    def test_writes_both_ground_truth_files(self, dataset, split):
        """Each test split carries one CSV per task."""
        _, entry = dataset

        for name in GROUND_TRUTH_FILES:
            assert (entry / split / name).is_file()

    def test_downloads_the_raw_repository(self, snapshot, tmp_path):
        """The loader fetches the repository it was given."""
        load_mmotu("UltraBench/MMOTU-raw", cache_dir=tmp_path / "cache")

        assert snapshot[0]["repo_id"] == "UltraBench/MMOTU-raw"
        assert snapshot[0]["repo_type"] == "dataset"

    def test_forwards_the_revision(self, snapshot, tmp_path):
        """A pinned revision reaches the download."""
        load_mmotu("UltraBench/MMOTU-raw", cache_dir=tmp_path / "cache", revision="0.4")

        assert snapshot[0]["revision"] == "0.4"

    def test_a_warm_cache_never_downloads(self, snapshot, tmp_path):
        """A cached dataset works without network access."""
        cache = tmp_path / "cache"
        load_mmotu("UltraBench/MMOTU-raw", cache_dir=cache)
        load_mmotu("UltraBench/MMOTU-raw", cache_dir=cache)

        assert len(snapshot) == 1

    def test_a_different_revision_gets_its_own_cache_entry(self, snapshot, tmp_path):
        """Two revisions do not share a cache entry."""
        cache = tmp_path / "cache"
        load_mmotu("UltraBench/MMOTU-raw", cache_dir=cache, revision="0.4")
        load_mmotu("UltraBench/MMOTU-raw", cache_dir=cache, revision="0.5")

        assert len(list((cache / NAME).iterdir())) == 2


class TestParityWithTheCli:
    """Tests that the CLI and the loader share one implementation."""

    def test_produces_the_same_dataset(self, snapshot, mmotu_raw, tmp_path):
        """`ultrabench2 process mmotu` and the loader agree."""
        from datasets import load_from_disk

        from ultrabench2.process.mmotu import OUTPUT_NAME, process_mmotu

        output = tmp_path / "out"
        output.mkdir()
        process_mmotu(str(mmotu_raw), str(output))
        processed = load_from_disk(str(output / OUTPUT_NAME))

        loaded = load_mmotu("UltraBench/MMOTU-raw", cache_dir=tmp_path / "cache")

        assert set(processed) == set(loaded)
        for split in processed:
            assert processed[split]["filename"] == loaded[split]["filename"]
            assert processed[split]["tumor_type"] == loaded[split]["tumor_type"]

    def test_writes_identical_ground_truth(self, snapshot, mmotu_raw, tmp_path):
        """Both paths write byte-identical ground-truth CSVs."""
        from ultrabench2.process.mmotu import OUTPUT_NAME, process_mmotu

        output = tmp_path / "out"
        output.mkdir()
        process_mmotu(str(mmotu_raw), str(output))

        cache = tmp_path / "cache"
        load_mmotu("UltraBench/MMOTU-raw", cache_dir=cache)
        entry = next(cache.glob(f"{NAME}/*"))

        for split in ("test", "test.256"):
            for name in GROUND_TRUTH_FILES:
                assert (output / OUTPUT_NAME / split / name).read_bytes() == (
                    entry / split / name
                ).read_bytes()
