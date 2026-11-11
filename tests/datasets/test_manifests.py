"""Unit tests for the tracked split manifests."""

import csv

import pytest

from ultrabench.datasets.utils import (
    MANIFEST_DIR,
    load_split_manifest,
    partition_by_manifest,
)

EXPECTED_ROWS = {
    ("GBCU", "train"): 1000,
    ("GBCU", "test"): 122,
    ("OpenKidney", "train"): 411,
    ("OpenKidney", "test"): 103,
    ("MMOTU", "train"): 1000,
    ("MMOTU", "test"): 469,
    ("COVID-BLUES", "train"): 247,
    ("COVID-BLUES", "test"): 115,
}
"""The row counts of revision 0.4, which the manifests must preserve."""

DATASETS = sorted({name for name, _ in EXPECTED_ROWS})


def read_rows(name: str, split: str) -> list[str]:
    """Read a manifest without de-duplicating or reordering it.

    Args:
        name: The registry name of the dataset.
        split: The split to read.

    Returns:
        The filenames in the order they appear in the file.
    """
    path = MANIFEST_DIR / name / f"{split}.csv"
    with path.open(newline="") as f:
        return [row["filename"] for row in csv.DictReader(f)]


class TestManifestFiles:
    """Tests for the manifests committed to the package."""

    @pytest.mark.parametrize(("name", "split"), sorted(EXPECTED_ROWS))
    def test_row_count_matches_revision_0_4(self, name, split):
        """Each manifest holds the number of examples revision 0.4 has."""
        assert len(read_rows(name, split)) == EXPECTED_ROWS[(name, split)]

    @pytest.mark.parametrize(("name", "split"), sorted(EXPECTED_ROWS))
    def test_entries_are_unique_and_sorted(self, name, split):
        """Each manifest lists every example once, in sorted order."""
        rows = read_rows(name, split)

        assert len(set(rows)) == len(rows)
        assert rows == sorted(rows)

    @pytest.mark.parametrize("name", DATASETS)
    def test_splits_are_disjoint(self, name):
        """No example appears in both the train and test manifests."""
        train = load_split_manifest(name, "train")
        test = load_split_manifest(name, "test")

        assert not train & test

    @pytest.mark.parametrize("name", DATASETS)
    def test_manifest_is_packaged(self, name):
        """Both manifests ship inside the package."""
        for split in ("train", "test"):
            assert (MANIFEST_DIR / name / f"{split}.csv").is_file()

    def test_covid_blues_is_listed_at_video_level(self):
        """COVID-BLUES lists video stems rather than frame filenames."""
        train = load_split_manifest("COVID-BLUES", "train")

        assert "patient_11_L1" in train
        assert not any(name.endswith("_0") for name in train)


class TestLoadSplitManifest:
    """Tests for load_split_manifest."""

    def test_reads_the_named_split(self):
        """A known example is read from the manifest it belongs to."""
        assert "im00001" in load_split_manifest("GBCU", "train")

    def test_raises_for_a_dataset_without_a_manifest(self):
        """A dataset with no manifest is reported, not silently empty."""
        with pytest.raises(FileNotFoundError, match="No train manifest for AUL"):
            load_split_manifest("AUL", "train")


class TestPartitionByManifest:
    """Tests for partition_by_manifest."""

    @pytest.fixture
    def manifest(self, tmp_path, monkeypatch):
        """Point the manifest helpers at a small synthetic dataset.

        Args:
            tmp_path: The pytest temporary directory fixture.
            monkeypatch: The pytest monkeypatch fixture.

        Returns:
            The registry name of the synthetic dataset.
        """
        directory = tmp_path / "Fake"
        directory.mkdir()
        (directory / "train.csv").write_text("filename\na\nb\n")
        (directory / "test.csv").write_text("filename\nc\n")
        monkeypatch.setattr("ultrabench.datasets.utils.MANIFEST_DIR", tmp_path)

        return "Fake"

    def test_partitions_and_drops_unlisted_items(self, manifest):
        """Items in neither manifest are dropped rather than kept."""
        items = [{"filename": name} for name in ("a", "b", "c", "unlisted")]

        train, test = partition_by_manifest(items, manifest, lambda r: r["filename"])

        assert [r["filename"] for r in train] == ["a", "b"]
        assert [r["filename"] for r in test] == ["c"]

    def test_preserves_the_given_order(self, manifest):
        """Each part keeps the order the items were given in."""
        items = [{"filename": name} for name in ("b", "c", "a")]

        train, _ = partition_by_manifest(items, manifest, lambda r: r["filename"])

        assert [r["filename"] for r in train] == ["b", "a"]

    def test_honours_a_custom_key(self, manifest):
        """A dataset listed at video level matches on the video stem."""
        items = [
            {"filename": "a_0", "video": "a"},
            {"filename": "c_0", "video": "c"},
            {"filename": "b_7", "video": "b"},
        ]

        train, test = partition_by_manifest(items, manifest, lambda r: r["video"])

        assert [r["filename"] for r in train] == ["a_0", "b_7"]
        assert [r["filename"] for r in test] == ["c_0"]

    def test_raises_when_the_raw_data_is_missing_an_example(self, manifest):
        """A manifest entry with no raw example is an error."""
        items = [{"filename": "a"}]

        with pytest.raises(ValueError, match=r"missing 2 example\(s\).*b, c"):
            partition_by_manifest(items, manifest, lambda r: r["filename"])

    def test_partitions_the_real_manifests(self):
        """Every example of a real dataset lands in exactly one split."""
        names = load_split_manifest("GBCU", "train") | load_split_manifest(
            "GBCU", "test"
        )
        items = [{"filename": name} for name in sorted(names)]

        train, test = partition_by_manifest(items, "GBCU", lambda r: r["filename"])

        assert (len(train), len(test)) == (1000, 122)
