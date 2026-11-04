"""Unit tests for ultrabench.datasets.covid_blues."""

import csv

import pytest

from ultrabench.datasets.covid_blues import NAME, load_covid_blues


@pytest.fixture
def snapshot(covid_blues_raw, monkeypatch):
    """Serve the fake raw tree in place of a Hub download.

    Args:
        covid_blues_raw: The fake raw data directory.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list of the arguments each download was called with.
    """
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})

        return str(covid_blues_raw)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    return calls


class TestLoadCovidBlues:
    """Tests for load_covid_blues."""

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
        data = load_covid_blues("UltraBench/COVID-BLUES-raw", cache_dir=cache)

        return data, next(cache.glob(f"{NAME}/*"))

    def test_returns_the_published_splits(self, dataset):
        """The loader builds the same three splits as the CLI."""
        data, _ = dataset

        assert set(data) == {"train", "test", "test.256"}

    def test_expands_the_video_manifest_into_frames(self, dataset):
        """Each listed video contributes all of its frames."""
        data, _ = dataset

        assert len(data["train"]) == 4 * 3
        assert len(data["test"]) == 2 * 3

    def test_keeps_patients_out_of_both_splits(self, dataset):
        """The manifests preserve the patient-disjoint split."""
        data, _ = dataset

        assert not set(data["train"]["patient"]) & set(data["test"]["patient"])

    def test_names_each_frame_after_its_video(self, dataset):
        """A frame's filename is its video stem and frame index."""
        data, _ = dataset

        assert "patient_3_L1_0" in data["test"]["filename"]

    def test_materialises_the_streamed_train_split(self, dataset):
        """The lazily generated train split survives the cache."""
        data, _ = dataset

        assert data["train"][0]["image"] is not None

    def test_writes_ground_truth_with_a_sequence_id(self, dataset):
        """The label applies per patient, so predictions can be pooled."""
        _, entry = dataset

        with (entry / "test" / "ground_truth_classification.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert "sequence_id" in rows[0]
        assert len({r["sequence_id"] for r in rows}) == 1

    def test_never_decodes_an_unlisted_video(self, snapshot, tmp_path, monkeypatch):
        """A video in neither manifest costs nothing to skip."""
        from ultrabench.process import covid_blues

        decoded = []
        original = covid_blues.extract_frames

        def counting_extract_frames(video_path):
            decoded.append(video_path)

            return original(video_path)

        monkeypatch.setattr(covid_blues, "extract_frames", counting_extract_frames)
        load_covid_blues("UltraBench/COVID-BLUES-raw", cache_dir=tmp_path / "cache")

        # Each test video is decoded twice, once per test split.
        assert len(decoded) == 4 + 2 * 2

    def test_a_warm_cache_never_downloads(self, snapshot, tmp_path):
        """A cached dataset works without network access."""
        cache = tmp_path / "cache"
        load_covid_blues("UltraBench/COVID-BLUES-raw", cache_dir=cache)
        load_covid_blues("UltraBench/COVID-BLUES-raw", cache_dir=cache)

        assert len(snapshot) == 1


class TestParityWithTheCli:
    """Tests that the CLI and the loader share one implementation."""

    def test_produces_the_same_dataset(self, snapshot, covid_blues_raw, tmp_path):
        """`ultrabench process covid-blues` and the loader agree."""
        from datasets import load_from_disk

        from ultrabench.process.covid_blues import OUTPUT_NAME, process_covid_blues

        output = tmp_path / "out"
        output.mkdir()
        process_covid_blues(str(covid_blues_raw), str(output))
        processed = load_from_disk(str(output / OUTPUT_NAME))

        loaded = load_covid_blues(
            "UltraBench/COVID-BLUES-raw", cache_dir=tmp_path / "cache"
        )

        assert set(processed) == set(loaded)
        for split in processed:
            assert sorted(processed[split]["filename"]) == sorted(
                loaded[split]["filename"]
            )

    def test_writes_identical_ground_truth(self, snapshot, covid_blues_raw, tmp_path):
        """Both paths write byte-identical ground-truth CSVs."""
        from ultrabench.process.covid_blues import OUTPUT_NAME, process_covid_blues

        output = tmp_path / "out"
        output.mkdir()
        process_covid_blues(str(covid_blues_raw), str(output))

        cache = tmp_path / "cache"
        load_covid_blues("UltraBench/COVID-BLUES-raw", cache_dir=cache)
        entry = next(cache.glob(f"{NAME}/*"))

        name = "ground_truth_classification.csv"
        for split in ("test", "test.256"):
            assert (output / OUTPUT_NAME / split / name).read_bytes() == (
                entry / split / name
            ).read_bytes()
