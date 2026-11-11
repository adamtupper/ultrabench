"""Unit tests for ultrabench.datasets.load_dataset."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

import ultrabench
from ultrabench.datasets import load_dataset
from ultrabench.datasets.registry import _REGISTRY, DatasetType

# `ultrabench.datasets.load_dataset` is the function, which shadows the
# module of the same name, so the module is fetched rather than imported
# and everything below is patched by object reference.
MODULE = import_module("ultrabench.datasets.load_dataset")


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace the Hugging Face loader with a recorder.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list the keyword arguments of each call are appended to.
    """
    calls = []

    def fake_load_dataset(path, **kwargs):
        calls.append({"path": path, **kwargs})

        return DatasetDict({"train": Dataset.from_dict({"a": [1]})})

    monkeypatch.setattr(
        MODULE, "hf_datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )

    return calls


@pytest.fixture
def fake_loaders(monkeypatch):
    """Replace every raw and local loader with a recorder.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A list the keyword arguments of each call are appended to.
    """
    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)

        return DatasetDict(
            {
                "train": Dataset.from_dict({"a": [1, 2]}),
                "test": Dataset.from_dict({"a": [3]}),
                "test.256": Dataset.from_dict({"a": [4]}),
            }
        )

    monkeypatch.setattr(MODULE, "_resolve_loader", lambda config: fake_loader)

    return calls


class TestPublicApi:
    """Tests for how load_dataset is exposed."""

    def test_is_re_exported_from_the_package_root(self):
        """The documented `ultrabench.load_dataset` entry point works."""
        assert ultrabench.load_dataset is load_dataset


class TestName:
    """Tests for the dataset name."""

    def test_rejects_an_unknown_name_and_lists_them(self):
        """An unknown name is reported with every valid one."""
        with pytest.raises(ValueError, match="Unknown dataset 'Nope'.*AUL"):
            load_dataset("Nope")

    def test_the_name_is_checked_before_anything_else(self):
        """A bad name is reported even alongside other bad arguments."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("Nope", subset="x", split="y", path="/missing")


class TestSubset:
    """Tests for the subset argument."""

    def test_requires_a_subset_and_lists_them(self, fake_hub):
        """A dataset with subsets cannot be loaded without one."""
        with pytest.raises(ValueError, match="classification, segmentation"):
            load_dataset("CAMUS")

    def test_rejects_an_unknown_subset(self, fake_hub):
        """An unknown subset is reported."""
        with pytest.raises(ValueError, match="Unknown subset 'nope'"):
            load_dataset("CAMUS", subset="nope")

    def test_rejects_a_subset_for_a_dataset_without_any(self, fake_hub):
        """A dataset with one subset takes no subset argument."""
        with pytest.raises(ValueError, match="AUL has no subsets"):
            load_dataset("AUL", subset="nope")

    def test_forwards_a_valid_subset(self, fake_hub):
        """A valid subset reaches the Hugging Face loader."""
        load_dataset("CAMUS", subset="segmentation")

        assert fake_hub[0]["name"] == "segmentation"


class TestSplit:
    """Tests for the split argument."""

    def test_rejects_an_unknown_split_and_lists_them(self, fake_hub):
        """An unknown split is reported with the valid ones."""
        with pytest.raises(ValueError, match="Unknown split 'test.512'.*test.256"):
            load_dataset("AUL", split="test.512")

    def test_validates_the_split_before_downloading(self, fake_hub):
        """A typo costs nothing, so the check works offline."""
        with pytest.raises(ValueError, match="Unknown split"):
            load_dataset("AUL", split="test.512")

        assert fake_hub == []

    def test_returns_a_dataset_dict_by_default(self, fake_loaders):
        """Omitting the split returns every split."""
        assert isinstance(load_dataset("GBCU", path="."), DatasetDict)

    def test_returns_one_split_when_named(self, fake_loaders):
        """Naming a split returns just that split."""
        data = load_dataset("GBCU", path=".", split="test")

        assert isinstance(data, Dataset)
        assert len(data) == 1


class TestPath:
    """Tests for the raw data path."""

    def test_requires_a_path_for_a_local_dataset(self):
        """A local dataset cannot be loaded without its raw data."""
        with pytest.raises(ValueError, match="GBCU cannot be republished"):
            load_dataset("GBCU")

    @pytest.mark.parametrize("name", ["GBCU", "OpenKidney"])
    def test_both_local_datasets_require_a_path(self, name):
        """Neither local dataset is available on the Hub."""
        with pytest.raises(ValueError, match="cannot be republished"):
            load_dataset(name)

    def test_rejects_a_missing_path_for_a_local_dataset(self, tmp_path):
        """A local dataset's path must exist."""
        with pytest.raises(FileNotFoundError, match="No raw data for GBCU"):
            load_dataset("GBCU", path=tmp_path / "missing")

    def test_warns_that_a_hosted_dataset_ignores_path(self, tmp_path, fake_hub):
        """Passing a path to a hosted dataset warns rather than failing."""
        with pytest.warns(UserWarning, match="AUL is hosted"):
            load_dataset("AUL", path=tmp_path)

    def test_does_not_forward_an_ignored_path(self, tmp_path, fake_hub):
        """The ignored path never reaches the Hugging Face loader."""
        with pytest.warns(UserWarning):
            load_dataset("AUL", path=tmp_path)

        assert "path" not in fake_hub[0] or fake_hub[0]["path"] == "UltraBench/AUL"

    def test_an_ignored_path_need_not_exist(self, tmp_path, fake_hub):
        """A path that is ignored is never checked for existence."""
        with pytest.warns(UserWarning, match="is ignored"):
            data = load_dataset("AUL", path=tmp_path / "missing")

        assert data is not None


class TestRevision:
    """Tests for the revision argument."""

    def test_is_forwarded_for_a_processed_dataset(self, fake_hub):
        """A processed dataset is pinned to the revision requested."""
        load_dataset("AUL", revision="0.4")

        assert fake_hub[0]["revision"] == "0.4"

    def test_is_forwarded_for_a_raw_dataset(self, fake_loaders):
        """A raw dataset pins the revision of its original files."""
        load_dataset("MMOTU", revision="0.4")

        assert fake_loaders[0]["revision"] == "0.4"

    def test_warns_for_a_local_dataset(self, tmp_path, fake_loaders):
        """A local dataset has no revision to pin."""
        with pytest.warns(UserWarning, match="GBCU is processed from local"):
            load_dataset("GBCU", path=tmp_path, revision="0.4")

    def test_is_not_forwarded_for_a_local_dataset(self, tmp_path, fake_loaders):
        """The ignored revision never reaches the loader."""
        with pytest.warns(UserWarning):
            load_dataset("GBCU", path=tmp_path, revision="0.4")

        assert "revision" not in fake_loaders[0]


class TestDispatch:
    """Tests for how each type of dataset is loaded."""

    def test_a_processed_dataset_goes_straight_to_the_hub(self, fake_hub):
        """A processed dataset is downloaded ready to use."""
        load_dataset("AUL", split="train")

        assert fake_hub[0]["path"] == "UltraBench/AUL"
        assert fake_hub[0]["split"] == "train"

    def test_a_raw_dataset_is_loaded_from_its_raw_repository(self, fake_loaders):
        """A raw dataset's loader is given the raw repository."""
        load_dataset("COVID-BLUES")

        assert fake_loaders[0]["hub_id"] == "UltraBench/COVID-BLUES-raw"

    def test_a_local_dataset_is_loaded_from_its_path(self, tmp_path, fake_loaders):
        """A local dataset's loader is given the resolved raw path."""
        load_dataset("OpenKidney", path=tmp_path)

        assert fake_loaders[0]["path"] == tmp_path

    def test_every_registered_dataset_dispatches(
        self, tmp_path, fake_hub, fake_loaders
    ):
        """No registry entry is missing a way to be loaded."""
        for name, config in _REGISTRY.items():
            kwargs = {"subset": config.subsets[0]} if config.subsets else {}
            if config.type is DatasetType.LOCAL:
                kwargs["path"] = tmp_path

            assert load_dataset(name, **kwargs) is not None


class TestCacheDir:
    """Tests for the cache directory."""

    def test_leaves_the_hugging_face_cache_alone_by_default(
        self, fake_hub, monkeypatch
    ):
        """Without a cache directory, HF_HOME stays in charge."""
        monkeypatch.delenv("ULTRABENCH_CACHE", raising=False)
        load_dataset("AUL")

        assert fake_hub[0]["cache_dir"] is None

    def test_keeps_an_explicit_cache_directory_self_contained(self, tmp_path, fake_hub):
        """An explicit cache directory also holds the Hub downloads."""
        load_dataset("AUL", cache_dir=tmp_path)

        assert fake_hub[0]["cache_dir"] == str(tmp_path / "_hub")

    def test_forwards_the_cache_directory_to_a_loader(self, tmp_path, fake_loaders):
        """A raw or local loader caches where it was told to."""
        load_dataset("GBCU", path=tmp_path, cache_dir=tmp_path / "cache")

        assert fake_loaders[0]["cache_dir"] == tmp_path / "cache"
