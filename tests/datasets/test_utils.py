"""Unit tests for ultrabench2.datasets.utils."""

from pathlib import Path

import pytest

from ultrabench2.datasets.utils import (
    CACHE_ENV_VAR,
    dataset_cache_key,
    dataset_cache_path,
    hub_cache_dir,
    is_cached,
    resolve_cache_dir,
    select_split,
    validate_raw_path,
    validate_split,
    validate_subset,
    warn_ignored_path,
    warn_ignored_revision,
)


@pytest.fixture
def no_cache_env(monkeypatch):
    """Remove the cache environment variable.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)


class TestResolveCacheDir:
    """Tests for resolve_cache_dir."""

    def test_prefers_the_explicit_argument(self, tmp_path, monkeypatch):
        """An explicit directory wins over the environment."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))

        assert resolve_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"

    def test_falls_back_to_the_environment(self, tmp_path, monkeypatch):
        """The environment variable is used when no argument is given."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))

        assert resolve_cache_dir() == tmp_path / "env"

    def test_falls_back_to_the_home_directory(self, no_cache_env, monkeypatch):
        """A hidden directory in the home directory is the last resort."""
        monkeypatch.setattr(
            "ultrabench2.datasets.utils.DEFAULT_CACHE_DIR", Path("/home/x/.ultrabench")
        )

        assert resolve_cache_dir() == Path("/home/x/.ultrabench")

    def test_expands_a_user_relative_path(self, no_cache_env):
        """A leading tilde is expanded."""
        assert resolve_cache_dir("~/cache") == Path.home() / "cache"

    def test_expands_a_user_relative_environment_variable(self, monkeypatch):
        """A tilde in the environment variable is expanded too."""
        monkeypatch.setenv(CACHE_ENV_VAR, "~/cache")

        assert resolve_cache_dir() == Path.home() / "cache"

    def test_does_not_create_the_directory(self, tmp_path, no_cache_env):
        """Resolving a path has no side effect on the filesystem."""
        resolve_cache_dir(tmp_path / "missing")

        assert not (tmp_path / "missing").exists()


class TestHubCacheDir:
    """Tests for hub_cache_dir."""

    def test_leaves_hf_home_alone_by_default(self, no_cache_env):
        """No cache directory means the Hugging Face cache is untouched."""
        assert hub_cache_dir() is None

    def test_is_self_contained_when_given_a_directory(self, tmp_path, no_cache_env):
        """An explicit cache directory also holds the Hub downloads."""
        assert hub_cache_dir(tmp_path) == tmp_path / "_hub"

    def test_follows_the_environment_variable(self, tmp_path, monkeypatch):
        """Setting the environment variable redirects Hub downloads."""
        monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))

        assert hub_cache_dir() == tmp_path / "_hub"


class TestDatasetCacheKey:
    """Tests for dataset_cache_key."""

    def test_is_stable_for_equal_inputs(self):
        """The same inputs always give the same key."""
        assert dataset_cache_key(name="GBCU") == dataset_cache_key(name="GBCU")

    def test_is_twelve_hex_characters(self):
        """The key is short enough to read in a path."""
        key = dataset_cache_key(name="GBCU")

        assert len(key) == 12
        assert all(c in "0123456789abcdef" for c in key)

    @pytest.mark.parametrize(
        "changed",
        [
            {"name": "OpenKidney"},
            {"size": 512},
            {"revision": "0.5"},
            {"package_version": "9.9.9"},
            {"dataset_revision": "9.9"},
        ],
    )
    def test_changes_when_an_input_changes(self, changed):
        """Every input the build depends on is part of the key."""
        base = {
            "name": "GBCU",
            "size": 256,
            "revision": "0.4",
            "package_version": "2.0.0",
            "dataset_revision": "0.4",
        }

        assert dataset_cache_key(**base) != dataset_cache_key(**{**base, **changed})

    def test_changes_with_the_raw_data_path(self, tmp_path):
        """Processing different raw data gives a different key."""
        one = dataset_cache_key(name="GBCU", path=tmp_path / "one")
        two = dataset_cache_key(name="GBCU", path=tmp_path / "two")

        assert one != two

    def test_resolves_the_raw_data_path(self, tmp_path):
        """Two spellings of one directory share a key."""
        direct = dataset_cache_key(name="GBCU", path=tmp_path)
        indirect = dataset_cache_key(name="GBCU", path=tmp_path / "sub" / "..")

        assert direct == indirect


class TestDatasetCachePath:
    """Tests for dataset_cache_path and is_cached."""

    def test_nests_the_key_under_the_dataset(self, tmp_path):
        """Each dataset holds one directory per set of build inputs."""
        assert dataset_cache_path("GBCU", "abc", tmp_path) == tmp_path / "GBCU" / "abc"

    def test_an_empty_directory_is_not_a_cache_hit(self, tmp_path):
        """A directory without the dataset marker is incomplete."""
        assert not is_cached(tmp_path)

    def test_the_dataset_marker_is_a_cache_hit(self, tmp_path):
        """The file save_to_disk writes last marks a complete entry."""
        (tmp_path / "dataset_dict.json").write_text("{}")

        assert is_cached(tmp_path)


class TestValidateSubset:
    """Tests for validate_subset."""

    def test_accepts_a_valid_subset(self):
        """A subset the dataset publishes is accepted."""
        validate_subset("CAMUS", "segmentation", ("classification", "segmentation"))

    def test_accepts_no_subset_for_a_single_subset_dataset(self):
        """A dataset with one subset needs no subset argument."""
        validate_subset("AUL", None, ())

    def test_requires_a_subset_and_lists_them(self):
        """Omitting a required subset lists the valid ones."""
        with pytest.raises(ValueError, match="classification, segmentation"):
            validate_subset("CAMUS", None, ("classification", "segmentation"))

    def test_rejects_an_unknown_subset(self):
        """An unknown subset is reported with the valid ones."""
        with pytest.raises(ValueError, match="Unknown subset 'nope' for CAMUS"):
            validate_subset("CAMUS", "nope", ("classification", "segmentation"))

    def test_rejects_a_subset_for_a_single_subset_dataset(self):
        """Passing a subset where there are none is an error."""
        with pytest.raises(ValueError, match="AUL has no subsets"):
            validate_subset("AUL", "nope", ())


class TestValidateSplit:
    """Tests for validate_split."""

    def test_accepts_a_published_split(self):
        """A split the dataset publishes is accepted."""
        validate_split("AUL", "train", ("train", "test"))

    def test_accepts_no_split(self):
        """Asking for every split is always valid."""
        validate_split("AUL", None, ("train", "test"))

    def test_rejects_an_unknown_split_and_lists_them(self):
        """An unknown split is reported with the valid ones."""
        with pytest.raises(ValueError, match=r"Unknown split 'test.512'.*train, test"):
            validate_split("AUL", "test.512", ("train", "test"))


class TestValidateRawPath:
    """Tests for validate_raw_path."""

    def test_returns_an_existing_directory(self, tmp_path):
        """A directory holding raw data is resolved and returned."""
        assert validate_raw_path("GBCU", tmp_path) == tmp_path

    def test_requires_a_path(self):
        """A local dataset with no path explains why one is needed."""
        with pytest.raises(ValueError, match="GBCU cannot be republished"):
            validate_raw_path("GBCU", None)

    def test_rejects_a_missing_directory(self, tmp_path):
        """A path that does not exist is reported."""
        with pytest.raises(FileNotFoundError, match="No raw data for GBCU"):
            validate_raw_path("GBCU", tmp_path / "missing")

    def test_rejects_a_file(self, tmp_path):
        """A file where a directory is expected is reported."""
        target = tmp_path / "file.txt"
        target.write_text("")

        with pytest.raises(FileNotFoundError, match="must be a directory"):
            validate_raw_path("GBCU", target)


class TestWarnings:
    """Tests for the warnings about ignored arguments."""

    def test_warns_that_a_hosted_dataset_ignores_path(self):
        """A path passed for a hosted dataset is reported as ignored."""
        with pytest.warns(UserWarning, match="AUL is hosted.*is ignored"):
            warn_ignored_path("AUL", "/raw")

    def test_warns_that_a_local_dataset_ignores_revision(self):
        """A revision passed for a local dataset is reported as ignored."""
        with pytest.warns(UserWarning, match="GBCU is processed from local"):
            warn_ignored_revision("GBCU", "0.4")


class TestSelectSplit:
    """Tests for select_split."""

    @pytest.fixture
    def data(self):
        """Build a small two-split dataset.

        Returns:
            A DatasetDict with a train and a test split.
        """
        from datasets import Dataset, DatasetDict

        return DatasetDict(
            {
                "train": Dataset.from_dict({"a": [1, 2]}),
                "test": Dataset.from_dict({"a": [3]}),
            }
        )

    def test_returns_everything_when_no_split_is_named(self, data):
        """Omitting the split returns the whole dataset."""
        assert select_split(data, "GBCU", None) is data

    def test_returns_the_named_split(self, data):
        """Naming a split returns just that split."""
        assert len(select_split(data, "GBCU", "test")) == 1

    def test_rejects_an_unknown_split(self, data):
        """An unknown split lists the ones the dataset holds."""
        with pytest.raises(ValueError, match=r"Unknown split 'val'.*test, train"):
            select_split(data, "GBCU", "val")
