"""Unit tests for ultrabench2.publish.raw."""

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from ultrabench2.publish.raw import RAW_DATASETS, publish_raw


@pytest.fixture
def hub(monkeypatch):
    """Replace every Hub call with a recorder.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        A dict of the arguments create_repo and upload_folder saw.
    """
    calls = {}

    def create_repo(self, repo_id, **kwargs):
        calls["create_repo"] = {"repo_id": repo_id, **kwargs}

    def upload_folder(self, **kwargs):
        calls["upload_folder"] = kwargs

    def list_repo_refs(self, repo_id, repo_type=None, **kwargs):
        response = httpx.Response(
            404, request=httpx.Request("GET", f"https://hf.co/{repo_id}")
        )
        raise RepositoryNotFoundError(repo_id, response=response)

    monkeypatch.setattr("ultrabench2.publish.raw.HfApi.create_repo", create_repo)
    monkeypatch.setattr("ultrabench2.publish.raw.HfApi.upload_folder", upload_folder)
    monkeypatch.setattr(
        "ultrabench2.process.utils.HfApi.list_repo_refs", list_repo_refs
    )

    return calls


@pytest.fixture
def covid_blues_dir(tmp_path):
    """Build a directory holding the COVID-BLUES marker file.

    Args:
        tmp_path: The pytest temporary directory fixture.

    Returns:
        The raw data directory.
    """
    marker = tmp_path / "covid_blues" / "clinical_variables.csv"
    marker.parent.mkdir(parents=True)
    marker.write_text("patient_id,cov_test\n1,0\n")

    return tmp_path


class TestPublishRaw:
    """Tests for publish_raw."""

    def test_rejects_a_dataset_that_is_not_republished(self, tmp_path):
        """Only the datasets that may be redistributed are accepted."""
        with pytest.raises(ValueError, match="covid-blues, mmotu"):
            publish_raw("aul", str(tmp_path))

    def test_rejects_a_directory_without_the_marker(self, tmp_path, hub):
        """A directory holding the wrong dataset is reported."""
        with pytest.raises(FileNotFoundError, match="clinical_variables.csv"):
            publish_raw("covid-blues", str(tmp_path))

    def test_creates_a_private_repository(self, covid_blues_dir, hub):
        """The repository is private unless asked otherwise."""
        publish_raw("covid-blues", str(covid_blues_dir))

        assert hub["create_repo"]["repo_id"] == "UltraBench/COVID-BLUES-raw"
        assert hub["create_repo"]["repo_type"] == "dataset"
        assert hub["create_repo"]["private"] is True

    def test_can_create_a_public_repository(self, covid_blues_dir, hub):
        """Publishing publicly is opt-in."""
        publish_raw("covid-blues", str(covid_blues_dir), private=False)

        assert hub["create_repo"]["private"] is False

    def test_uploads_only_the_files_the_processor_reads(self, covid_blues_dir, hub):
        """The source code shipped alongside the data is left out."""
        publish_raw("covid-blues", str(covid_blues_dir))

        assert hub["upload_folder"]["allow_patterns"] == [
            "covid_blues/clinical_variables.csv",
            "covid_blues/lus_videos/*.mp4",
        ]

    def test_uploads_the_directory_unmodified(self, covid_blues_dir, hub):
        """The layout on the Hub is the layout the processor expects."""
        publish_raw("covid-blues", str(covid_blues_dir))

        assert hub["upload_folder"]["folder_path"] == str(covid_blues_dir)
        assert hub["upload_folder"]["repo_type"] == "dataset"

    def test_refuses_to_overwrite_a_published_revision(
        self, covid_blues_dir, hub, monkeypatch
    ):
        """A revision that is already tagged is never republished."""

        def list_repo_refs(self, repo_id, repo_type=None, **kwargs):
            refs = type("GitRefs", (), {})()
            refs.tags = [type("GitRefInfo", (), {"name": "0.4"})()]

            return refs

        monkeypatch.setattr(
            "ultrabench2.process.utils.HfApi.list_repo_refs", list_repo_refs
        )

        with pytest.raises(ValueError, match="already been published"):
            publish_raw("covid-blues", str(covid_blues_dir))

        assert "upload_folder" not in hub


class TestRawDatasets:
    """Tests for the raw dataset table."""

    def test_matches_the_registry(self):
        """Every raw dataset in the registry can be published."""
        from ultrabench2.datasets.registry import _REGISTRY, DatasetType

        registered = {
            config.hub_id
            for config in _REGISTRY.values()
            if config.type is DatasetType.RAW
        }

        assert {repo.repo_id for repo in RAW_DATASETS.values()} == registered
