"""Unit tests for the dataset registry."""

import dataclasses
from typing import get_args

import pytest
import yaml

from ultrabench2.datasets.registry import (
    DEFAULT_SPLITS,
    _REGISTRY,
    DatasetConfig,
    DatasetName,
    DatasetType,
)
from ultrabench2.evaluate.tasks import TASK_CONFIG_DIR

EXPECTED_HUB_IDS = {
    "AUL": "UltraBench/AUL",
    "BUS-BRA": "UltraBench/BUS-BRA",
    "CAMUS": "UltraBench/CAMUS",
    "FetalPlanes": "UltraBench/FetalPlanes",
    "MASLD": "UltraBench/MASLD",
    "PSFHS": "UltraBench/PSFHS",
    "STMUS": "UltraBench/STMUS",
    "TN3K": "UltraBench/TN3K",
    "COVID-BLUES": "UltraBench/COVID-BLUES-raw",
    "MMOTU": "UltraBench/MMOTU-raw",
    "GBCU": None,
    "OpenKidney": None,
}
"""The repository each dataset is loaded from, pinned against drift."""

EXPECTED_SUBSETS = {
    "CAMUS": ("classification", "segmentation"),
    "FetalPlanes": ("fetal", "fetal_brain"),
}
"""The only two datasets that publish more than one subset."""


class TestRegistry:
    """Tests for the contents of the registry."""

    def test_covers_every_declared_name(self):
        """DatasetName and the registry list the same datasets."""
        assert set(get_args(DatasetName)) == set(_REGISTRY)

    @pytest.mark.parametrize(("name", "hub_id"), sorted(EXPECTED_HUB_IDS.items()))
    def test_hub_id(self, name, hub_id):
        """Each dataset points at the repository it is loaded from."""
        assert _REGISTRY[name].hub_id == hub_id

    def test_mmotu_loads_the_raw_repository(self):
        """MMOTU's raw repo differs from its processed one."""
        assert _REGISTRY["MMOTU"].hub_id != "UltraBench/MMOTU-2D"

    @pytest.mark.parametrize("name", sorted(_REGISTRY))
    def test_type_invariants(self, name):
        """Each type carries exactly the fields it needs."""
        config = _REGISTRY[name]

        if config.type is DatasetType.PROCESSED:
            assert config.hub_id is not None
            assert config.loader is None
        elif config.type is DatasetType.RAW:
            assert config.hub_id is not None
            assert config.loader is not None
        else:
            assert config.hub_id is None
            assert config.loader is not None

    @pytest.mark.parametrize("name", sorted(_REGISTRY))
    def test_subsets(self, name):
        """Only CAMUS and FetalPlanes publish more than one subset."""
        assert _REGISTRY[name].subsets == EXPECTED_SUBSETS.get(name, ())

    @pytest.mark.parametrize("name", sorted(_REGISTRY))
    def test_splits(self, name):
        """Every dataset publishes the same three splits."""
        assert _REGISTRY[name].splits == DEFAULT_SPLITS

    @pytest.mark.parametrize("name", sorted(_REGISTRY))
    def test_loader_is_importable(self, name):
        """Every named loader resolves to a callable."""
        config = _REGISTRY[name]
        if config.loader is None:
            pytest.skip("processed datasets have no loader")

        from ultrabench2.datasets.load_dataset import _resolve_loader

        assert callable(_resolve_loader(config))

    def test_config_is_frozen(self):
        """A registry entry cannot be mutated in place."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            _REGISTRY["AUL"].hub_id = "other"

    def test_config_defaults(self):
        """A minimal entry gets sensible defaults."""
        config = DatasetConfig(type=DatasetType.LOCAL)

        assert (config.hub_id, config.loader, config.subsets) == (None, None, ())


class TestTaskConfigsMatchRegistry:
    """Tests that the training and dataset registries cannot drift."""

    @pytest.fixture(scope="class")
    def configs(self):
        """Load every task config.

        Returns:
            A mapping from the task name to its parsed config.
        """
        return {
            path.stem: yaml.safe_load(path.read_text())
            for path in sorted(TASK_CONFIG_DIR.glob("*.yaml"))
        }

    def test_every_task_names_a_registered_dataset(self, configs):
        """Each task's name key resolves in the registry."""
        unknown = {
            task: config.get("name")
            for task, config in configs.items()
            if config.get("name") not in _REGISTRY
        }

        assert not unknown

    def test_processed_tasks_agree_on_the_repository(self, configs):
        """A processed task's dataset key matches the registry."""
        mismatched = {
            task: (config["dataset"], _REGISTRY[config["name"]].hub_id)
            for task, config in configs.items()
            if _REGISTRY[config["name"]].type is DatasetType.PROCESSED
            and config["dataset"] != _REGISTRY[config["name"]].hub_id
        }

        assert not mismatched

    def test_local_tasks_supply_a_path(self, configs):
        """A local dataset's task config names its raw data."""
        missing = [
            task
            for task, config in configs.items()
            if _REGISTRY[config["name"]].type is DatasetType.LOCAL
            and not config.get("path")
        ]

        assert not missing
