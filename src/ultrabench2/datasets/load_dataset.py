"""The single entry point for loading any UltraBench dataset."""

from importlib import import_module
from pathlib import Path
from typing import Callable

import datasets as hf_datasets
from datasets import Dataset, DatasetDict

from .registry import _REGISTRY, DatasetConfig, DatasetName, DatasetType
from .utils import (
    hub_cache_dir,
    select_split,
    validate_raw_path,
    validate_split,
    validate_subset,
    warn_ignored_path,
    warn_ignored_revision,
)


def _resolve_loader(config: DatasetConfig) -> Callable[..., DatasetDict]:
    """Import the loader a dataset's configuration names.

    Loaders are named rather than imported by the registry so that
    importing the package does not pull in the preprocessing
    dependencies.

    Args:
        config: The configuration of the dataset to load.

    Returns:
        The loader function.
    """
    assert config.loader is not None, "only raw and local datasets have a loader"
    module_name, _, function_name = config.loader.partition(":")

    return getattr(import_module(module_name), function_name)


def load_dataset(
    name: DatasetName | str,
    path: str | Path | None = None,
    subset: str | None = None,
    split: str | None = None,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Dataset | DatasetDict:
    """Load an UltraBench dataset, whatever form it is distributed in.

    A processed dataset is downloaded ready to use. A raw dataset has
    its original files downloaded and is preprocessed on first use. A
    local dataset cannot be republished, so you download it yourself
    and pass the directory as `path`. Raw and local datasets are cached
    after preprocessing, so only the first call pays for it.

    Args:
        name: The dataset to load.
        path: The directory the raw data was downloaded to. Required
            for local datasets and ignored for the others.
        subset: The subset to load, for a dataset that has more than
            one.
        split: The split to load, or None for all of them.
        cache_dir: The directory to cache processed datasets in.
            Defaults to $ULTRABENCH_CACHE, then `~/.ultrabench`.
        revision: The Hub revision to load. Ignored for local datasets.

    Returns:
        The whole dataset, or the requested split of it.

    Raises:
        ValueError: If the name, subset or split is not valid, or if a
            local dataset was given no path.
        FileNotFoundError: If a local dataset's path does not exist.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. Valid names: {', '.join(sorted(_REGISTRY))}."
        )

    config = _REGISTRY[name]

    validate_subset(name, subset, config.subsets)
    validate_split(name, split, config.splits)

    if config.type is DatasetType.LOCAL:
        raw_path = validate_raw_path(name, path)
        if revision is not None:
            warn_ignored_revision(name, revision)

        loader = _resolve_loader(config)

        return select_split(loader(path=raw_path, cache_dir=cache_dir), name, split)

    if path is not None:
        warn_ignored_path(name, path)

    if config.type is DatasetType.RAW:
        loader = _resolve_loader(config)
        data = loader(hub_id=config.hub_id, cache_dir=cache_dir, revision=revision)

        return select_split(data, name, split)

    assert config.hub_id is not None, "a processed dataset always has a hub_id"

    hub_cache = hub_cache_dir(cache_dir)
    hub_cache_str = str(hub_cache) if hub_cache is not None else None

    # `datasets.load_dataset` overloads on whether a split is named, so
    # each branch is written out to keep the return type precise.
    if split is None:
        return hf_datasets.load_dataset(
            config.hub_id,
            name=subset,
            revision=revision,
            cache_dir=hub_cache_str,
        )

    return hf_datasets.load_dataset(
        config.hub_id,
        name=subset,
        split=split,
        revision=revision,
        cache_dir=hub_cache_str,
    )
