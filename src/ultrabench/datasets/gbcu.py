"""Load the GBCU dataset from a local copy of the raw data.

GBCU cannot be republished, so download it from
https://gbc-iitd.github.io/data/gbcu and pass the directory as `path`.
"""

from pathlib import Path

from datasets import DatasetDict

from .utils import (
    IMAGE_SIZE,
    dataset_cache_key,
    dataset_cache_path,
    is_cached,
    load_cached,
    save_to_cache,
    validate_raw_path,
)

NAME = "GBCU"
"""The registry name of the dataset."""


def load_gbcu(path: str | Path, cache_dir: str | Path | None = None) -> DatasetDict:
    """Load the GBCU dataset, preprocessing it on first use.

    The processed dataset is cached, so only the first call for a given
    raw data directory and package version pays for preprocessing.

    Args:
        path: The directory the raw GBCU data was downloaded to.
        cache_dir: The directory to cache the processed dataset in.
            Defaults to $ULTRABENCH_CACHE, then `~/.ultrabench`.

    Returns:
        The train, test and test.256 splits.

    Raises:
        ValueError: If no path was given.
        FileNotFoundError: If the path is not an existing directory.
    """
    from ultrabench.process.gbcu import build_gbcu_dataset, write_ground_truth

    raw_path = validate_raw_path(NAME, path)
    key = dataset_cache_key(name=NAME, path=raw_path)
    cache_path = dataset_cache_path(NAME, key, cache_dir)

    if is_cached(cache_path):
        return load_cached(cache_path)

    print(f"Processing {NAME} from {raw_path}. This happens once.")
    result = build_gbcu_dataset(raw_path, size=IMAGE_SIZE)

    return save_to_cache(
        result, cache_path, size=IMAGE_SIZE, ground_truth_writer=write_ground_truth
    )
