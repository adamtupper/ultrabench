"""Load the COVID-BLUES dataset from its original files on the Hub.

COVID-BLUES is republished unmodified as `UltraBench/COVID-BLUES-raw`
and preprocessed on the user's machine.
"""

from pathlib import Path

from datasets import DatasetDict

from .utils import (
    IMAGE_SIZE,
    dataset_cache_key,
    dataset_cache_path,
    hub_cache_dir,
    is_cached,
    load_cached,
    save_to_cache,
)

NAME = "COVID-BLUES"
"""The registry name of the dataset."""


def load_covid_blues(
    hub_id: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> DatasetDict:
    """Load the COVID-BLUES dataset, preprocessing it on first use.

    The processed dataset is cached, so only the first call for a given
    revision and package version downloads or preprocesses anything.

    Args:
        hub_id: The repository holding the original files.
        cache_dir: The directory to cache the processed dataset in.
            Defaults to $ULTRABENCH_CACHE, then `~/.ultrabench`.
        revision: The revision of the raw repository to download.

    Returns:
        The train, test and test.256 splits.
    """
    from huggingface_hub import snapshot_download

    from ultrabench.datasets.builders.covid_blues import (
        build_covid_blues_dataset,
        write_ground_truth,
    )

    key = dataset_cache_key(name=NAME, revision=revision)
    cache_path = dataset_cache_path(NAME, key, cache_dir)

    # The cache is checked before the download, so a warm cache needs
    # no network and works under HF_HUB_OFFLINE.
    if is_cached(cache_path):
        return load_cached(cache_path)

    hub_cache = hub_cache_dir(cache_dir)
    raw_dir = snapshot_download(
        hub_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(hub_cache) if hub_cache is not None else None,
    )

    print(f"Processing {NAME} from {raw_dir}. This happens once.")
    result = build_covid_blues_dataset(raw_dir, size=IMAGE_SIZE)

    return save_to_cache(
        result, cache_path, size=IMAGE_SIZE, ground_truth_writer=write_ground_truth
    )
