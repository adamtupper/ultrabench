"""Shared helpers for locating, partitioning and caching datasets."""

import csv
import hashlib
import os
import shutil
import uuid
import warnings
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from datasets import Dataset, DatasetDict, load_from_disk

from .registry import IMAGE_SIZE

if TYPE_CHECKING:
    from ultrabench.process.utils import BuildResult

__version__ = version("ultrabench")

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
"""The directory holding the tracked split manifests."""

CACHE_ENV_VAR = "ULTRABENCH_CACHE"
"""The environment variable naming the cache directory."""

DEFAULT_CACHE_DIR = Path.home() / ".ultrabench"
"""Where processed datasets are cached when nothing else is set."""

KEY_LENGTH = 12
"""The number of hex characters a cache key is truncated to."""

T = TypeVar("T")


def load_split_manifest(name: str, split: str) -> set[str]:
    """Read the names of the examples belonging to one split.

    The raw and local datasets are preprocessed on the user's machine,
    so their splits are read from a tracked manifest rather than
    re-derived from a seeded splitter, which would make the result
    depend on the scikit-learn version and on the raw data being
    byte-identical.

    Args:
        name: The registry name of the dataset.
        split: The split to read, either "train" or "test".

    Returns:
        The names of the examples in the split.

    Raises:
        FileNotFoundError: If the dataset has no manifest for the split.
    """
    path = MANIFEST_DIR / name / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No {split} manifest for {name}: {path}")

    with path.open(newline="") as f:
        return {row["filename"] for row in csv.DictReader(f)}


def partition_by_manifest(
    items: Sequence[T], name: str, key: Callable[[T], str]
) -> tuple[list[T], list[T]]:
    """Split a sequence into its train and test parts by manifest.

    An item in neither manifest is dropped, which is how subsampling and
    de-duplication are expressed. A manifest entry with no matching item
    is an error: it means the raw data no longer holds the examples the
    published splits were built from.

    Args:
        items: The items to partition.
        name: The registry name of the dataset.
        key: Returns the manifest name identifying an item. Datasets
            listed at video level match on the video stem rather than
            the record's filename.

    Returns:
        The train items and the test items, each in the order given.

    Raises:
        ValueError: If a manifest entry has no matching item.
    """
    train_names = load_split_manifest(name, "train")
    test_names = load_split_manifest(name, "test")

    train = [item for item in items if key(item) in train_names]
    test = [item for item in items if key(item) in test_names]

    found = {key(item) for item in items}
    missing = sorted((train_names | test_names) - found)
    if missing:
        raise ValueError(
            f"The raw data for {name} is missing {len(missing)} example(s) "
            f"listed in its split manifests: {', '.join(missing[:10])}"
            f"{', ...' if len(missing) > 10 else ''}"
        )

    return train, test


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the directory processed datasets are cached in.

    Falls back to the ULTRABENCH_CACHE environment variable and then to
    `~/.ultrabench`. This governs the UltraBench cache only; when no
    cache directory is given, Hub downloads are left to HF_HOME.

    Args:
        cache_dir: The directory to cache processed datasets in, or
            None to fall back to the environment.

    Returns:
        The resolved directory, which is not created here.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    from_env = os.environ.get(CACHE_ENV_VAR)

    return Path(from_env).expanduser() if from_env else DEFAULT_CACHE_DIR


def hub_cache_dir(cache_dir: str | Path | None = None) -> Path | None:
    """Resolve where Hub downloads belonging to a cache directory go.

    Args:
        cache_dir: The cache directory the caller passed, or None.

    Returns:
        A subdirectory of the resolved cache when a cache directory is
        set explicitly or in the environment, so that the directory is
        self-contained; otherwise None, leaving HF_HOME in charge.
    """
    if cache_dir is None and CACHE_ENV_VAR not in os.environ:
        return None

    return resolve_cache_dir(cache_dir) / "_hub"


def dataset_cache_key(
    *,
    name: str,
    size: int = IMAGE_SIZE,
    path: Path | None = None,
    revision: str | None = None,
    package_version: str = __version__,
    dataset_revision: str | None = None,
) -> str:
    """Fingerprint the inputs that determine a processed dataset.

    A change to any input yields a different key, so a stale entry is
    never read; it is simply left behind next to the new one.

    Args:
        name: The registry name of the dataset.
        size: The height and width of the resized test images.
        path: The raw data directory for a local dataset, or None.
        revision: The Hub revision for a raw dataset, or None.
        package_version: The version of the ultrabench package.
        dataset_revision: The revision the processing code produces, or
            None to read it from the processing layer.

    Returns:
        The first 12 hex characters of the SHA-256 of the inputs.
    """
    if dataset_revision is None:
        from ultrabench.process.utils import DATASET_REVISION

        dataset_revision = DATASET_REVISION

    payload = "|".join(
        [
            name,
            package_version,
            dataset_revision,
            str(size),
            str(path.resolve()) if path is not None else "",
            revision or "",
        ]
    )

    return hashlib.sha256(payload.encode()).hexdigest()[:KEY_LENGTH]


def dataset_cache_path(
    name: str, key: str, cache_dir: str | Path | None = None
) -> Path:
    """Build the directory a processed dataset is cached in.

    Args:
        name: The registry name of the dataset.
        key: The cache key fingerprinting the build inputs.
        cache_dir: The cache directory, or None to resolve it.

    Returns:
        The path to the dataset's cache entry.
    """
    return resolve_cache_dir(cache_dir) / name / key


def is_cached(cache_path: Path) -> bool:
    """Report whether a complete dataset has been written to a path.

    Args:
        cache_path: The cache entry to check.

    Returns:
        Whether the entry holds a complete dataset.
    """
    return (cache_path / "dataset_dict.json").is_file()


def load_cached(cache_path: Path) -> DatasetDict:
    """Read a cached dataset from disk.

    `datasets.load_dataset` silently misreads a `save_to_disk`
    directory rather than failing, so the cache is only ever read with
    `load_from_disk`.

    Args:
        cache_path: The cache entry to read.

    Returns:
        The cached dataset.

    Raises:
        TypeError: If the entry holds a single split rather than a
            whole dataset.
    """
    data = load_from_disk(str(cache_path))
    if not isinstance(data, DatasetDict):
        raise TypeError(
            f"The cache entry at {cache_path} holds a single split rather "
            f"than a whole dataset."
        )

    return data


def save_to_cache(
    result: "BuildResult",
    cache_path: Path,
    *,
    size: int = IMAGE_SIZE,
    ground_truth_writer: Callable[[list[dict], Path, int | None], None],
) -> DatasetDict:
    """Write a freshly built dataset into the cache and read it back.

    The dataset is built into a sibling temporary directory and renamed
    into place, so an interrupted build never leaves an entry that
    `is_cached` would accept.

    Args:
        result: The dataset and the records its ground truth comes from.
        cache_path: The cache entry to write.
        size: The height and width of the resized test images.
        ground_truth_writer: Writes one test split's ground-truth CSVs,
            given its records, its directory and the size its masks are
            encoded at.

    Returns:
        The cached dataset, read back so that the first call and every
        later one return the same thing.
    """
    from ultrabench.process.utils import save_version_info, write_test_ground_truth

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    staging = cache_path.parent / f".tmp-{uuid.uuid4().hex}"

    try:
        result.dataset.save_to_disk(str(staging))
        write_test_ground_truth(result, staging, size, ground_truth_writer)
        save_version_info(str(staging), __version__)
        staging.replace(cache_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return load_cached(cache_path)


def validate_subset(name: str, subset: str | None, subsets: Sequence[str]) -> None:
    """Check that a dataset's subset was given and is valid.

    Args:
        name: The registry name of the dataset.
        subset: The subset the caller asked for, or None.
        subsets: The subsets the dataset publishes.

    Raises:
        ValueError: If the subset is missing, unknown, or given for a
            dataset that has only one.
    """
    if not subsets:
        if subset is not None:
            raise ValueError(
                f"{name} has no subsets, so the subset argument must be "
                f"omitted, but {subset!r} was given."
            )
        return

    valid = ", ".join(subsets)
    if subset is None:
        raise ValueError(
            f"{name} has more than one subset, so one must be chosen. "
            f"Valid subsets: {valid}."
        )
    if subset not in subsets:
        raise ValueError(
            f"Unknown subset {subset!r} for {name}. Valid subsets: {valid}."
        )


def validate_split(name: str, split: str | None, splits: Sequence[str]) -> None:
    """Check that a named split is one the dataset publishes.

    Args:
        name: The registry name of the dataset.
        split: The split the caller asked for, or None for all of them.
        splits: The splits the dataset publishes.

    Raises:
        ValueError: If the split is not one the dataset publishes.
    """
    if split is not None and split not in splits:
        raise ValueError(
            f"Unknown split {split!r} for {name}. Valid splits: {', '.join(splits)}."
        )


def validate_raw_path(name: str, path: str | Path | None) -> Path:
    """Resolve the raw data path a local dataset was given.

    Args:
        name: The registry name of the dataset.
        path: The directory the raw data was downloaded to, or None.

    Returns:
        The resolved directory.

    Raises:
        ValueError: If no path was given.
        FileNotFoundError: If the path is not an existing directory.
    """
    if path is None:
        raise ValueError(
            f"{name} cannot be republished, so it is not available on the "
            f"Hugging Face Hub. Download it yourself and pass the directory "
            f"as path. See docs/dataset_cards/ for where to download it."
        )

    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"No raw data for {name} at {resolved}.")
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"The raw data path for {name} must be a directory, but "
            f"{resolved} is a file."
        )

    return resolved


def warn_ignored_path(name: str, path: str | Path) -> None:
    """Warn that a hosted dataset ignores the path it was given.

    Args:
        name: The registry name of the dataset.
        path: The path the caller passed.
    """
    warnings.warn(
        f"{name} is hosted on the Hugging Face Hub, so path={str(path)!r} is ignored.",
        UserWarning,
        stacklevel=3,
    )


def warn_ignored_revision(name: str, revision: str) -> None:
    """Warn that a local dataset ignores the revision it was given.

    Args:
        name: The registry name of the dataset.
        revision: The revision the caller passed.
    """
    warnings.warn(
        f"{name} is processed from local raw data rather than downloaded, so "
        f"revision={revision!r} is ignored.",
        UserWarning,
        stacklevel=3,
    )


def select_split(
    data: DatasetDict, name: str, split: str | None
) -> Dataset | DatasetDict:
    """Return a whole dataset, or one split of it.

    Args:
        data: The dataset to select from.
        name: The registry name of the dataset.
        split: The split to return, or None for all of them.

    Returns:
        The dataset, or the requested split of it.

    Raises:
        ValueError: If the split is not one the dataset holds.
    """
    if split is None:
        return data

    validate_split(name, split, sorted(str(key) for key in data))

    return data[split]
