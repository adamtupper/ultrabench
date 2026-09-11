"""Generate the tracked split manifests from a processed dataset.

The raw and local datasets are preprocessed on the user's machine, so
their splits cannot be derived from a seeded splitter without making the
result depend on the scikit-learn version and on the raw data being
byte-identical. Instead the split membership is committed to the package
and the build cores partition by it.

This script writes those manifests from an already published dataset, so
that the committed splits are provably the published ones. Run it once to
bootstrap the manifests, and again whenever a new revision intentionally
re-splits a dataset.

Usage:
    uv run python scripts/generate_split_manifests.py PROCESSED_DIR
"""

import argparse
import csv
from pathlib import Path

from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "src" / "ultrabench2" / "datasets" / "manifests"
SPLITS = ("train", "test")

DATASETS = {
    "GBCU": "GBCU",
    "OpenKidney": "OpenKidney",
    "MMOTU-2D": "MMOTU",
    "COVID-BLUES": "COVID-BLUES",
}
"""A mapping from the published repository name to the registry name."""

VIDEO_DATASETS = frozenset({"COVID-BLUES"})
"""Datasets listed one video per row rather than one image per row."""


def split_names(dataset, name: str) -> list[str]:
    """Collect the names identifying the examples in one split.

    A video dataset is listed at video level: every frame of a video
    belongs to the same split, so the video stems determine the same
    split as the frame filenames would, at a fraction of the size.

    Args:
        dataset: The split to read.
        name: The registry name of the dataset the split belongs to.

    Returns:
        The sorted, unique names of the examples in the split.

    Raises:
        ValueError: If a video stem does not match its frame filenames.
    """
    if name not in VIDEO_DATASETS:
        return sorted(set(dataset["filename"]))

    for filename, video in zip(dataset["filename"], dataset["video"]):
        if filename.rsplit("_", 1)[0] != video:
            raise ValueError(
                f"Frame {filename!r} does not belong to video {video!r}. The "
                "manifest cannot be written at video level."
            )

    return sorted(set(dataset["video"]))


def write_manifest(names: list[str], path: Path) -> None:
    """Write one split manifest.

    Args:
        names: The names of the examples in the split.
        path: The file to write the manifest to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        writer.writerows([name] for name in names)


def main() -> None:
    """Write the split manifests for every raw and local dataset.

    Raises:
        FileNotFoundError: If a published dataset is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "processed_dir",
        type=Path,
        help="The directory holding the published datasets.",
    )
    args = parser.parse_args()

    for repo, name in DATASETS.items():
        source = args.processed_dir / repo
        if not source.is_dir():
            raise FileNotFoundError(f"Published dataset not found: {source}")

        dataset = load_from_disk(str(source))
        for split in SPLITS:
            names = split_names(dataset[split], name)
            path = MANIFEST_DIR / name / f"{split}.csv"
            write_manifest(names, path)
            print(f"{name}/{split}.csv: {len(names)} rows")


if __name__ == "__main__":
    main()
