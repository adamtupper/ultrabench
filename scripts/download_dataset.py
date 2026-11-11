"""Download a raw dataset for processing.

Only the datasets that can be downloaded programmatically are supported.
The others must be requested from their authors and downloaded by hand.

Usage:
    uv run python scripts/download_dataset.py DATASET [DOWNLOAD_DIR]
"""

import argparse

from downloading import DOWNLOADERS


def main() -> None:
    """Parse the command-line arguments and download the dataset."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("dataset", choices=sorted(DOWNLOADERS))
    parser.add_argument(
        "download_dir",
        nargs="?",
        default=".",
        help="The directory to download the dataset into.",
    )
    args = parser.parse_args()
    DOWNLOADERS[args.dataset](args.download_dir)


if __name__ == "__main__":
    main()
