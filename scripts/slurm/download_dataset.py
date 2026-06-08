"""Download a dataset from the Hugging Face Hub.

Usage:
    python download_dataset.py
        --name <dataset_name>
        --cache_dir <cache_dir>
"""

import argparse

import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a dataset from the Hugging Face Hub."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the dataset to download.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="The configuration of the dataset to download. Defaults to None.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory to cache the downloaded dataset. Defaults "
        "to the default Hugging Face cache directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _ = datasets.load_dataset(args.name, name=args.config, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
