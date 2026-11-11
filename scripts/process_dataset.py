"""Process a raw dataset into the UltraBench format.

The processed dataset is saved under OUTPUT_DIR/UltraBench/<Dataset>,
together with the ground-truth CSVs of its test splits. With --publish
it is also pushed to the Hugging Face Hub, which requires write access
to the UltraBench organisation.

Usage:
    uv run python scripts/process_dataset.py DATASET RAW_DIR OUTPUT_DIR \\
        [--size 256] [--publish]
"""

import argparse
import warnings
from importlib import import_module

from processing import PUBLISHED_PROCESSORS
from processing.local import LOCAL_DATASETS, process_local

from ultrabench.datasets.registry import IMAGE_SIZE

# Filter scikit-image deprecation warnings. Must be migrated before 0.28.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*(`min_size`|`area_threshold`|`binary_erosion`|`binary_dilation`).*",
)


def main() -> None:
    """Parse the command-line arguments and process the dataset."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "dataset", choices=sorted([*PUBLISHED_PROCESSORS, *LOCAL_DATASETS])
    )
    parser.add_argument("raw_dir", help="The raw data directory (or file, for MASLD).")
    parser.add_argument(
        "output_dir", help="The directory to save the processed dataset under."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=IMAGE_SIZE,
        help="The height and width of the resized test images in pixels.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the dataset to the Hugging Face Hub.",
    )
    args = parser.parse_args()

    if args.dataset in LOCAL_DATASETS:
        process_local(
            args.dataset, args.raw_dir, args.output_dir, args.size, args.publish
        )
        return

    module_name, function_name = PUBLISHED_PROCESSORS[args.dataset]
    module = import_module(f"processing.{module_name}")
    process = getattr(module, function_name)
    process(args.raw_dir, args.output_dir, args.size, args.publish)


if __name__ == "__main__":
    main()
