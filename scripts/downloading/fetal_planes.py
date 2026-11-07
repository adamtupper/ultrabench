"""Download the Fetal Planes dataset from Zenodo.

The dataset is available at https://zenodo.org/records/3904280.

Usage:
    uv run python scripts/download_dataset.py fetal-planes DOWNLOAD_DIR
"""

import os
from datetime import date

from zenodo_get import download as zenodo_download

from .utils import extract_archive, print_next_step


def download_fetal_planes(
    download_dir: str = ".",
) -> None:
    """Download the Fetal Planes dataset from Zenodo.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"fetal_planes_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading Fetal Planes dataset from Zenodo...")
    zenodo_download("3904280", output_dir=output_dir)

    for filename in os.listdir(output_dir):
        if filename.endswith(".zip"):
            archive_path = os.path.join(output_dir, filename)
            extract_archive(archive_path, output_dir)
            os.remove(archive_path)

    print_next_step("fetal-planes", output_dir)
