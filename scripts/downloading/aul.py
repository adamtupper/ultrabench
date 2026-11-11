"""Download the Annotated Ultrasound Liver (AUL) dataset from Zenodo.

The dataset is available at https://zenodo.org/records/7272660.

Usage:
    uv run python scripts/download_dataset.py aul DOWNLOAD_DIR
"""

import os
from datetime import date

import zenodo_get

from .utils import extract_archive, print_next_step


def download_aul(
    download_dir: str = ".",
) -> None:
    """Download the AUL dataset from Zenodo.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"aul_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading AUL dataset from Zenodo...")
    zenodo_get.download("7272660", output_dir=output_dir)

    for filename in os.listdir(output_dir):
        if filename.endswith(".zip"):
            archive_path = os.path.join(output_dir, filename)
            extract_archive(archive_path, output_dir)
            os.remove(archive_path)

    print_next_step("aul", output_dir)
