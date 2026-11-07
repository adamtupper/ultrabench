"""Download the MASLD dataset from Zenodo.

The dataset is available at https://zenodo.org/records/1009146.

Usage:
    uv run python scripts/download_dataset.py masld DOWNLOAD_DIR
"""

import os
from datetime import date

from zenodo_get import download as zenodo_download

from .utils import print_next_step


def download_masld(
    download_dir: str = ".",
) -> None:
    """Download the MASLD dataset from Zenodo.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"masld_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading MASLD dataset from Zenodo...")
    zenodo_download("1009146", output_dir=output_dir)
    print_next_step(
        "masld",
        os.path.join(
            output_dir,
            "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat",
        ),
    )
