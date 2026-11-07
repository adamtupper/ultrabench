"""Download the COVID-BLUES dataset from GitHub.

The dataset is available from:
https://github.com/NinaWie/COVID-BLUES

The full repository is downloaded as a ZIP archive and extracted. Pass
the resulting directory to `scripts/process_dataset.py covid-blues` as
RAW_DATA_DIR.

Usage:
    uv run python scripts/download_dataset.py covid-blues DOWNLOAD_DIR
"""

import os
import shutil
from datetime import date


from .utils import download_file, extract_archive, print_next_step

_REPO_ZIP_URL = "https://github.com/NinaWie/COVID-BLUES/archive/refs/heads/main.zip"
_ARCHIVE_FILENAME = "covid_blues.zip"
_EXTRACTED_DIRNAME = "COVID-BLUES-main"


def download_covid_blues(
    download_dir: str = ".",
) -> None:
    """Download the COVID-BLUES dataset from GitHub.

    Downloads the full repository as a ZIP archive and extracts it.
    Pass the resulting directory to `scripts/process_dataset.py covid-blues`
    as the RAW_DATA_DIR.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"covid_blues_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading COVID-BLUES dataset from GitHub...")
    zip_path = os.path.join(output_dir, _ARCHIVE_FILENAME)
    download_file(_REPO_ZIP_URL, zip_path)

    print("Extracting COVID-BLUES dataset...")
    extract_archive(zip_path, output_dir)
    os.remove(zip_path)

    extracted_path = os.path.join(output_dir, _EXTRACTED_DIRNAME)
    if os.path.exists(extracted_path):
        shutil.move(extracted_path, os.path.join(output_dir, "covid_blues"))

    print_next_step("covid-blues", output_dir)
