"""Download the POCUS dataset from the covid19_ultrasound GitHub repository.

The dataset is part of the repository at:
https://github.com/jannisborn/covid19_ultrasound

The full repository is downloaded as a ZIP archive and extracted. The relevant
data lives in the `data/` subdirectory of the extracted archive, which is what
should be passed as RAW_DATA_DIR to `ultrabench pocus process`.

Usage:
    ultrabench pocus dl DOWNLOAD_DIR
"""

import os
import shutil
from typing import Annotated

import typer

from .utils import download_file, extract_archive

REPO_ZIP_URL = (
    "https://github.com/jannisborn/covid19_ultrasound/archive/refs/heads/master.zip"
)
ARCHIVE_FILENAME = "covid19_ultrasound.zip"
EXTRACTED_DIRNAME = "covid19_ultrasound-master"


def download_pocus(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ],
) -> None:
    """Download the POCUS dataset from GitHub (covid19_ultrasound repository).

    The repository is downloaded as a ZIP archive and extracted. Pass the
    resulting directory to `ultrabench pocus process` as the RAW_DATA_DIR.

    Args:
        download_dir: The directory to download the dataset into.
    """
    os.makedirs(download_dir, exist_ok=True)
    typer.echo("Downloading POCUS dataset from GitHub...")

    dest = os.path.join(download_dir, ARCHIVE_FILENAME)
    download_file(REPO_ZIP_URL, dest)
    extract_archive(dest, download_dir)
    os.remove(dest)

    # Rename the extracted folder to a cleaner name
    extracted_path = os.path.join(download_dir, EXTRACTED_DIRNAME)
    final_path = os.path.join(download_dir, "covid19_ultrasound")
    if os.path.exists(extracted_path):
        shutil.move(extracted_path, final_path)

    typer.echo(f"Download complete. Data saved to: {final_path}")
    typer.echo(f"Pass '{final_path}' as RAW_DATA_DIR to `ultrabench pocus process`.")
