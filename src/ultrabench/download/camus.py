"""Download the CAMUS dataset from the CREATIS database.

The dataset is available at:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8

Note: Access to the CAMUS dataset may require prior registration at
https://humanheart-project.creatis.insa-lyon.fr/. If the download fails
with an authentication error, please register and try again.

Usage:
    ultrabench camus dl DOWNLOAD_DIR
"""

import os
from datetime import date
from typing import Annotated

import typer

from .utils import download_file, extract_archive

DOWNLOAD_URL = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/6373703d73e9f0047faa1bc8/download"
ARCHIVE_FILENAME = "camus.zip"


def download_camus(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ],
) -> None:
    """Download the CAMUS dataset from the CREATIS database.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"camus_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    typer.echo("Downloading CAMUS dataset from the CREATIS database...")

    dest = os.path.join(output_dir, ARCHIVE_FILENAME)
    download_file(DOWNLOAD_URL, dest)
    extract_archive(dest, output_dir)
    os.remove(dest)

    typer.echo(f"Download complete. Saved to {output_dir}.")
    typer.echo(
        f"Process the dataset by running `ultrabench process camus {output_dir}`."
    )
