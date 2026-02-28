"""Download the Annotated Ultrasound Liver (AUL) dataset from Zenodo.

The dataset is available at https://zenodo.org/records/7272660.

Usage:
    ultrabench aul dl DOWNLOAD_DIR
"""

import os
from typing import Annotated

import typer
import zenodo_get


def download_aul(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ],
) -> None:
    """Download the AUL dataset from Zenodo (Record 7272660).

    Args:
        download_dir: The directory to download the dataset into.
    """
    os.makedirs(download_dir, exist_ok=True)
    typer.echo("Downloading AUL dataset from Zenodo (Record 7272660)...")
    zenodo_get.download("7272660", output_dir=download_dir)
    typer.echo(f"Download complete. Data saved to: {download_dir}")
