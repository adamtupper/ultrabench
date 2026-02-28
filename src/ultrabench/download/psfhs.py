"""Download the PSFHS dataset from Zenodo.

The dataset is available at https://zenodo.org/records/10969427.

Usage:
    ultrabench psfhs dl DOWNLOAD_DIR
"""

import os
from typing import Annotated

import typer
from zenodo_get import download as zenodo_download


def download_psfhs(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ],
) -> None:
    """Download the PSFHS dataset from Zenodo (Record 10969427).

    Args:
        download_dir: The directory to download the dataset into.
    """
    os.makedirs(download_dir, exist_ok=True)
    typer.echo("Downloading PSFHS dataset from Zenodo (Record 10969427)...")
    zenodo_download("10969427", output_dir=download_dir)
    typer.echo(f"Download complete. Data saved to: {download_dir}")
