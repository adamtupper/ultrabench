"""Download the Fatty Liver dataset from Zenodo.

The dataset is available at https://zenodo.org/records/1009146.

Usage:
    ultrabench fatty-liver dl DOWNLOAD_DIR
"""

import os
from typing import Annotated

import typer
from zenodo_get import download as zenodo_download


def download_fatty_liver(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ],
) -> None:
    """Download the Fatty Liver dataset from Zenodo (Record 1009146).

    Args:
        download_dir: The directory to download the dataset into.
    """
    os.makedirs(download_dir, exist_ok=True)
    typer.echo("Downloading Fatty Liver dataset from Zenodo (Record 1009146)...")
    zenodo_download("1009146", output_dir=download_dir)
    typer.echo(f"Download complete. Data saved to: {download_dir}")
