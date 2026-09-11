"""Download the BUS-BRA dataset from Zenodo.

The dataset is available at https://zenodo.org/records/8231412.

Usage:
    ultrabench2 download busbra DOWNLOAD_DIR
"""

import os
from datetime import date
from typing import Annotated

import typer
from zenodo_get import download as zenodo_download

from .utils import extract_archive


def download_busbra(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ] = ".",
) -> None:
    """Download the BUS-BRA dataset from Zenodo.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"busbra_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    typer.echo("Downloading BUS-BRA dataset from Zenodo...")
    zenodo_download("8231412", output_dir=output_dir)

    for filename in os.listdir(output_dir):
        if filename.endswith(".zip"):
            archive_path = os.path.join(output_dir, filename)
            extract_archive(archive_path, output_dir)
            os.remove(archive_path)

    typer.echo(f"Download complete. Saved to {output_dir}.")
    typer.echo(
        f"Process the dataset by running `ultrabench2 process busbra {output_dir}`."
    )
