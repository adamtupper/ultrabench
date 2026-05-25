"""Download the STMUS dataset from Mendeley Data.

The dataset is available at
https://data.mendeley.com/datasets/3jykz7wz8d/1.

Usage:
    ultrabench download stmus DOWNLOAD_DIR
"""

import os
from datetime import date
from typing import Annotated

import typer

from .utils import download_file, extract_archive

_DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/3jykz7wz8d/download/1"


def download_stmus(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ] = ".",
) -> None:
    """Download the STMUS dataset from Mendeley Data.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"stmus_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)

    archive_path = os.path.join(output_dir, "stmus.zip")
    typer.echo("Downloading STMUS dataset from Mendeley Data...")
    download_file(_DOWNLOAD_URL, archive_path)
    extract_archive(archive_path, output_dir)
    os.remove(archive_path)

    nested_dir = os.path.join(
        output_dir,
        "DATASET for Deep learning segmentation of transverse"
        " musculoskeletal ultrasound images for neuromuscular"
        " disease assessment",
    )
    nested_archive = os.path.join(
        nested_dir,
        "DATASET_10.1016j.compbiomed.2021.104623.zip",
    )
    extract_archive(nested_archive, output_dir)
    os.remove(nested_archive)
    os.rmdir(nested_dir)

    typer.echo(f"Download complete. Saved to {output_dir}.")
    typer.echo(
        f"Process the dataset by running `ultrabench process stmus {output_dir}`."
    )
