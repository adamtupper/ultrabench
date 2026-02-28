"""Download the PSFHS dataset from Zenodo.

The dataset is available at https://zenodo.org/records/10969427.

Usage:
    ultrabench psfhs dl DOWNLOAD_DIR
"""

import os
from datetime import date
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
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"psfhs_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    typer.echo("Downloading PSFHS dataset from Zenodo (Record 10969427)...")
    zenodo_download("10969427", output_dir=output_dir)
    typer.echo(f"Download complete. Saved to {output_dir}.")
    typer.echo(
        f"Process the dataset by running `ultrabench process psfhs {output_dir}`."
    )
