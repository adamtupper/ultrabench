"""Download the MASLD dataset from Zenodo.

The dataset is available at https://zenodo.org/records/1009146.

Usage:
    ultrabench download masld DOWNLOAD_DIR
"""

import os
from datetime import date
from typing import Annotated

import typer
from zenodo_get import download as zenodo_download


def download_masld(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ] = ".",
) -> None:
    """Download the MASLD dataset from Zenodo.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"masld_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    typer.echo("Downloading MASLD dataset from Zenodo...")
    zenodo_download("1009146", output_dir=output_dir)
    typer.echo(f"Download complete. Saved to {output_dir}.")
    typer.echo(
        f"Process the dataset by running `ultrabench process masld {output_dir}/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat`."  # noqa: E501
    )
