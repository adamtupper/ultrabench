"""Download the Butterfly dataset from GitHub releases.

The dataset consists of three archives hosted on GitHub:
  - butterfly_dataset_test.tar.gz
  - butterfly_dataset_training1.tar.gz
  - butterfly_dataset_training2.tar.gz

Usage:
    ultrabench butterfly dl DOWNLOAD_DIR
"""

import os
from datetime import date
from typing import Annotated

import typer

from .utils import download_file, extract_archive

URLS = [
    "https://github.com/ButterflyNetwork/MITGrandHack2018/releases/download/v.0.0.1/butterfly_dataset_test.tar.gz",
    "https://github.com/ButterflyNetwork/MITGrandHack2018/releases/download/v.0.0.1/butterfly_dataset_training1.tar.gz",
    "https://github.com/ButterflyNetwork/MITGrandHack2018/releases/download/v.0.0.1/butterfly_dataset_training2.tar.gz",
]


def download_butterfly(
    download_dir: Annotated[
        str, typer.Argument(help="The directory to download the dataset into")
    ],
) -> None:
    """Download the Butterfly dataset archives from GitHub releases.

    Args:
        download_dir: The directory to download the dataset into.
    """
    date_str = date.today().strftime("%Y%m%d")
    output_dir = os.path.join(download_dir, f"butterfly_raw_{date_str}")
    os.makedirs(output_dir, exist_ok=True)
    typer.echo("Downloading Butterfly dataset from GitHub releases...")

    for url in URLS:
        filename = url.split("/")[-1]
        dest = os.path.join(output_dir, filename)
        download_file(url, dest)
        extract_archive(dest, output_dir)
        os.remove(dest)

    typer.echo(f"Download complete. Saved to {output_dir}.")
    typer.echo(
        f"Process the dataset by running `ultrabench process butterfly {output_dir}`."
    )
