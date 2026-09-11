"""Shared utilities for downloading dataset archives."""

import os
import tarfile
import zipfile

import requests
import typer
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)


def download_file(url: str, dest_path: str) -> None:
    """Download a file from a URL to a local path, streaming with progress output.

    Displays a Rich progress bar when the server provides Content-Length, or a
    spinner when the total size is unknown.

    Args:
        url: The URL of the file to download.
        dest_path: The local file path to write the downloaded content to.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0)) or None
    chunk_size = 1024 * 1024  # 1 MB
    filename = os.path.basename(dest_path)

    if total:
        columns = [
            TextColumn(filename),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ]
    else:
        columns = [
            SpinnerColumn(),
            TextColumn(f"Downloading {filename}..."),
            DownloadColumn(),
        ]

    with Progress(*columns, transient=True) as progress:
        task = progress.add_task("", total=total)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress.update(task, advance=len(chunk))


def extract_archive(archive_path: str, output_dir: str) -> None:
    """Extract a .tar.gz or .zip archive into a directory.

    Args:
        archive_path: Path to the archive file.
        output_dir: Directory to extract the archive contents into.
    """
    typer.echo(f"Extracting {os.path.basename(archive_path)}...")

    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
