"""Shared utilities for downloading dataset archives."""

import os
import tarfile
import zipfile

import requests
import typer


def download_file(url: str, dest_path: str) -> None:
    """Download a file from a URL to a local path, streaming with progress output.

    Args:
        url: The URL of the file to download.
        dest_path: The local file path to write the downloaded content to.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    filename = os.path.basename(dest_path)
    typer.echo(f"Downloading {filename}...")

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                typer.echo(
                    f"\r  {pct:.1f}% ({downloaded // 1024 // 1024} MB)", nl=False
                )

    if total:
        typer.echo("")  # newline after progress


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
