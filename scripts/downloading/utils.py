"""Shared utilities for downloading dataset archives."""

import os
import tarfile
import zipfile

import requests
from tqdm.auto import tqdm


def download_file(url: str, dest_path: str) -> None:
    """Download a file from a URL to a local path, showing progress.

    Args:
        url: The URL of the file to download.
        dest_path: The local file path to write the downloaded content to.

    Raises:
        requests.HTTPError: If the server returns an error status.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0)) or None
    chunk_size = 1024 * 1024  # 1 MB

    with (
        open(dest_path, "wb") as f,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(dest_path),
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            progress.update(len(chunk))


def print_next_step(dataset: str, raw_path: str) -> None:
    """Tell the maintainer how to process the dataset they downloaded.

    Args:
        dataset: The dataset name, as `process_dataset.py` expects it.
        raw_path: The path to pass to `process_dataset.py` as the raw
            data.
    """
    print(f"Download complete. Saved to {raw_path}.")
    print(
        "Process it with:\n"
        f"    uv run python scripts/process_dataset.py {dataset} {raw_path} "
        "OUTPUT_DIR"
    )


def extract_archive(archive_path: str, output_dir: str) -> None:
    """Extract a .tar.gz or .zip archive into a directory.

    Args:
        archive_path: Path to the archive file.
        output_dir: Directory to extract the archive contents into.
    """
    print(f"Extracting {os.path.basename(archive_path)}...")

    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
