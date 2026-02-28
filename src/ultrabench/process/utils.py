"""Utility functions for dataset processing."""

import os


def save_version_info(output_dir: str, version: str) -> None:
    """Create a file with the UltraBench version and current date.

    Args:
        output_dir: The path to the output directory.
        version: The version string.
    """
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"UltraBench Version: {version}\n")
        f.write(f"Created: {os.popen('date').read().strip()}\n")
