"""Common utility functions for tests for the evaluate module."""

from pathlib import Path

import pandas as pd


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    """Write a CSV file with the given rows.

    Each row is a dictionary where the keys are column names and the
    values are the corresponding cell values.

    Args:
        path: Path to the CSV file to write.
        rows: List of dictionaries representing the rows of the CSV file.
        columns: Optional list of column names to use as the header of
            the CSV file.
    """
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
