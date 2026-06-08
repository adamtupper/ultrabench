"""Utility functions for notebooks."""

from pathlib import Path

import pandas as pd
from tbparse import SummaryReader
from tqdm.auto import tqdm


def parse_tensorboard_logs(sweeps_dir: Path) -> pd.DataFrame:
    """Parse the Tensorboard logs for each run of a set of sweeps.

    Searches for all run directories under `sweeps_dir` (identified by
    the `test_metrics.csv` file) and parses the Tensorboard logs for
    each run.

    Note: This only parses the scalar values of each set of logs.

    Args:
        sweeps_dir: The root directory to search for runs.

    Returns:
        Data frame containing the parsed Tensorboard logs.
    """
    run_dirs = [p.parent for p in sweeps_dir.rglob("test_metrics.csv")]

    dfs = []
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        reader = SummaryReader(
            log_path=str(run_dir),
            extra_columns={"dir_name", "file_name"},
            pivot=True,
        )
        df = reader.scalars

        rel_run_dir = run_dir.relative_to(sweeps_dir)
        df["dir_name"] = f"{rel_run_dir}/" + df["dir_name"]

        model, task, shot_str, seed_str, datetime, mode = rel_run_dir.parts
        df["model"] = model
        df["task"] = task
        df["shot_pct"] = float(shot_str.split("_")[1])
        df["seed"] = int(seed_str.split("_")[1])
        df["datetime"] = datetime.replace("_", " ")
        df["mode"] = mode

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def aggregate_test_metrics(sweeps_dir: Path) -> pd.DataFrame:
    """Aggregate the test metrics for each run of a set of sweeps.

    Searches for all run directories and parses the `test_metrics.csv` file.

    Args:
        sweeps_dir: The root directory to search for runs.

    Returns:
        Data frame containing the test metrics and key configuration parameters
        for each run.
    """
    file_paths = [p for p in sweeps_dir.rglob("test_metrics.csv")]

    dfs = []
    for file_path in tqdm(file_paths, desc="Processing runs"):
        df = pd.read_csv(file_path)

        rel_run_dir = file_path.parent.relative_to(sweeps_dir)

        model, task, shot_str, seed_str, datetime, mode = rel_run_dir.parts
        df["model"] = model
        df["task"] = task
        df["shot_pct"] = float(shot_str.split("_")[1])
        df["seed"] = int(seed_str.split("_")[1])
        df["datetime"] = datetime.replace("_", " ")
        df["mode"] = mode

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
