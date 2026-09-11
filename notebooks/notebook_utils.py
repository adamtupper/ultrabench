"""Utility functions for notebooks.

Includes the discovery of training runs under a sweeps directory. A run
lives at `<model>/<task>/shot_<pct>/seed_<seed>/<timestamp>/<mode>` below
the sweeps root, and `parse_run_dir` is the single place that layout is
read, so the helpers here cannot disagree about what a run is.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tbparse import SummaryReader
from tqdm.auto import tqdm

from ultrabench2.evaluate.tasks import split_mode

logger = logging.getLogger(__name__)

PREDICTIONS_FILENAME = "predictions.csv"
"""The per-example predictions written at the end of a run."""

TEST_METRICS_FILENAME = "test_metrics.csv"
"""The metrics the training layer writes at the end of a run."""

RUN_DEPTH = 6
"""Path components between the sweeps root and a run directory.

A run lives at `<model>/<task>/shot_<pct>/seed_<seed>/<timestamp>/<mode>`.
"""


@dataclass(frozen=True)
class RunMeta:
    """The identity of a single training run.

    Args:
        path: The run's directory.
        model: The pretrained backbone that was evaluated.
        task: The benchmark task the run was trained on.
        shot_pct: The proportion of the training set that was used.
        seed: The random seed the run used.
        datetime: The run's timestamp, used to pick the latest rerun.
        mode: The head and training paradigm, for example
            `upsampler_head_only`.
    """

    path: Path
    model: str
    task: str
    shot_pct: float
    seed: int
    datetime: str
    mode: str

    @property
    def head(self) -> str:
        """The task head the run trained.

        Returns:
            The name of the head.
        """
        return split_mode(self.mode)[0]

    @property
    def paradigm(self) -> str:
        """The training paradigm the run used.

        Returns:
            Either `head_only` or `full_finetuning`.
        """
        return split_mode(self.mode)[1]

    @property
    def key(self) -> tuple:
        """The identity of the run, ignoring when it was run.

        Returns:
            A tuple identifying the cell of the sweep grid the run fills.
        """
        return (self.model, self.task, self.shot_pct, self.mode, self.seed)


def parse_run_dir(run_dir: Path, sweeps_dir: Path) -> RunMeta:
    """Read a run's identity out of its directory path.

    Args:
        run_dir: The run's directory.
        sweeps_dir: The root the sweep was written under.

    Returns:
        The run's metadata.

    Raises:
        ValueError: If the path is not laid out as a run directory.
    """
    relative = Path(run_dir).relative_to(sweeps_dir)
    if len(relative.parts) != RUN_DEPTH:
        raise ValueError(
            f"Expected a run directory {RUN_DEPTH} levels below {sweeps_dir}, "
            f"got '{relative}' ({len(relative.parts)} levels)."
        )

    model, task, shot, seed, datetime, mode = relative.parts
    split_mode(mode)

    return RunMeta(
        path=Path(run_dir),
        model=model,
        task=task,
        shot_pct=float(shot.split("_")[1]),
        seed=int(seed.split("_")[1]),
        datetime=datetime.replace("_", " "),
        mode=mode,
    )


def find_runs(sweeps_dir: Path) -> list[RunMeta]:
    """Find every run below a sweeps directory.

    Runs whose directory does not parse are skipped with a warning, so
    that an unrelated directory under the sweeps root does not stop the
    collection.

    Args:
        sweeps_dir: The root the sweeps were written under.

    Returns:
        The metadata of every run found.
    """
    sweeps_dir = Path(sweeps_dir)

    runs = []
    for path in sorted(sweeps_dir.rglob(PREDICTIONS_FILENAME)):
        try:
            runs.append(parse_run_dir(path.parent, sweeps_dir))
        except ValueError as error:
            logger.warning("Skipping %s: %s", path.parent, error)

    return runs


def deduplicate_latest(runs: Iterable[RunMeta]) -> list[RunMeta]:
    """Keep only the most recent run of each cell of the sweep grid.

    Args:
        runs: The runs to deduplicate.

    Returns:
        One run per cell, preferring the latest timestamp.
    """
    latest: dict[tuple, RunMeta] = {}
    for run in runs:
        current = latest.get(run.key)
        if current is None or run.datetime > current.datetime:
            latest[run.key] = run

    return sorted(latest.values(), key=lambda run: (run.key, run.datetime))


def _describe_run(run: RunMeta) -> dict:
    """Build the identifying columns of a run's results table.

    Args:
        run: The run to describe.

    Returns:
        A mapping of column name to value, matching the columns the
        results notebooks group by.
    """
    return {
        "model": run.model,
        "task": run.task,
        "shot_pct": run.shot_pct,
        "seed": run.seed,
        "datetime": run.datetime,
        "mode": run.mode,
    }


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
    sweeps_dir = Path(sweeps_dir)
    run_dirs = [p.parent for p in sweeps_dir.rglob(TEST_METRICS_FILENAME)]

    dfs = []
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        run = parse_run_dir(run_dir, sweeps_dir)

        reader = SummaryReader(
            log_path=str(run_dir),
            extra_columns={"dir_name", "file_name"},
            pivot=True,
        )
        df = reader.scalars

        rel_run_dir = run_dir.relative_to(sweeps_dir)
        df["dir_name"] = f"{rel_run_dir}/" + df["dir_name"]
        df = df.assign(**_describe_run(run))

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
    sweeps_dir = Path(sweeps_dir)
    file_paths = list(sweeps_dir.rglob(TEST_METRICS_FILENAME))

    dfs = []
    for file_path in tqdm(file_paths, desc="Processing runs"):
        run = parse_run_dir(file_path.parent, sweeps_dir)

        df = pd.read_csv(file_path)
        df = df.assign(**_describe_run(run))

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
