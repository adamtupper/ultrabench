"""Unit tests for the notebook helpers that discover training runs."""

from pathlib import Path

import pandas as pd
import pytest

from notebook_utils import (
    aggregate_test_metrics,
    deduplicate_latest,
    find_runs,
    parse_run_dir,
)

CLASSIFICATION_TASK = "tn3k_classification"
PREDICTIONS = pd.DataFrame({"image_id": ["a.png", "b.png"], "label": [0, 1]})


def write_run(
    sweeps_dir: Path,
    model: str,
    task: str,
    seed: int,
    datetime: str,
    mode: str,
    filename: str = "predictions.csv",
    shot_pct: float = 1.0,
) -> Path:
    """Create a run directory holding a per-run CSV.

    Args:
        sweeps_dir: The root to write the run under.
        model: The pretrained backbone the run evaluated.
        task: The benchmark task the run was trained on.
        seed: The random seed the run used.
        datetime: The run's timestamp, as it appears in the path.
        mode: The head and training paradigm, e.g. `upsampler_head_only`.
        filename: The CSV to write into the run directory.
        shot_pct: The proportion of the training set that was used.

    Returns:
        The run's directory.
    """
    run_dir = (
        Path(sweeps_dir)
        / model
        / task
        / f"shot_{shot_pct}"
        / f"seed_{seed}"
        / datetime
        / mode
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.to_csv(run_dir / filename, index=False)

    return run_dir


@pytest.fixture
def sweep(tmp_path):
    """Build a small sweeps directory of six runs.

    Args:
        tmp_path: The temporary directory to build under.

    Returns:
        The sweeps directory.
    """
    sweeps_dir = tmp_path / "runs"

    for model in ("model_a", "model_b"):
        for seed in (42, 43, 44):
            write_run(
                sweeps_dir,
                model,
                CLASSIFICATION_TASK,
                seed,
                "2026-09-01_10-00-00",
                "linear_classifier_head_only",
            )

    return sweeps_dir


def test_parse_run_dir(tmp_path):
    """Check that a run's identity is read out of its path."""
    run_dir = (
        tmp_path
        / "deblurmim_mae"
        / "busbra_segmentation"
        / "shot_1.0"
        / "seed_42"
        / "2026-09-03_08-47-34"
        / "upsampler_head_only"
    )
    run = parse_run_dir(run_dir, tmp_path)

    assert run.model == "deblurmim_mae"
    assert run.task == "busbra_segmentation"
    assert run.shot_pct == 1.0
    assert run.seed == 42
    assert run.datetime == "2026-09-03 08-47-34"
    assert run.head == "upsampler"
    assert run.paradigm == "head_only"


def test_parse_run_dir_rejects_the_wrong_depth(tmp_path):
    """Check that a path that is not a run directory is rejected."""
    run_dir = tmp_path / "model" / "task" / "upsampler_head_only"

    with pytest.raises(ValueError, match="6 levels below"):
        parse_run_dir(run_dir, tmp_path)


def test_parse_run_dir_rejects_an_unknown_paradigm(tmp_path):
    """Check that a run whose mode has no known paradigm is rejected."""
    run_dir = (
        tmp_path / "m" / "t" / "shot_1.0" / "seed_42" / "2026-09-03_08-47-34" / "probe"
    )

    with pytest.raises(ValueError, match="Cannot parse mode 'probe'"):
        parse_run_dir(run_dir, tmp_path)


def test_find_runs_skips_unparseable_directories(sweep, caplog):
    """Check that an unrelated predictions CSV does not stop collection."""
    stray = sweep / "stray"
    stray.mkdir()
    (stray / "predictions.csv").write_text("image_id,label\n")

    runs = find_runs(sweep)

    assert len(runs) == 6
    assert "Skipping" in caplog.text


def test_deduplicate_latest_keeps_the_newest_rerun(tmp_path):
    """Check that a rerun cell keeps only its most recent run."""
    for datetime in ("2026-09-01_10-00-00", "2026-09-02_11-00-00"):
        write_run(
            tmp_path,
            "model_a",
            CLASSIFICATION_TASK,
            42,
            datetime,
            "linear_classifier_head_only",
        )

    runs = deduplicate_latest(find_runs(tmp_path))

    assert len(runs) == 1
    assert runs[0].datetime == "2026-09-02 11-00-00"


def test_aggregate_test_metrics_labels_every_row(tmp_path):
    """Check that the metrics table carries each run's identity.

    The results notebooks group by these columns, so they have to match
    the directory the metrics were read from.
    """
    write_run(
        tmp_path,
        "model_a",
        CLASSIFICATION_TASK,
        42,
        "2026-09-01_10-00-00",
        "linear_classifier_head_only",
        filename="test_metrics.csv",
    )

    metrics = aggregate_test_metrics(tmp_path)

    assert len(metrics) == 2
    assert set(metrics["model"]) == {"model_a"}
    assert set(metrics["task"]) == {CLASSIFICATION_TASK}
    assert set(metrics["shot_pct"]) == {1.0}
    assert set(metrics["seed"]) == {42}
    assert set(metrics["datetime"]) == {"2026-09-01 10-00-00"}
    assert set(metrics["mode"]) == {"linear_classifier_head_only"}
