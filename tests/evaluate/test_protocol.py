"""Unit tests for the submission scoring protocol."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from ultrabench.core.rle import mask_to_rle
from ultrabench.evaluate import protocol
from ultrabench.evaluate.tasks import CATEGORIES, get_task

IMAGE_SIZE = 256
"""The size of the test images every benchmark task is scored at."""

N_RESAMPLES = 64
"""Replications drawn in the tests, where the count is not the point."""


def _units(count: int) -> np.ndarray:
    """Build a canonical array of unit IDs.

    Args:
        count: The number of units.

    Returns:
        The unit IDs, sorted.
    """
    return np.array([f"unit_{index:04d}" for index in range(count)])


def test_draws_ignore_row_order_and_repeats():
    """The draw depends on the set of unit IDs, not on their order."""
    units = _units(500)
    shuffled = np.random.default_rng(1).permutation(units)

    baseline = protocol.resample_indices(units, "A", N_RESAMPLES)

    assert np.array_equal(baseline, protocol.resample_indices(units, "A", N_RESAMPLES))
    assert np.array_equal(
        baseline, protocol.resample_indices(shuffled, "A", N_RESAMPLES)
    )
    assert np.array_equal(
        baseline,
        protocol.resample_indices(
            np.concatenate([shuffled, shuffled]), "A", N_RESAMPLES
        ),
    )


def test_draws_differ_across_datasets():
    """Two datasets of the same size draw independently."""
    units = _units(500)

    assert not np.array_equal(
        protocol.resample_indices(units, "A", N_RESAMPLES),
        protocol.resample_indices(units, "B", N_RESAMPLES),
    )


def test_multiplicity_matrix_counts_the_draws():
    """Each replication draws as many units as the dataset has."""
    units = _units(50)

    multiplicity = protocol.multiplicity_matrix(units, "A", N_RESAMPLES)

    assert multiplicity.shape == (N_RESAMPLES, len(units))
    assert np.array_equal(multiplicity.sum(axis=1), np.full(N_RESAMPLES, len(units)))


def test_multiplicity_matrix_matches_the_drawn_ids():
    """The counts are those of the IDs `resample_indices` draws."""
    units = _units(20)

    multiplicity = protocol.multiplicity_matrix(units, "A", 8)
    draws = protocol.resample_indices(units, "A", 8)

    for row, drawn in zip(multiplicity, draws):
        expected = [int((drawn == unit).sum()) for unit in units]
        assert row.tolist() == expected


def _confusion(
    labels_gt: np.ndarray, labels_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """Build the confusion matrix of one set of predictions.

    Args:
        labels_gt: The ground-truth label of each row.
        labels_pred: The predicted label of each row.
        n_classes: The number of classes of the task.

    Returns:
        The confusion matrix, of shape (1, n_classes, n_classes).
    """
    confusion = np.zeros((1, n_classes, n_classes))
    for label_gt, label_pred in zip(labels_gt, labels_pred):
        confusion[0, label_gt, label_pred] += 1.0

    return confusion


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_macro_f1_matches_sklearn(n_classes):
    """The vectorised macro F1 reproduces `sklearn.metrics.f1_score`."""
    rng = np.random.default_rng(0)
    labels_gt = rng.integers(0, n_classes, 200)
    labels_pred = rng.integers(0, n_classes, 200)

    confusion = _confusion(labels_gt, labels_pred, n_classes)

    assert protocol.macro_f1_from_confusion(confusion)[0] == pytest.approx(
        f1_score(labels_gt, labels_pred, average="macro")
    )


def test_macro_f1_skips_absent_classes():
    """A class in neither the truth nor the prediction is not averaged."""
    labels_gt = np.array([0, 0, 1, 1])
    labels_pred = np.array([0, 1, 1, 1])

    confusion = _confusion(labels_gt, labels_pred, 3)

    assert protocol.macro_f1_from_confusion(confusion)[0] == pytest.approx(
        f1_score(labels_gt, labels_pred, average="macro")
    )


def test_interval_combines_both_variance_terms():
    """The error bar is sqrt(sigma_run^2 / S + sigma_boot^2)."""
    per_run = np.array([[0.6, 0.7, 0.8], [0.2, 0.3, 0.4]])
    per_replication = np.array([[0.6, 0.65, 0.7], [0.2, 0.25, 0.3]])

    result = protocol.interval(per_run, per_replication, confidence=1.96)

    sigma_run = per_run.mean(axis=0).std(ddof=1)
    sigma_boot = per_replication.mean(axis=0).std(ddof=1)
    expected = np.sqrt(sigma_run**2 / 3 + sigma_boot**2)

    assert result["mean"] == pytest.approx(0.5)
    assert result["se"] == pytest.approx(expected)
    assert result["ci_low"] == pytest.approx(0.5 - 1.96 * expected)
    assert result["ci_high"] == pytest.approx(0.5 + 1.96 * expected)
    assert result["k_tasks"] == 2
    assert result["n_runs"] == 3


def test_interval_rejects_a_single_run():
    """One run leaves the run term undefined rather than zero."""
    with pytest.raises(ValueError, match="at least two runs"):
        protocol.interval(np.array([[0.5]]), np.array([[0.5, 0.6]]))


def test_ground_truth_files_match_the_registry():
    """Every task names the ground truth of its own split and type."""
    protocol.check_ground_truth_files()


def _submission(root: Path, task: str, runs: int) -> Path:
    """Create an empty submission directory for one task.

    Args:
        root: The root of the submission.
        task: The name of the task.
        runs: The number of run CSVs to create.

    Returns:
        The task's directory.
    """
    directory = root / task
    directory.mkdir(parents=True)
    for index in range(runs):
        (directory / f"seed_{index}.csv").write_text("image_id,label\n")

    return directory


def test_find_predictions_reads_one_directory_per_task(tmp_path):
    """Each task directory contributes its CSVs in sorted order."""
    _submission(tmp_path, "gbcu_classification", runs=5)
    _submission(tmp_path, "psfhs_segmentation", runs=5)

    found = protocol.find_predictions(tmp_path)

    assert sorted(found) == ["gbcu_classification", "psfhs_segmentation"]
    assert [path.name for path in found["gbcu_classification"]] == [
        f"seed_{index}.csv" for index in range(5)
    ]


def test_find_predictions_selects_the_requested_tasks(tmp_path):
    """Naming the tasks scores only those."""
    _submission(tmp_path, "gbcu_classification", runs=5)
    _submission(tmp_path, "psfhs_segmentation", runs=5)

    found = protocol.find_predictions(tmp_path, tasks=["psfhs_segmentation"])

    assert list(found) == ["psfhs_segmentation"]


def test_find_predictions_rejects_an_unknown_task(tmp_path):
    """A directory that names no benchmark task is an error, not a skip."""
    _submission(tmp_path, "gbcu_classification", runs=5)
    _submission(tmp_path, "gbcu_classifcation", runs=5)

    with pytest.raises(ValueError, match="do not name a benchmark task"):
        protocol.find_predictions(tmp_path)


def test_find_predictions_enforces_the_run_count(tmp_path):
    """The protocol's five runs per task are enforced by default."""
    _submission(tmp_path, "gbcu_classification", runs=3)

    with pytest.raises(ValueError, match="requires 5 runs"):
        protocol.find_predictions(tmp_path)

    assert len(protocol.find_predictions(tmp_path, runs_per_task=None)) == 1


def test_find_predictions_rejects_a_missing_task(tmp_path):
    """Asking for a task the submission does not cover is an error."""
    _submission(tmp_path, "gbcu_classification", runs=5)

    with pytest.raises(ValueError, match="no predictions for"):
        protocol.find_predictions(tmp_path, tasks=["psfhs_segmentation"])


def _category_inputs(tasks, n_runs=5, n_replications=16):
    """Build synthetic per-run and per-replication scores.

    Args:
        tasks: The tasks to score.
        n_runs: The number of runs per task.
        n_replications: The number of replications per task.

    Returns:
        The per-run and the per-replication scores, keyed by task.
    """
    rng = np.random.default_rng(0)

    return (
        {task: rng.uniform(0.4, 0.9, n_runs) for task in tasks},
        {task: rng.uniform(0.4, 0.9, n_replications) for task in tasks},
    )


def test_category_score_averages_its_tasks():
    """A category score is the unweighted mean of its tasks' scores."""
    category = CATEGORIES["anatomy_classification"]
    run_scores, replication_scores = _category_inputs(category.tasks)

    table = protocol._category_scores(run_scores, replication_scores, 1.96)

    row = table[table["category"] == "anatomy_classification"].iloc[0]
    expected = np.mean([run_scores[task].mean() for task in category.tasks])

    assert row["mean"] == pytest.approx(expected)
    assert row["k_tasks"] == len(category.tasks)


def test_incomplete_category_is_skipped():
    """A category is reported only when every one of its tasks is in."""
    category = CATEGORIES["anatomy_classification"]
    run_scores, replication_scores = _category_inputs(category.tasks[:-1])

    table = protocol._category_scores(run_scores, replication_scores, 1.96)

    assert table.empty
    assert list(table.columns) == protocol._CATEGORY_COLUMNS


def test_leaderboard_row_needs_every_category():
    """The row is offered only for a submission covering the benchmark."""
    partial = pd.DataFrame(
        {"category": ["anatomy_classification"], "mean": [0.5], "se": [0.01]}
    )

    assert protocol.leaderboard_row(partial) is None

    complete = pd.DataFrame(
        {
            "category": list(CATEGORIES),
            "mean": [0.5, 0.25, 0.125, 0.06289],
            "se": [0.01, 0.02, 0.003, 0.00051],
        }
    )

    assert protocol.leaderboard_row(complete) == (
        "<model>,<url>,<release_date>,<params>,"
        "0.5000,0.0100,0.2500,0.0200,0.1250,0.0030,0.0629,0.0005"
    )


def _write_classification_ground_truth(
    path: Path, labels: dict[str, int], sequences: dict[str, str] | None = None
) -> None:
    """Write a classification ground-truth CSV.

    Args:
        path: The file to write.
        labels: The label of each image.
        sequences: The sequence each image belongs to, where the task is
            scored per sequence.
    """
    table = pd.DataFrame({"image_id": list(labels), "label": list(labels.values())})
    if sequences is not None:
        table["sequence_id"] = [sequences[image_id] for image_id in labels]

    table.to_csv(path, index=False)


def _write_classification_predictions(
    path: Path, labels: dict[str, int], n_classes: int
) -> None:
    """Write a classification predictions CSV with hard probabilities.

    Args:
        path: The file to write.
        labels: The predicted label of each image.
        n_classes: The number of classes of the task.
    """
    table = pd.DataFrame({"image_id": list(labels), "label": list(labels.values())})
    for index in range(n_classes):
        table[f"prob_{index}"] = (table["label"] == index).astype(float)

    table.to_csv(path, index=False)


def test_scores_a_classification_submission(tmp_path, monkeypatch):
    """The per-task score of a real task reproduces macro F1."""
    task = "gbcu_classification"
    n_classes = get_task(task).num_classes
    rng = np.random.default_rng(7)
    images = [f"image_{index:03d}" for index in range(60)]
    truth = dict(zip(images, rng.integers(0, n_classes, len(images)).tolist()))

    ground_truth = tmp_path / "ground_truth.csv"
    _write_classification_ground_truth(ground_truth, truth)
    monkeypatch.setattr(protocol, "ground_truth_path", lambda _: ground_truth)

    predicted = []
    directory = tmp_path / "runs" / task
    directory.mkdir(parents=True)
    for index in range(5):
        labels = dict(zip(images, rng.integers(0, n_classes, len(images)).tolist()))
        predicted.append(labels)
        _write_classification_predictions(
            directory / f"seed_{index}.csv", labels, n_classes
        )

    task_scores, category_scores = protocol.score_submission(
        tmp_path / "runs", n_resamples=N_RESAMPLES
    )

    expected = np.mean(
        [
            f1_score(
                [truth[image] for image in images],
                [labels[image] for image in images],
                average="macro",
            )
            for labels in predicted
        ]
    )
    row = task_scores.iloc[0]

    assert row["task"] == task
    assert row["metric"] == "f1_macro"
    assert row["unit"] == "image"
    assert row["n_rows"] == len(images)
    assert row["n_runs"] == 5
    assert row["mean"] == pytest.approx(expected)
    assert row["se"] > 0
    assert row["ci_high"] - row["ci_low"] == pytest.approx(2 * 1.96 * row["se"])
    assert category_scores.empty


def test_scores_a_sequence_task_over_patients(tmp_path, monkeypatch):
    """A sequence-level task is scored per sequence and resampled per
    patient.
    """
    task = "masld_classification"
    images = [
        f"patient_{patient}_frame_{frame}" for patient in range(6) for frame in range(4)
    ]
    sequences = {image: image.split("_frame_")[0] for image in images}
    truth = {image: int(sequences[image].endswith(("0", "2", "4"))) for image in images}

    ground_truth = tmp_path / "ground_truth.csv"
    _write_classification_ground_truth(ground_truth, truth, sequences)
    monkeypatch.setattr(protocol, "ground_truth_path", lambda _: ground_truth)
    monkeypatch.setattr(
        protocol, "load_patient_map", lambda spec, cache_dir=None: dict(sequences)
    )

    directory = tmp_path / "runs" / task
    directory.mkdir(parents=True)
    for index in range(5):
        # Every frame of the first patient is misread, so exactly one of
        # the six sequences is wrong in every run.
        labels = {
            image: (
                1 - truth[image] if sequences[image] == "patient_0" else truth[image]
            )
            for image in images
        }
        _write_classification_predictions(directory / f"seed_{index}.csv", labels, 2)

    task_scores, _ = protocol.score_submission(
        tmp_path / "runs", n_resamples=N_RESAMPLES
    )

    row = task_scores.iloc[0]
    pooled_truth = [
        truth[f"{sequence}_frame_0"] for sequence in sorted(set(sequences.values()))
    ]
    pooled_pred = [
        1 - label if sequence == "patient_0" else label
        for sequence, label in zip(sorted(set(sequences.values())), pooled_truth)
    ]

    assert row["metric"] == "seq_f1_macro"
    assert row["unit"] == "patient"
    assert row["n_rows"] == 6
    assert row["n_units"] == 6
    assert row["mean"] == pytest.approx(
        f1_score(pooled_truth, pooled_pred, average="macro")
    )
    assert row["sigma_run"] == pytest.approx(0.0)
    assert row["sigma_boot"] > 0


def _mask(size: int, fill: bool) -> np.ndarray:
    """Build a square binary mask.

    Args:
        size: The height and width of the mask.
        fill: Whether the top-left quadrant is set.

    Returns:
        The mask.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    if fill:
        mask[: size // 2, : size // 2] = 1

    return mask


def test_scores_a_segmentation_submission(tmp_path, monkeypatch):
    """A perfect segmentation run scores a positive Dice of one."""
    task = "psfhs_segmentation"
    spec = get_task(task)
    images = [f"image_{index:02d}" for index in range(8)]

    rows = [
        {
            "image_id": image,
            "class_id": class_id,
            "encoded_pixels": mask_to_rle(_mask(spec.image_size, class_id == 1)),
        }
        for image in images
        for class_id in spec.foreground_classes
    ]
    ground_truth = tmp_path / "ground_truth.csv"
    pd.DataFrame(rows).to_csv(ground_truth, index=False)
    monkeypatch.setattr(protocol, "ground_truth_path", lambda _: ground_truth)

    directory = tmp_path / "runs" / task
    directory.mkdir(parents=True)
    for index in range(5):
        pd.DataFrame(rows).to_csv(directory / f"seed_{index}.csv", index=False)

    task_scores, _ = protocol.score_submission(
        tmp_path / "runs", n_resamples=N_RESAMPLES
    )

    row = task_scores.iloc[0]

    assert row["metric"] == "positive_dice"
    assert row["n_rows"] == len(images)
    # Class 2 is empty in both masks of every image, so it is undefined
    # and drops out, leaving the perfect class 1 alone.
    assert row["mean"] == pytest.approx(1.0)
    assert row["se"] == pytest.approx(0.0)


def test_missing_predictions_are_an_error(tmp_path, monkeypatch):
    """A run that does not cover the test split cannot be scored."""
    task = "gbcu_classification"
    images = [f"image_{index:02d}" for index in range(10)]
    truth = dict.fromkeys(images, 0)

    ground_truth = tmp_path / "ground_truth.csv"
    _write_classification_ground_truth(ground_truth, truth)
    monkeypatch.setattr(protocol, "ground_truth_path", lambda _: ground_truth)

    directory = tmp_path / "runs" / task
    directory.mkdir(parents=True)
    for index in range(5):
        labels = truth if index else dict(list(truth.items())[:-1])
        _write_classification_predictions(directory / f"seed_{index}.csv", labels, 3)

    with pytest.raises(ValueError, match="no scored entries"):
        protocol.score_submission(tmp_path / "runs", n_resamples=N_RESAMPLES)
