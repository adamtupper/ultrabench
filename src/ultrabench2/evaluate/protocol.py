"""Score a submission the way `docs/evaluation_protocol.md` specifies.

Reads a submission -- one directory of prediction CSVs per task, each
holding one CSV per training run -- and reports the score of every task
and of every category together with its error bar. This is the runnable
form of `notebooks/simplified_protocol.ipynb`, with the sweep-directory
walk replaced by a layout a submitter can produce from any framework::

    python -m ultrabench2.evaluate.protocol predictions=path/to/runs

    path/to/runs/
        tn3k_segmentation/
            seed_0.csv
            ...
        tn3k_classification/
            ...

The error bar combines the variability across training runs with the
sampling noise of the test set. The second term comes from bootstrap
replications whose draws are generated from a frozen seed at run time,
so a submission scored here and the published results are bootstrapped
on identical resamples without either side exchanging any files.

Scoring a subset of the benchmark is supported and gives the same
numbers as scoring all of it: a replication resamples the test set of a
*source dataset*, so the resampling cohort of a dataset is built from
every benchmark task that uses it, whether or not the submission covers
them. That costs the ground truth of the tasks left out, never their
predictions.
"""

import logging
import tempfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from ultrabench2.core.aggregation import aggregate_sequence_probabilities
from ultrabench2.datasets import load_dataset
from ultrabench2.datasets.utils import resolve_cache_dir
from ultrabench2.evaluate.segmentation import SegmentationEvaluator
from ultrabench2.evaluate.tasks import CATEGORIES, TaskSpec, get_task

logger = logging.getLogger(__name__)

RESAMPLE_SEED = 20260906
"""Frozen at release. Never change: it carries the whole guarantee."""

N_RESAMPLES = 2000
"""Bootstrap replications drawn per source dataset (B in the formula)."""

REQUIRED_RUNS = 5
"""Runs per task the submission protocol enforces."""

CONFIDENCE_MULTIPLIER = 1.96
"""Normal quantile used for the interval. No t correction, by design."""

PATIENT_COLUMN = "patient"
"""The dataset column holding a per-image patient identifier."""

PATIENT_DATASETS = frozenset(
    {"BUS-BRA", "CAMUS", "COVID-BLUES", "FetalPlanes", "MASLD", "STMUS"}
)
"""The datasets resampled per patient rather than per image.

Declared rather than discovered, because which cohort a dataset is
resampled over is part of the protocol: a dataset that gained a patient
column later must not silently change the draws. The column is checked
to exist when the map is built, so the two cannot drift apart.
"""

BENCHMARK_TASKS = tuple(
    sorted({task for category in CATEGORIES.values() for task in category.tasks})
)
"""Every task the benchmark reports a score for."""

TASK_CATEGORIES = {
    task: name for name, category in CATEGORIES.items() for task in category.tasks
}
"""The category each benchmark task belongs to."""

GROUND_TRUTH_FILES = {
    "aul_liver_segmentation": "test.256-ground-truth-segmentation-liver.csv",
    "aul_mass_classification": "test.256-ground-truth-classification.csv",
    "aul_mass_segmentation": "test.256-ground-truth-segmentation-mass.csv",
    "busbra_histology_classification": (
        "test.256-ground-truth-classification-histology.csv"
    ),
    "busbra_pathology_classification": (
        "test.256-ground-truth-classification-pathology.csv"
    ),
    "busbra_segmentation": "test.256-ground-truth-segmentation.csv",
    "camus_classification": "classification-test.256-ground-truth-classification.csv",
    "camus_segmentation": "segmentation-test.256-ground-truth-segmentation.csv",
    "covid_blues_classification": "test.256-ground-truth-classification.csv",
    "fetal_planes_brain_plane_classification": (
        "fetal-brain-test.256-ground-truth-classification.csv"
    ),
    "fetal_planes_plane_classification": (
        "fetal-test.256-ground-truth-classification.csv"
    ),
    "gbcu_classification": "test.256-ground-truth-classification.csv",
    "masld_classification": "test.256-ground-truth-classification.csv",
    "mmotu_classification": "test.256-ground-truth-classification.csv",
    "mmotu_segmentation": "test.256-ground-truth-segmentation.csv",
    "open_kidney_capsule_segmentation": (
        "test.256-ground-truth-segmentation-capsule.csv"
    ),
    "open_kidney_regions_segmentation": (
        "test.256-ground-truth-segmentation-regions.csv"
    ),
    "psfhs_segmentation": "test.256-ground-truth-segmentation.csv",
    "stmus_segmentation": "test.256-ground-truth-segmentation.csv",
    "tn3k_classification": "test.256-ground-truth-classification.csv",
    "tn3k_segmentation": "test.256-ground-truth-segmentation.csv",
}
"""The published filename of each task's ground truth.

A dataset repository holds every ground-truth CSV in its root, so the
name carries the split and, for a dataset with more than one subset, the
subset. The task suffix is not derivable from the registry -- AUL names
its two segmentation CSVs after the mask while TN3K has only one -- so
the mapping is written out and checked by `check_ground_truth_files`.
"""


def check_ground_truth_files() -> None:
    """Check the ground-truth filenames against the task registry.

    A hand-maintained mapping is easy to get wrong, and naming the CSV
    of the wrong split would score resized predictions against
    full-resolution masks rather than fail.

    Raises:
        ValueError: If the mapping does not name every benchmark task
            exactly once, or names a file belonging to the wrong split
            or the wrong task type.
    """
    if set(GROUND_TRUTH_FILES) != set(BENCHMARK_TASKS):
        raise ValueError("GROUND_TRUTH_FILES does not name every task exactly once.")

    for task, filename in GROUND_TRUTH_FILES.items():
        spec = get_task(task)
        split = f"test.{spec.image_size}-"
        if split not in filename:
            raise ValueError(f"'{filename}' is not the ground truth of a '{split}'.")
        if spec.task_type not in filename:
            raise ValueError(f"'{filename}' is not a '{spec.task_type}' task.")


check_ground_truth_files()


@dataclass(frozen=True)
class TaskCohort:
    """What a task's rows are and which unit each of them resamples with.

    Args:
        task: The name of the task.
        dataset: The Hugging Face dataset the task comes from, which is
            what owns the bootstrap draw.
        rows: The sorted unique IDs of the rows the task scores, which
            are sequences for a sequence-level task and images
            otherwise.
        units: The resampling unit of each row, in the order of `rows`.
        unit_kind: Whether the units are patients or images.
        unit_index: The position of each row's unit in the sorted units
            of the source dataset.
    """

    task: str
    dataset: str
    rows: np.ndarray
    units: np.ndarray
    unit_kind: str
    unit_index: np.ndarray


@dataclass(frozen=True)
class Cohorts:
    """The resampling cohorts of a set of tasks and their datasets.

    Args:
        tasks: The cohort of each task, keyed by task name.
        dataset_units: The sorted unique units of each source dataset,
            taken over every benchmark task built on it.
    """

    tasks: dict[str, TaskCohort]
    dataset_units: dict[str, np.ndarray]


def resample_indices(
    unit_ids: np.ndarray, dataset: str, n_resamples: int = N_RESAMPLES
) -> np.ndarray:
    """Draw the frozen bootstrap resamples of a source dataset's units.

    Three details carry the guarantee that two submitters draw the same
    resamples: `np.unique` sorts, so the draw depends on the set of unit
    IDs rather than the order the rows were written in; the generator is
    seeded from the frozen release seed and the dataset's name, so two
    datasets of the same size still draw independently while every task
    built on one dataset draws alike; and `np.random.default_rng` uses
    PCG64, whose stream NumPy guarantees across versions.

    Args:
        unit_ids: The unit IDs of the dataset's test split, in any order
            and with any repeats.
        dataset: The name of the source dataset, mixed into the seed.
        n_resamples: The number of resamples to draw.

    Returns:
        An array of shape (n_resamples, n_units) holding the unit IDs
        drawn for each resample.
    """
    units = np.unique(unit_ids)  # sorted, canonical
    seed = [RESAMPLE_SEED, zlib.crc32(dataset.encode())]
    rng = np.random.default_rng(seed)  # PCG64: stream is version-stable

    return units[rng.integers(0, len(units), (n_resamples, len(units)))]


def resample_positions(
    units: np.ndarray, dataset: str, n_resamples: int = N_RESAMPLES
) -> np.ndarray:
    """Express the frozen resamples as positions into the sorted units.

    Every metric here is computed from per-unit contributions held in
    sorted-unit order, so the draws are more useful as integer positions
    than as the IDs themselves.

    Args:
        units: The sorted unique unit IDs of a dataset's test split.
        dataset: The name of the source dataset.
        n_resamples: The number of resamples to draw.

    Returns:
        An array of shape (n_resamples, n_units) of positions.

    Raises:
        ValueError: If the positions do not reproduce the drawn IDs.
    """
    draws = resample_indices(units, dataset, n_resamples)
    positions = np.searchsorted(units, draws)

    if not np.array_equal(units[positions], draws):
        raise ValueError("Resample positions do not reproduce the drawn unit IDs.")

    return positions


def multiplicity_matrix(
    units: np.ndarray, dataset: str, n_resamples: int = N_RESAMPLES
) -> np.ndarray:
    """Count how many times each unit is drawn in each replication.

    Weighting a run's per-unit contributions by this matrix recomputes
    its metric on every replication at once.

    Args:
        units: The sorted unique unit IDs of a dataset's test split.
        dataset: The name of the source dataset.
        n_resamples: The number of resamples to draw.

    Returns:
        An array of shape (n_resamples, n_units) of draw counts.
    """
    positions = resample_positions(units, dataset, n_resamples)
    n_units = len(units)

    flat = positions + np.arange(n_resamples)[:, None] * n_units
    counts = np.bincount(flat.ravel(), minlength=n_resamples * n_units)

    return counts.reshape(n_resamples, n_units).astype(np.float32)


@cache
def ground_truth_path(task: str) -> Path:
    """Download the ground-truth CSV published with a task's dataset.

    The result is memoised, so scoring the runs of a task resolves the
    file once rather than once per run.

    The revision comes from the task registry, so the ground truth can
    only ever be the one belonging to the pinned dataset revision.
    Revisions before 0.4 published segmentation masks at full resolution
    under a `test.256` name and cannot be scored; see
    `docs/evaluation.md`.

    Args:
        task: The name of the task.

    Returns:
        The path of the CSV in the local Hugging Face cache.
    """
    spec = get_task(task)

    return Path(
        hf_hub_download(
            spec.dataset,
            GROUND_TRUTH_FILES[task],
            repo_type="dataset",
            revision=spec.revision,
        )
    )


def load_ground_truth(task: str) -> pd.DataFrame:
    """Read the ground-truth CSV published with a task's dataset.

    Args:
        task: The name of the task.

    Returns:
        The ground truth, with `encoded_pixels` kept as a string so that
        an empty mask does not become NaN.
    """
    return pd.read_csv(ground_truth_path(task), converters={"encoded_pixels": str})


def row_column(task: str) -> str:
    """Name the ground-truth column identifying a task's scored rows.

    Args:
        task: The name of the task.

    Returns:
        `sequence_id` for a sequence-level task, `image_id` otherwise.
    """
    return "sequence_id" if get_task(task).sequence_by else "image_id"


def load_patient_map(
    spec: TaskSpec, cache_dir: str | Path | None = None
) -> dict[str, str] | None:
    """Map each image of a test split to its patient, where there is one.

    The split is read with only the two columns needed, and the result
    is cached as a CSV so that scoring a submission twice downloads
    nothing the second time.

    Args:
        spec: The task whose source split to read.
        cache_dir: The directory to cache the map in. Defaults to
            $ULTRABENCH_CACHE, then `~/.ultrabench`.

    Returns:
        A mapping of image ID to patient ID, or None if the dataset
        provides no patient identifier.

    Raises:
        ValueError: If a dataset declared to carry patient identifiers
            has no patient column.
    """
    if spec.dataset_name not in PATIENT_DATASETS:
        return None

    directory = resolve_cache_dir(cache_dir) / "protocol"
    directory.mkdir(parents=True, exist_ok=True)
    subset = spec.subset or "default"
    path = directory / f"patients_{spec.dataset.replace('/', '-')}_{subset}.csv"

    if not path.exists():
        logger.info("Reading the patient column of %s (%s).", spec.dataset, subset)
        split = load_dataset(
            name=spec.dataset_name,
            subset=spec.subset,
            split=f"test.{spec.image_size}",
            revision=spec.revision,
        )
        if PATIENT_COLUMN not in split.column_names:
            raise ValueError(
                f"'{spec.dataset}' is declared to be resampled per patient but "
                f"its test split has no '{PATIENT_COLUMN}' column."
            )
        columns = split.select_columns(["filename", PATIENT_COLUMN])
        pd.DataFrame(
            {
                "image_id": [str(value) for value in columns["filename"]],
                "patient": [str(value) for value in columns[PATIENT_COLUMN]],
            }
        ).to_csv(path, index=False)

    table = pd.read_csv(path, dtype=str)

    return dict(zip(table["image_id"], table["patient"]))


def task_units(
    task: str, rows: np.ndarray, cache_dir: str | Path | None = None
) -> tuple[np.ndarray, str]:
    """Give the resampling unit of each scored row of a task.

    Args:
        task: The name of the task.
        rows: The sorted unique row IDs of the task.
        cache_dir: The directory to cache the patient map in.

    Returns:
        The unit ID of each row in sorted-row order, and whether the
        units are patients or images.

    Raises:
        ValueError: If a sequence spans more than one patient, which
            would leave the sequence unassignable to a unit.
    """
    spec = get_task(task)
    patients = load_patient_map(spec, cache_dir)

    if patients is None:
        return rows.copy(), "image"

    if spec.sequence_by is None:
        return np.array([patients[row] for row in rows]), "patient"

    ground_truth = load_ground_truth(task)
    by_sequence = (
        ground_truth["image_id"]
        .astype(str)
        .map(patients)
        .groupby(ground_truth["sequence_id"].astype(str))
        .agg(set)
    )
    split_sequences = by_sequence[by_sequence.map(len) != 1]
    if len(split_sequences) > 0:
        raise ValueError(
            f"Task '{task}' has {len(split_sequences)} sequence(s) spanning "
            f"more than one patient, starting with '{split_sequences.index[0]}'."
        )

    sequence_patient = by_sequence.map(lambda values: next(iter(values)))

    return np.array([sequence_patient[row] for row in rows]), "patient"


def build_cohorts(tasks: Sequence[str], cache_dir: str | Path | None = None) -> Cohorts:
    """Resolve the rows and resampling units of a set of tasks.

    The unit pool of a source dataset is taken over every benchmark task
    built on that dataset, not only the requested ones, so that scoring
    part of the benchmark draws the same resamples as scoring all of it.

    Args:
        tasks: The tasks to score.
        cache_dir: The directory to cache the patient maps in.

    Returns:
        The cohort of every task needed to score the requested ones,
        and the unit pool of each source dataset.

    Raises:
        ValueError: If a task has rows outside its dataset's unit pool.
    """
    datasets = {get_task(task).dataset for task in tasks}
    pooled = [task for task in BENCHMARK_TASKS if get_task(task).dataset in datasets]

    rows = {task: _task_rows(task) for task in pooled}
    units = {task: task_units(task, rows[task], cache_dir) for task in pooled}

    dataset_units = {
        dataset: np.unique(
            np.concatenate(
                [units[task][0] for task in pooled if get_task(task).dataset == dataset]
            )
        )
        for dataset in datasets
    }

    cohorts = {}
    for task in pooled:
        dataset = get_task(task).dataset
        index = np.searchsorted(dataset_units[dataset], units[task][0])
        if not np.array_equal(dataset_units[dataset][index], units[task][0]):
            raise ValueError(f"Task '{task}' has rows outside its unit pool.")

        cohorts[task] = TaskCohort(
            task=task,
            dataset=dataset,
            rows=rows[task],
            units=units[task][0],
            unit_kind=units[task][1],
            unit_index=index,
        )

    return Cohorts(tasks=cohorts, dataset_units=dataset_units)


def _task_rows(task: str) -> np.ndarray:
    """List the sorted unique scored rows of a task's test split.

    Args:
        task: The name of the task.

    Returns:
        The sorted unique row IDs, as strings.
    """
    return np.unique(load_ground_truth(task)[row_column(task)].astype(str).to_numpy())


def _reindex(values: pd.Series, rows: np.ndarray, task: str) -> np.ndarray:
    """Put a per-row series into sorted-row order.

    Args:
        values: The values, indexed by row ID.
        rows: The sorted unique row IDs of the task.
        task: The name of the task, used in the error message.

    Returns:
        The values in the order of `rows`.

    Raises:
        ValueError: If the series does not cover every row of the task.
    """
    values.index = values.index.astype(str)
    missing = np.setdiff1d(rows, values.index.to_numpy())
    if len(missing) > 0:
        raise ValueError(
            f"Task '{task}' has {len(missing)} row(s) with no scored entries, "
            f"starting with '{missing[0]}'."
        )

    return values.reindex(rows).to_numpy()


def classification_contributions(
    cohort: TaskCohort, predictions_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one classification run to a label pair per scored row.

    Macro F1 is a function of the confusion matrix, so a run is fully
    described by the true and predicted label of each row. Sequence
    pooling reuses the shipped `aggregate_sequence_probabilities`, so it
    cannot drift from what the evaluation layer reports.

    Args:
        cohort: The cohort of the task being scored.
        predictions_path: Path to the run's predictions CSV.

    Returns:
        The ground-truth and predicted labels, in sorted-row order.
    """
    spec = get_task(cohort.task)
    merged = load_ground_truth(cohort.task).merge(
        pd.read_csv(predictions_path), on="image_id", suffixes=("_gt", "_pred")
    )

    if spec.sequence_by is None:
        indexed = merged.set_index("image_id")

        return (
            _reindex(indexed["label_gt"], cohort.rows, cohort.task).astype(np.int64),
            _reindex(indexed["label_pred"], cohort.rows, cohort.task).astype(np.int64),
        )

    probability_columns = sorted(
        (column for column in merged.columns if column.startswith("prob_")),
        key=lambda column: int(column.removeprefix("prob_")),
    )
    sequence_ids, probabilities, sequence_labels = aggregate_sequence_probabilities(
        merged["sequence_id"].astype(str).tolist(),
        merged[probability_columns].to_numpy(),
        merged["label_gt"].tolist(),
    )
    pooled = pd.DataFrame(
        {"label_gt": sequence_labels, "label_pred": probabilities.argmax(axis=1)},
        index=pd.Index(sequence_ids, name="sequence_id"),
    )

    return (
        _reindex(pooled["label_gt"], cohort.rows, cohort.task).astype(np.int64),
        _reindex(pooled["label_pred"], cohort.rows, cohort.task).astype(np.int64),
    )


def segmentation_contributions(
    cohort: TaskCohort, predictions_path: Path
) -> np.ndarray:
    """Reduce one segmentation run to a positive Dice score per image.

    The per-image score is the mean over the foreground classes of the
    image, skipping classes absent from both the ground truth and the
    prediction. That is how `SegmentationEvaluator` builds
    `positive_dice`, so averaging the returned vector reproduces the
    reported metric.

    Args:
        cohort: The cohort of the task being scored.
        predictions_path: Path to the run's predictions CSV.

    Returns:
        The per-image positive Dice scores in sorted-image order. An
        image whose every foreground class is undefined is NaN.
    """
    spec = get_task(cohort.task)
    with tempfile.TemporaryDirectory() as temporary:
        evaluator = SegmentationEvaluator(
            ground_truth_path=ground_truth_path(cohort.task),
            predictions_path=predictions_path,
            output_dir=temporary,
            image_height=spec.image_size,
            image_width=spec.image_size,
        )
        evaluator.evaluate()
        per_image = pd.read_csv(Path(temporary) / "per_image_metrics.csv")

    foreground = per_image[per_image["class_id"] != 0]

    return _reindex(
        foreground.groupby("image_id")["dice"].mean(), cohort.rows, cohort.task
    ).astype(np.float64)


def macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Compute macro F1 from a stack of confusion matrices.

    A class is averaged over only if it appears in the true or the
    predicted labels, which is the label set `sklearn.metrics.f1_score`
    uses.

    Args:
        confusion: Confusion matrices of shape (n, n_classes,
            n_classes), indexed as [replication, true, predicted].

    Returns:
        The macro F1 of each confusion matrix.
    """
    true_positives = np.einsum("bii->bi", confusion)
    # 2*TP + FP + FN reduces to the row plus column sums of the class.
    denominator = confusion.sum(axis=2) + confusion.sum(axis=1)
    f1 = np.divide(
        2.0 * true_positives,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    return f1.sum(axis=1) / (denominator > 0).sum(axis=1)


def _unit_cell_counts(
    cohort: TaskCohort,
    n_units: int,
    n_classes: int,
    labels_gt: np.ndarray,
    labels_pred: np.ndarray,
) -> np.ndarray:
    """Count each unit's (true, predicted) confusion cells for one run.

    Weighting these rows by how many times each unit was drawn and
    summing gives the confusion matrix of a replication.

    Args:
        cohort: The cohort of the task being scored.
        n_units: The number of units in the source dataset's pool.
        n_classes: The number of classes of the task.
        labels_gt: The ground-truth label of each row.
        labels_pred: The predicted label of each row.

    Returns:
        An array of shape (n_units, n_classes ** 2) of counts.
    """
    counts = np.zeros((n_units, n_classes * n_classes), dtype=np.float32)
    np.add.at(counts, (cohort.unit_index, labels_gt * n_classes + labels_pred), 1.0)

    return counts


def _unit_dice_sums(
    cohort: TaskCohort, n_units: int, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum one run's per-image Dice scores within each unit.

    Args:
        cohort: The cohort of the task being scored.
        n_units: The number of units in the source dataset's pool.
        values: The per-image Dice scores, NaN where undefined.

    Returns:
        The per-unit sum of defined Dice scores and the per-unit count
        of defined images.
    """
    defined = np.isfinite(values).astype(np.float64)
    numerator = np.bincount(
        cohort.unit_index, weights=np.nan_to_num(values) * defined, minlength=n_units
    )
    denominator = np.bincount(cohort.unit_index, weights=defined, minlength=n_units)

    return numerator.astype(np.float32), denominator.astype(np.float32)


def task_run_scores(
    cohort: TaskCohort, predictions: Sequence[Path], multiplicity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Score every run of one task, on the test set and on the resamples.

    Args:
        cohort: The cohort of the task being scored.
        predictions: The prediction CSV of each run.
        multiplicity: The source dataset's multiplicity matrix.

    Returns:
        The metric of each run on the test set, and the run-averaged
        metric on each replication.

    Raises:
        ValueError: If two runs disagree on the ground-truth labels,
            which means one of them does not cover the test split.
    """
    spec = get_task(cohort.task)
    n_units = multiplicity.shape[1]
    point_scores = []
    replication_scores = []

    if spec.task_type == "segmentation":
        for path in predictions:
            values = segmentation_contributions(cohort, path)
            numerator, denominator = _unit_dice_sums(cohort, n_units, values)
            point_scores.append(float(np.nanmean(values)))
            replication_scores.append(
                (multiplicity @ numerator) / (multiplicity @ denominator)
            )

        return np.array(point_scores), np.stack(replication_scores).mean(axis=0)

    n_classes = spec.num_classes
    labels: np.ndarray | None = None
    for path in predictions:
        labels_gt, labels_pred = classification_contributions(cohort, path)
        if labels is None:
            labels = labels_gt
        elif not np.array_equal(labels, labels_gt):
            raise ValueError(
                f"Runs of '{cohort.task}' disagree on the ground-truth labels."
            )

        counts = _unit_cell_counts(cohort, n_units, n_classes, labels_gt, labels_pred)
        confusion = counts.sum(axis=0, dtype=np.float64).reshape(
            1, n_classes, n_classes
        )
        point_scores.append(float(macro_f1_from_confusion(confusion)[0]))
        replication_scores.append(
            macro_f1_from_confusion(
                (multiplicity @ counts).reshape(-1, n_classes, n_classes)
            )
        )

    return np.array(point_scores), np.stack(replication_scores).mean(axis=0)


def interval(
    per_run: np.ndarray,
    per_replication: np.ndarray,
    confidence: float = CONFIDENCE_MULTIPLIER,
) -> dict[str, float]:
    """Combine per-task scores into one score and its error bar.

    Both variance terms are formed on the aggregate score rather than
    per task, so the correlations between tasks that share a test set
    survive into the error bar. A single task is the same computation
    with K = 1.

    Args:
        per_run: The per-run score of each task, of shape (K, S).
        per_replication: The run-averaged per-replication score of each
            task, of shape (K, B).
        confidence: The normal quantile the interval spans.

    Returns:
        The mean, its standard error, the interval and the two variance
        terms.

    Raises:
        ValueError: If there are fewer than two runs, which leaves the
            run term undefined.
    """
    k_tasks, n_runs = per_run.shape
    if n_runs < 2:
        raise ValueError(
            f"The run term needs at least two runs per task, got {n_runs}."
        )

    sigma_run = float(per_run.mean(axis=0).std(ddof=1))
    sigma_boot = float(per_replication.mean(axis=0).std(ddof=1))
    standard_error = float(np.sqrt(sigma_run**2 / n_runs + sigma_boot**2))
    mean = float(per_run.mean())

    return {
        "mean": mean,
        "se": standard_error,
        "ci_low": mean - confidence * standard_error,
        "ci_high": mean + confidence * standard_error,
        "sigma_run": sigma_run,
        "sigma_boot": sigma_boot,
        "k_tasks": k_tasks,
        "n_runs": n_runs,
    }


def find_predictions(
    predictions_dir: str | Path,
    tasks: Sequence[str] | None = None,
    runs_per_task: int | None = REQUIRED_RUNS,
) -> dict[str, list[Path]]:
    """Find the prediction CSV of every run of every submitted task.

    A submission is one directory per task, named after the task, each
    holding one CSV per training run. The CSVs are read in sorted order
    and their names are not otherwise interpreted.

    Args:
        predictions_dir: The root of the submission.
        tasks: The tasks to score, or None for every task found.
        runs_per_task: The number of runs each task must have, or None
            to accept whatever is there.

    Returns:
        The prediction CSVs of each task, keyed by task name.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no
            task directory.
        ValueError: If a directory does not name a benchmark task, or a
            task does not have the required number of runs.
    """
    root = Path(predictions_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Submission directory not found: {root}")

    found = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name[0] != "."
    )
    unknown = [name for name in found if name not in BENCHMARK_TASKS]
    if unknown:
        raise ValueError(
            f"{root} holds {len(unknown)} directory(ies) that do not name a "
            f"benchmark task, starting with '{unknown[0]}'."
        )

    selected = found if tasks is None else list(tasks)
    missing = [task for task in selected if task not in found]
    if missing:
        raise ValueError(f"{root} has no predictions for {missing}.")
    if not selected:
        raise FileNotFoundError(f"{root} holds no task directory.")

    predictions = {task: sorted((root / task).glob("*.csv")) for task in selected}

    if runs_per_task is not None:
        wrong = {
            task: len(paths)
            for task, paths in predictions.items()
            if len(paths) != runs_per_task
        }
        if wrong:
            raise ValueError(
                f"The protocol requires {runs_per_task} runs per task. "
                f"Wrong run counts: {wrong}."
            )

    return predictions


def score_submission(
    predictions_dir: str | Path,
    tasks: Sequence[str] | None = None,
    runs_per_task: int | None = REQUIRED_RUNS,
    n_resamples: int = N_RESAMPLES,
    confidence: float = CONFIDENCE_MULTIPLIER,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a submission per task and per category, with error bars.

    A category is reported only when the submission covers every one of
    its tasks, since the category score is the unweighted mean over all
    of them. The tasks of a category that is left incomplete are still
    scored individually.

    Args:
        predictions_dir: The root of the submission, laid out as one
            directory per task.
        tasks: The tasks to score, or None for every task found.
        runs_per_task: The number of runs each task must have, or None
            to accept whatever is there.
        n_resamples: The number of bootstrap replications to draw.
        confidence: The normal quantile the intervals span.
        cache_dir: The directory to cache the patient maps in.

    Returns:
        The per-task scores and the per-category scores.
    """
    predictions = find_predictions(predictions_dir, tasks, runs_per_task)
    selected = sorted(predictions)
    logger.info("Scoring %d task(s) from %s.", len(selected), predictions_dir)

    cohorts = build_cohorts(selected, cache_dir)
    run_scores: dict[str, np.ndarray] = {}
    replication_scores: dict[str, np.ndarray] = {}
    rows = []

    for dataset, units in cohorts.dataset_units.items():
        dataset_tasks = [
            task for task in selected if cohorts.tasks[task].dataset == dataset
        ]
        if not dataset_tasks:
            continue

        multiplicity = multiplicity_matrix(units, dataset, n_resamples)

        for task in dataset_tasks:
            cohort = cohorts.tasks[task]
            logger.info("Scoring '%s' (%d runs).", task, len(predictions[task]))
            scores, resampled = task_run_scores(cohort, predictions[task], multiplicity)
            run_scores[task] = scores
            replication_scores[task] = resampled

            rows.append(
                {
                    "task": task,
                    "category": TASK_CATEGORIES[task],
                    "metric": get_task(task).primary_metric,
                    "n_rows": len(cohort.rows),
                    "unit": cohort.unit_kind,
                    "n_units": len(np.unique(cohort.units)),
                }
                | interval(scores[None, :], resampled[None, :], confidence)
            )

    task_scores = pd.DataFrame(rows).sort_values("task", ignore_index=True)
    task_scores = task_scores.drop(columns="k_tasks")

    return task_scores, _category_scores(run_scores, replication_scores, confidence)


_CATEGORY_COLUMNS = [
    "category",
    "display_name",
    "mean",
    "se",
    "ci_low",
    "ci_high",
    "sigma_run",
    "sigma_boot",
    "k_tasks",
    "n_runs",
]
"""The columns of the per-category table, in reporting order."""


def _category_scores(
    run_scores: dict[str, np.ndarray],
    replication_scores: dict[str, np.ndarray],
    confidence: float,
) -> pd.DataFrame:
    """Aggregate the scored tasks into a score per complete category.

    Args:
        run_scores: The per-run score of each task.
        replication_scores: The run-averaged per-replication score of
            each task.
        confidence: The normal quantile the intervals span.

    Returns:
        One row per category the submission covers completely.
    """
    rows = []
    for name, category in CATEGORIES.items():
        missing = [task for task in category.tasks if task not in run_scores]
        if missing:
            logger.warning(
                "Skipping category '%s': the submission is missing %d of its "
                "%d tasks (%s).",
                name,
                len(missing),
                len(category.tasks),
                ", ".join(missing),
            )
            continue

        per_run = np.stack([run_scores[task] for task in category.tasks])
        per_replication = np.stack(
            [replication_scores[task] for task in category.tasks]
        )
        rows.append(
            {"category": name, "display_name": category.display_name}
            | interval(per_run, per_replication, confidence)
        )

    return pd.DataFrame(rows, columns=_CATEGORY_COLUMNS)


def leaderboard_row(category_scores: pd.DataFrame) -> str | None:
    """Build the leaderboard row of a submission covering every category.

    The README's leaderboard reports the four category scores to three
    decimal places, in a fixed column order, so the row is printed ready
    to paste into the table.

    Args:
        category_scores: The per-category table.

    Returns:
        The Markdown table row, or None if the submission leaves a
        category unscored.
    """
    scores = category_scores.set_index("category")["mean"]
    if not set(CATEGORIES).issubset(scores.index):
        return None

    cells = " | ".join(f"{scores[category]:.3f}" for category in CATEGORIES)

    return f"| <model> | {cells} |"


def _format(table: pd.DataFrame) -> str:
    """Round a table's float columns for printing.

    Args:
        table: The table to format.

    Returns:
        The table as text, without the index.
    """
    return table.round(4).to_string(index=False)


@hydra.main(config_path="conf", config_name="protocol", version_base=None)
def main(cfg: DictConfig) -> None:
    """Score a submission and write the two tables of results.

    Args:
        cfg: Hydra configuration.
    """
    logging.getLogger("ultrabench2").setLevel(logging.INFO)

    task_scores, category_scores = score_submission(
        predictions_dir=cfg.predictions,
        tasks=None if cfg.tasks is None else list(cfg.tasks),
        runs_per_task=cfg.runs_per_task,
        n_resamples=cfg.n_resamples,
        confidence=cfg.confidence,
        cache_dir=cfg.cache_dir,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_scores.to_csv(output_dir / "task_scores.csv", index=False)
    category_scores.to_csv(output_dir / "category_scores.csv", index=False)

    print(f"\nPer-task scores (+/- {cfg.confidence} SE):")
    print(_format(task_scores))
    print(f"\nPer-category scores (+/- {cfg.confidence} SE):")
    print(_format(category_scores))

    row = leaderboard_row(category_scores)
    if row is not None:
        print(f"\nLeaderboard row:\n{row}")

    print(f"\nWritten to {output_dir}")


if __name__ == "__main__":
    main()
