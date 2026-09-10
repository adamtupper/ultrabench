"""The registry of benchmark tasks and the categories they belong to.

The mechanical properties of a task (its type, number of classes, and the
dataset it comes from) are read from the training layer's task configs so
that the two cannot drift apart. Everything the configs cannot know --
which metric a task is scored on, and which category it belongs to -- is
declared here. See `docs/evaluation_protocol.md` for how the per-task
scores are reduced to a score per category.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

TASK_CONFIG_DIR = Path(__file__).resolve().parents[1] / "training" / "config" / "task"
"""Directory holding the training layer's task configs."""

DEFAULT_IMAGE_SIZE = 256
"""Size of the test images, from `data.image_size` in the training config."""

PRIMARY_METRICS = {
    "classification": "f1_macro",
    "segmentation": "positive_dice",
}
"""The metric each type of task is scored on."""

SEQUENCE_PREFIX = "seq_"
"""Prefix marking a metric computed over sequences rather than images."""

EXCLUDED_TASKS = frozenset(
    {
        "butterfly_classification",
        "pocus_classification",
        "stanford_thyroid_segmentation",
    }
)
"""Tasks that have a config but are not part of the reported benchmark."""

PARADIGM_PATTERN = re.compile(
    r"^(?P<head>.*?)_(?P<paradigm>full_finetuning|head_only)$"
)
"""Splits a run's `mode` directory into its head and training paradigm."""

PARADIGMS = ("head_only", "full_finetuning")
"""The training paradigms a task is evaluated under."""


@dataclass(frozen=True)
class TaskSpec:
    """The properties of a benchmark task needed to score it.

    Args:
        name: The task name, matching its training config file.
        task_type: Whether the task is classification or segmentation.
        num_classes: The number of classes, including the background for
            segmentation tasks.
        dataset: The Hugging Face dataset the task comes from.
        subset: The dataset subset, or None if the dataset has only one.
        revision: The pinned dataset revision.
        target_key: The dataset column holding the label or mask.
        sequence_by: The columns that jointly identify the sequence an
            image belongs to, or None for tasks scored per image.
        image_size: The height and width of the test images in pixels.
    """

    name: str
    task_type: Literal["classification", "segmentation"]
    num_classes: int
    dataset: str
    subset: str | None
    revision: str | None
    target_key: str
    sequence_by: tuple[str, ...] | None
    image_size: int = DEFAULT_IMAGE_SIZE

    @property
    def primary_metric(self) -> str:
        """The metric this task is scored on.

        Returns:
            The name of the metric, as reported by the evaluator.
        """
        metric = PRIMARY_METRICS[self.task_type]
        if self.sequence_by is None:
            return metric

        return f"{SEQUENCE_PREFIX}{metric}"

    @property
    def foreground_classes(self) -> tuple[int, ...]:
        """The non-background class IDs of a segmentation task.

        Returns:
            The foreground class IDs, or an empty tuple for a
            classification task.
        """
        if self.task_type != "segmentation":
            return ()

        return tuple(range(1, self.num_classes))


@dataclass(frozen=True)
class Category:
    """A group of tasks whose scores are averaged into one category score.

    Args:
        name: The category's identifier.
        display_name: The label to use for the category in figures.
        tasks: The tasks belonging to the category.
    """

    name: str
    display_name: str
    tasks: tuple[str, ...]


def load_task_specs(config_dir: Path = TASK_CONFIG_DIR) -> dict[str, TaskSpec]:
    """Load the task registry from the training layer's task configs.

    Args:
        config_dir: The directory holding the task config files.

    Returns:
        A dictionary mapping each task name to its specification.

    Raises:
        FileNotFoundError: If the config directory does not exist.
    """
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Task config directory not found: {config_dir}")

    specs = {}
    for path in sorted(config_dir.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        sequence_by = config.get("sequence_by")

        specs[path.stem] = TaskSpec(
            name=path.stem,
            task_type=config["type"],
            num_classes=config["num_classes"],
            dataset=config["dataset"],
            subset=config.get("subset"),
            revision=config.get("revision"),
            target_key=config["target"],
            sequence_by=tuple(sequence_by) if sequence_by else None,
        )

    return specs


TASKS: dict[str, TaskSpec] = load_task_specs()
"""Every task with a training config, including the excluded ones."""


def get_task(name: str) -> TaskSpec:
    """Look up a task by name.

    Args:
        name: The name of the task.

    Returns:
        The task's specification.

    Raises:
        KeyError: If no task with that name is registered.
    """
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(
            f"Unknown task '{name}'. Known tasks: {', '.join(sorted(TASKS))}."
        ) from None


def included_tasks() -> tuple[str, ...]:
    """List the tasks that are part of the reported benchmark.

    Returns:
        The names of every registered task that is not excluded.
    """
    return tuple(sorted(set(TASKS) - EXCLUDED_TASKS))


CATEGORIES: dict[str, Category] = {
    category.name: category
    for category in (
        Category(
            name="pathology_classification",
            display_name="Pathological (classification)",
            tasks=(
                "aul_mass_classification",
                "busbra_histology_classification",
                "busbra_pathology_classification",
                "covid_blues_classification",
                "gbcu_classification",
                "masld_classification",
                "mmotu_classification",
                "tn3k_classification",
            ),
        ),
        Category(
            name="anatomy_classification",
            display_name="Non-pathological (classification)",
            tasks=(
                "camus_classification",
                "fetal_planes_brain_plane_classification",
                "fetal_planes_plane_classification",
            ),
        ),
        Category(
            name="pathology_segmentation",
            display_name="Pathological (segmentation)",
            tasks=(
                "aul_mass_segmentation",
                "busbra_segmentation",
                "mmotu_segmentation",
                "tn3k_segmentation",
            ),
        ),
        Category(
            name="anatomy_segmentation",
            display_name="Anatomical (segmentation)",
            tasks=(
                "aul_liver_segmentation",
                "camus_segmentation",
                "open_kidney_capsule_segmentation",
                "open_kidney_regions_segmentation",
                "psfhs_segmentation",
                "stmus_segmentation",
            ),
        ),
    )
}
"""The categories that tasks are aggregated over."""


def split_mode(mode: str) -> tuple[str, str]:
    """Split a run's `mode` directory into its head and paradigm.

    Args:
        mode: The mode directory name, for example
            `upsampler_full_finetuning`.

    Returns:
        A tuple of the head name and the training paradigm.

    Raises:
        ValueError: If the mode does not end with a known paradigm.
    """
    match = PARADIGM_PATTERN.match(mode)
    if match is None:
        raise ValueError(
            f"Cannot parse mode '{mode}'. Expected it to end with one of: "
            f"{', '.join(PARADIGMS)}."
        )

    return match.group("head"), match.group("paradigm")


def resolve_tasks(tasks: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve a task selection to a validated tuple of task names.

    Args:
        tasks: The requested task names, or None for every included task.

    Returns:
        The resolved task names.

    Raises:
        KeyError: If any requested task is not registered.
    """
    if tasks is None:
        return included_tasks()

    for name in tasks:
        get_task(name)

    return tuple(tasks)
