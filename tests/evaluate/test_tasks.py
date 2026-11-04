"""Unit tests for the task and category registry."""

import pytest

from ultrabench.evaluate.tasks import (
    CATEGORIES,
    TASKS,
    get_task,
    included_tasks,
    resolve_tasks,
    split_mode,
)

SEQUENCE_TASKS = {
    "camus_classification",
    "covid_blues_classification",
    "masld_classification",
}

# Transcribed from the PATHO_CLS_TASKS / NONPATHO_CLS_TASKS /
# PATHO_SEG_TASKS / ANATOMICAL_SEG_TASKS lists that
# create_results_tables.ipynb and plot_tasks.ipynb each define, so that
# the categories reported by the notebooks and the ones declared here
# cannot drift apart. See docs/evaluation_protocol.md.
NOTEBOOK_CATEGORIES = {
    "pathology_segmentation": [
        "aul_mass_segmentation",
        "busbra_segmentation",
        "mmotu_segmentation",
        "tn3k_segmentation",
    ],
    "anatomy_segmentation": [
        "aul_liver_segmentation",
        "camus_segmentation",
        "open_kidney_capsule_segmentation",
        "open_kidney_regions_segmentation",
        "psfhs_segmentation",
        "stmus_segmentation",
    ],
    "pathology_classification": [
        "aul_mass_classification",
        "busbra_histology_classification",
        "busbra_pathology_classification",
        "covid_blues_classification",
        "gbcu_classification",
        "masld_classification",
        "mmotu_classification",
        "tn3k_classification",
    ],
    "anatomy_classification": [
        "camus_classification",
        "fetal_planes_brain_plane_classification",
        "fetal_planes_plane_classification",
    ],
}


def test_registry_covers_every_task_config():
    """Check that every training task config is registered."""
    assert len(TASKS) == 21
    assert all(spec.name == name for name, spec in TASKS.items())


def test_included_tasks_covers_every_registered_task():
    """Check that every registered task is part of the reported benchmark."""
    assert set(included_tasks()) == set(TASKS)


def test_categories_match_the_notebook_lists():
    """Check that the centralised categories match the notebook lists."""
    assert set(CATEGORIES) == set(NOTEBOOK_CATEGORIES)
    for name, tasks in NOTEBOOK_CATEGORIES.items():
        assert list(CATEGORIES[name].tasks) == tasks


def test_category_tasks_are_all_included():
    """Check that no category references an unknown or excluded task."""
    for category in CATEGORIES.values():
        for name in category.tasks:
            assert name in included_tasks()


def test_every_included_task_belongs_to_exactly_one_category():
    """Check that the categories partition the reported benchmark."""
    for name in included_tasks():
        holders = [
            category.name for category in CATEGORIES.values() if name in category.tasks
        ]
        assert len(holders) == 1


def test_sequence_tasks_use_the_sequence_metric():
    """Check that only the sequence tasks are scored per sequence.

    Scoring a sequence task per frame would weight patients by how many
    frames they contribute.
    """
    for name, spec in TASKS.items():
        if name in SEQUENCE_TASKS:
            assert spec.sequence_by is not None
            assert spec.primary_metric.startswith("seq_")
        else:
            assert spec.sequence_by is None
            assert not spec.primary_metric.startswith("seq_")


def test_primary_metric_matches_the_task_type():
    """Check that each task type contributes the expected metric."""
    assert get_task("tn3k_classification").primary_metric == "f1_macro"
    assert get_task("tn3k_segmentation").primary_metric == "positive_dice"
    assert get_task("camus_classification").primary_metric == "seq_f1_macro"


def test_foreground_classes_exclude_the_background():
    """Check that the background class is not treated as foreground."""
    assert get_task("psfhs_segmentation").foreground_classes == (1, 2)
    assert get_task("busbra_segmentation").foreground_classes == (1,)
    assert get_task("tn3k_classification").foreground_classes == ()


def test_unknown_names_raise():
    """Check that a mistyped task fails loudly."""
    with pytest.raises(KeyError, match="Unknown task 'nope'"):
        get_task("nope")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("upsampler_head_only", ("upsampler", "head_only")),
        ("upsampler_full_finetuning", ("upsampler", "full_finetuning")),
        ("linear_classifier_head_only", ("linear_classifier", "head_only")),
        ("setr_mla_full_finetuning", ("setr_mla", "full_finetuning")),
    ],
)
def test_split_mode(mode, expected):
    """Check that a run's mode splits into its head and paradigm."""
    assert split_mode(mode) == expected


def test_split_mode_rejects_an_unknown_paradigm():
    """Check that an unrecognised mode is rejected rather than guessed."""
    with pytest.raises(ValueError, match="Cannot parse mode 'upsampler_linear_probe'"):
        split_mode("upsampler_linear_probe")


def test_resolve_tasks():
    """Check that a task selection is validated against the registry."""
    assert resolve_tasks(None) == included_tasks()
    assert resolve_tasks(["tn3k_segmentation"]) == ("tn3k_segmentation",)

    with pytest.raises(KeyError, match="Unknown task 'nope'"):
        resolve_tasks(["nope"])
