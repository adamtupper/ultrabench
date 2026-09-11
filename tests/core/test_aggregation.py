"""Unit tests for the sequence aggregation utilities."""

import numpy as np
import pytest

from ultrabench2.core.aggregation import (
    aggregate_sequence_probabilities,
    format_sequence_id,
    join_sequence_values,
)


def test_join_sequence_values_casts_to_string():
    """Check that mixed-type grouping values join into one string."""
    assert join_sequence_values([313, "2CH"]) == "313_2CH"


def test_format_sequence_id_uses_key_order():
    """Check that the sequence ID follows the given key order."""
    record = {"view": "4CH", "patient": "patient0313", "frame": 7}
    assert format_sequence_id(record, ["patient", "view"]) == "patient0313_4CH"


def test_format_sequence_id_missing_key_raises():
    """Check that a record missing a sequence key is rejected."""
    with pytest.raises(KeyError, match="view"):
        format_sequence_id({"patient": 1}, ["patient", "view"])


def test_mean_probabilities_per_sequence():
    """Check that probabilities are averaged within each sequence."""
    ids, probs, labels = aggregate_sequence_probabilities(
        ["a", "a", "b"],
        np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]),
        labels=[0, 0, 1],
    )

    assert ids == ["a", "b"]
    np.testing.assert_allclose(probs, [[0.6, 0.4], [0.1, 0.9]])
    assert labels == [0, 1]


def test_order_follows_first_appearance():
    """Check that sequences are returned in order of first appearance."""
    ids, _, _ = aggregate_sequence_probabilities(
        ["b", "a", "b"], np.zeros((3, 2)), labels=[1, 0, 1]
    )
    assert ids == ["b", "a"]


def test_single_image_sequence_is_unchanged():
    """Check that a one-image sequence keeps its probabilities."""
    _, probs, _ = aggregate_sequence_probabilities(["a"], np.array([[0.3, 0.7]]))
    np.testing.assert_allclose(probs, [[0.3, 0.7]])


def test_labels_optional():
    """Check that labels are optional and None is returned without them."""
    _, _, labels = aggregate_sequence_probabilities(["a", "a"], np.zeros((2, 3)))
    assert labels is None


def test_mean_probability_can_outvote_the_majority():
    """Check that averaging probabilities, not votes, decides the class.

    Two frames weakly favour class 0 and one frame strongly favours
    class 1, so a majority vote would pick class 0 but the mean
    probability picks class 1.
    """
    _, probs, _ = aggregate_sequence_probabilities(
        ["a", "a", "a"],
        np.array([[0.55, 0.45], [0.55, 0.45], [0.0, 1.0]]),
    )
    assert probs.argmax(axis=1).tolist() == [1]


def test_conflicting_labels_raise():
    """Check that a sequence with two distinct labels is rejected."""
    with pytest.raises(ValueError, match="Sequence 'a' has more than one"):
        aggregate_sequence_probabilities(["a", "a"], np.zeros((2, 2)), labels=[0, 1])


def test_mismatched_lengths_raise():
    """Check that mismatched inputs are rejected."""
    with pytest.raises(ValueError, match="probability rows"):
        aggregate_sequence_probabilities(["a", "b"], np.zeros((3, 2)))

    with pytest.raises(ValueError, match="labels"):
        aggregate_sequence_probabilities(["a", "b"], np.zeros((2, 2)), labels=[0])


def test_non_2d_probabilities_raise():
    """Check that 1D probabilities are rejected."""
    with pytest.raises(ValueError, match="num_images, num_classes"):
        aggregate_sequence_probabilities(["a"], np.array([0.3, 0.7]))
