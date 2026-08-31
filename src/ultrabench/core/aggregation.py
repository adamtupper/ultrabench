"""Sequence-level aggregation of image-level classification predictions.

Some datasets label whole sequences rather than individual images. CAMUS
grades the image quality of a (patient, view) half-sequence, COVID-BLUES
records a per-patient PCR result, and MASLD records a per-patient
diagnosis. Models still predict per image, so the per-image predictions
must be pooled before a sequence-level metric can be computed.

Both the training layer and the evaluation layer pool predictions the
same way: average the predicted probabilities over every image in the
sequence, then take the argmax. Keeping that rule here means the two
layers cannot drift apart.

Examples:
    Pool two frames of one sequence and one frame of another::

        sequence_ids = ["a", "a", "b"]
        probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
        ids, probs, labels = aggregate_sequence_probabilities(
            sequence_ids, probabilities, labels=[0, 0, 1]
        )
        # ids == ["a", "b"], probs == [[0.6, 0.4], [0.1, 0.9]]
        # probs.argmax(axis=1) == [0, 1]
"""

from collections.abc import Iterable, Sequence

import numpy as np


def join_sequence_values(values: Iterable[object]) -> str:
    """Join grouping column values into a sequence identifier.

    The dataset layer writes sequence IDs into the ground-truth CSV and
    the training layer builds them from the Hugging Face dataset. Both go
    through this function so that the two can never disagree.

    Args:
        values: The grouping column values for one image, in a fixed
            column order.

    Returns:
        The values converted to strings and joined with underscores.
    """
    return "_".join(str(value) for value in values)


def format_sequence_id(record: dict, sequence_keys: Sequence[str]) -> str:
    """Build a sequence identifier from a record's grouping columns.

    Args:
        record: A mapping holding at least every key in sequence_keys.
        sequence_keys: The columns that jointly identify a sequence, for
            example ("patient", "view").

    Returns:
        The values of sequence_keys joined with underscores.

    Raises:
        KeyError: If the record is missing one of the sequence keys.
    """
    missing = [key for key in sequence_keys if key not in record]
    if missing:
        raise KeyError(f"Record is missing sequence key(s): {missing}")

    return join_sequence_values(record[key] for key in sequence_keys)


def aggregate_sequence_probabilities(
    sequence_ids: Sequence[str],
    probabilities: np.ndarray,
    labels: Sequence[int] | None = None,
) -> tuple[list[str], np.ndarray, list[int] | None]:
    """Pool per-image probabilities into one prediction per sequence.

    Average the predicted probabilities over all images sharing a
    sequence identifier. Sequences are returned in order of first
    appearance so that the output is deterministic.

    Args:
        sequence_ids: The sequence identifier of each image.
        probabilities: Predicted probabilities of shape
            (num_images, num_classes).
        labels: Optional ground-truth label of each image. Every image in
            a sequence must carry the same label, since the label is a
            property of the sequence.

    Returns:
        A tuple of the unique sequence identifiers, the mean
        probabilities of shape (num_sequences, num_classes), and the
        per-sequence labels (None if labels was not given).

    Raises:
        ValueError: If the inputs have mismatched lengths, if
            probabilities is not 2D, or if a sequence holds more than one
            distinct label.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected probabilities of shape (num_images, num_classes), got "
            f"shape {probabilities.shape}."
        )
    if len(sequence_ids) != probabilities.shape[0]:
        raise ValueError(
            f"Got {len(sequence_ids)} sequence IDs but "
            f"{probabilities.shape[0]} probability rows."
        )
    if labels is not None and len(labels) != len(sequence_ids):
        raise ValueError(
            f"Got {len(sequence_ids)} sequence IDs but {len(labels)} labels."
        )

    grouped_rows: dict[str, list[int]] = {}
    for index, sequence_id in enumerate(sequence_ids):
        grouped_rows.setdefault(sequence_id, []).append(index)

    unique_ids = list(grouped_rows)
    mean_probabilities = np.stack(
        [probabilities[rows].mean(axis=0) for rows in grouped_rows.values()]
    )

    if labels is None:
        return unique_ids, mean_probabilities, None

    sequence_labels = []
    for sequence_id, rows in grouped_rows.items():
        distinct = {int(labels[row]) for row in rows}
        if len(distinct) > 1:
            raise ValueError(
                f"Sequence '{sequence_id}' has more than one ground-truth "
                f"label: {sorted(distinct)}. Sequence labels must be constant "
                f"across the images of a sequence."
            )
        sequence_labels.append(distinct.pop())

    return unique_ids, mean_probabilities, sequence_labels
