"""Tests that the training and dataset layers agree on sequence IDs.

The evaluation layer joins per-image predictions against the
`sequence_id` column of the ground-truth CSV, which the dataset layer
writes. The training layer builds the same IDs from the Hugging Face
dataset to report sequence-level test metrics. If the two disagree, the
join silently produces the wrong sequences, so they are checked against
each other here.
"""

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from ultrabench.datasets.builders.utils import (
    generate_classification_ground_truth_from_records,
)
from ultrabench.training.data.utils import add_sequence_id

CAMUS_RECORDS = [
    {"filename": "patient0313_2CH_half_sequence_1", "patient": 313, "view": "2CH"},
    {"filename": "patient0313_2CH_half_sequence_2", "patient": 313, "view": "2CH"},
    {"filename": "patient0313_4CH_half_sequence_1", "patient": 313, "view": "4CH"},
    {"filename": "patient0007_2CH_half_sequence_1", "patient": 7, "view": "2CH"},
]
"""Records shaped like the ones CAMUS processing builds, where the
patient ID is an integer that the published dataset stores as a string.
"""


def make_dataset(records: list[dict], patient_type: str) -> Dataset:
    """Build a dataset with the identifier columns of a real one.

    Args:
        records: Record dicts holding "filename", "patient", and "view".
        patient_type: The Hugging Face value type of the patient column,
            "string" for CAMUS or "int32" for COVID-BLUES and MASLD.

    Returns:
        A dataset with an image column, so that the column restriction
        applied by the training layer is exercised realistically.
    """
    image = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    features = Features(
        {
            "filename": Value("string"),
            "patient": Value(patient_type),
            "view": Value("string"),
            "image": Image(),
        }
    )

    return Dataset.from_list(
        [{**r, "image": image} for r in records], features=features
    )


@pytest.mark.parametrize(
    ("patient_type", "sequence_by"),
    [
        ("string", ["patient", "view"]),
        ("int32", ["patient"]),
    ],
)
def test_training_sequence_ids_match_ground_truth(tmp_path, patient_type, sequence_by):
    """The IDs in the ground-truth CSV match those built at training time.

    CAMUS stores the patient ID as a string while COVID-BLUES and MASLD
    store it as an integer, so both are covered.
    """
    records = [{**r, "label": 0} for r in CAMUS_RECORDS]
    output_path = tmp_path / "ground_truth_classification.csv"
    generate_classification_ground_truth_from_records(
        records, output_path, sequence_keys=sequence_by
    )
    ground_truth = pd.read_csv(output_path, dtype=str)

    data = add_sequence_id(make_dataset(CAMUS_RECORDS, patient_type), sequence_by)

    assert data["sequence_id"] == ground_truth["sequence_id"].tolist()


def test_missing_column_raises():
    """A dataset without the grouping columns is rejected."""
    data = make_dataset(CAMUS_RECORDS, "int32")
    with pytest.raises(ValueError, match="no column\\(s\\) \\['video'\\]"):
        add_sequence_id(data, ["video"])
