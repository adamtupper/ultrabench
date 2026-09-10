"""Unit tests for ultrabench.process.stmus."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skimage.io as io
from sklearn.model_selection import GroupShuffleSplit

from datasets import load_from_disk

from ultrabench.core.rle import rle_to_mask
from ultrabench.process.stmus import (
    CHARACTERISTICS_FILENAME,
    MUSCLE_NAMES,
    OUTPUT_NAME,
    assign_patient_ids,
    collect_examples,
    load_characteristics,
    process_stmus,
)
from ultrabench.process.utils import cap_train_split_by_patient

ECHOGENICITIES = ["healthy", "pathological"]


def make_subject(muscle, echogenicity, number, age, sex="M", frames=1):
    """Build a subject entry for the fake raw dataset.

    Args:
        muscle: Muscle code (BB, GM, or TA).
        echogenicity: Echogenicity level (healthy or pathological).
        number: Subject number used in the filenames.
        age: Patient age, which drives the demographics.
        sex: Patient sex.
        frames: Number of images to write for the subject.

    Returns:
        A dict describing one subject entry.
    """
    return {
        "muscle": muscle,
        "echogenicity": echogenicity,
        "number": number,
        "age": age,
        "height": 170.0 + age,
        "weight": 60.0 + age,
        "sex": sex,
        "frames": frames,
    }


def write_characteristics(path, healthy, pathological):
    """Write a spreadsheet mimicking the raw subject characteristics.

    The real files hold two side-by-side blocks of differing length
    followed by mean, standard deviation and total rows, so the fake
    reproduces that shape.

    Args:
        path: Destination path for the spreadsheet.
        healthy: Healthy subject entries.
        pathological: Pathological subject entries.

    Returns:
        None.
    """

    def cells(entry):
        if entry is None:
            return [None] * 5
        return [
            entry["number"],
            entry["age"],
            entry["height"],
            entry["weight"],
            entry["sex"],
        ]

    rows = [
        ["Healthy", *[None] * 5, "Pathological", *[None] * 5],
        [*["Number", "Age", "Length", "Weigth", "Sex"], None] * 2,
    ]
    for index in range(max(len(healthy), len(pathological))):
        left = healthy[index] if index < len(healthy) else None
        right = pathological[index] if index < len(pathological) else None
        rows.append([*cells(left), None, *cells(right), None])

    rows.append([None] * 12)
    rows.append(["mean", 1.0, 1.0, 1.0, 1, "Male", None, 1.0, 1.0, 1.0, 1, "Male"])
    rows.append(["SD", 1.0, 1.0, 1.0, 1, "Female", None, 1.0, 1.0, 1.0, 1, "Female"])
    rows.append([None] * 12)
    rows.append(
        [
            "N =",
            len(healthy),
            "healthy",
            None,
            None,
            None,
            None,
            len(pathological),
            "pathological",
            None,
            None,
            None,
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def write_raw_tree(root, subjects, orphan_images=()):
    """Create a minimal raw STMUS directory tree.

    Args:
        root: Directory to create the dataset in.
        subjects: Subject entries to write.
        orphan_images: (muscle, echogenicity, number) triples to write
            images for without a matching subject entry.

    Returns:
        The dataset root path.
    """
    root = Path(root)
    for muscle in MUSCLE_NAMES:
        blocks = {
            echogenicity: [
                subject
                for subject in subjects
                if subject["muscle"] == muscle
                and subject["echogenicity"] == echogenicity
            ]
            for echogenicity in ECHOGENICITIES
        }
        write_characteristics(
            root / muscle / CHARACTERISTICS_FILENAME,
            blocks["healthy"],
            blocks["pathological"],
        )

    written = [
        (s["muscle"], s["echogenicity"], s["number"], frame + 1)
        for s in subjects
        for frame in range(s["frames"])
    ]
    written += [(m, e, n, 1) for m, e, n in orphan_images]

    for muscle, echogenicity, number, frame in written:
        folder = root / muscle / echogenicity.capitalize()
        name = f"anon_{number}_{frame}.png"
        for kind, value in [("Images", 40), ("Masks", 255)]:
            (folder / kind).mkdir(parents=True, exist_ok=True)
            image = np.full((4, 4), value, dtype=np.uint8)
            io.imsave(folder / kind / name, image, check_contrast=False)

    return root


class TestLoadCharacteristics:
    """Tests for load_characteristics."""

    def test_parses_both_blocks(self, tmp_path):
        """Healthy and pathological blocks are both read."""
        subjects = [
            make_subject("BB", "healthy", 1, 30.0),
            make_subject("BB", "pathological", 2, 40.0),
            make_subject("GM", "healthy", 3, 50.0),
            make_subject("TA", "healthy", 4, 60.0),
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)

        characteristics = load_characteristics(str(root))

        assert len(characteristics) == 4
        assert set(characteristics["echogenicity"]) == {"healthy", "pathological"}
        assert set(characteristics["muscle_code"]) == {"BB", "GM", "TA"}

    def test_drops_summary_and_padding_rows(self, tmp_path):
        """Mean, SD and total rows and block padding are discarded."""
        subjects = [
            make_subject("BB", "healthy", 1, 30.0),
            make_subject("BB", "healthy", 2, 31.0),
            make_subject("BB", "pathological", 3, 40.0),
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)

        characteristics = load_characteristics(str(root))

        assert len(characteristics) == 3
        assert sorted(characteristics["number"]) == [1, 2, 3]

    def test_column_types(self, tmp_path):
        """Numbers, demographics and sex are given usable types."""
        root = write_raw_tree(
            tmp_path / "raw", [make_subject("BB", "healthy", 7, 12.5, sex="F")]
        )

        characteristics = load_characteristics(str(root))
        row = characteristics.iloc[0]

        assert characteristics["number"].dtype == int
        assert row["number"] == 7
        assert row["age"] == pytest.approx(12.5)
        assert row["height"] == pytest.approx(182.5)
        assert row["sex"] == "F"

    def test_missing_spreadsheet_raises(self, tmp_path):
        """A muscle without a spreadsheet is reported."""
        root = write_raw_tree(
            tmp_path / "raw", [make_subject("BB", "healthy", 1, 30.0)]
        )
        (root / "GM" / CHARACTERISTICS_FILENAME).unlink()

        with pytest.raises(FileNotFoundError, match="characteristics"):
            load_characteristics(str(root))


class TestAssignPatientIds:
    """Tests for assign_patient_ids."""

    @staticmethod
    def patient_of(patients, muscle, echogenicity, number):
        """Return the patient ID assigned to one subject entry.

        Args:
            patients: The labelled subject entries.
            muscle: Muscle code.
            echogenicity: Echogenicity level.
            number: Subject number.

        Returns:
            The assigned patient ID.
        """
        match = patients[
            (patients["muscle_code"] == muscle)
            & (patients["echogenicity"] == echogenicity)
            & (patients["number"] == number)
        ]
        return match["patient"].item()

    def test_same_number_in_two_muscles_is_two_patients(self, tmp_path):
        """A reused subject number denotes different patients."""
        subjects = [
            make_subject("BB", "healthy", 100, 30.0),
            make_subject("TA", "healthy", 100, 55.0, sex="F"),
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)

        patients = assign_patient_ids(load_characteristics(str(root)))

        assert patients["patient"].nunique() == 2

    def test_different_numbers_across_muscles_is_one_patient(self, tmp_path):
        """Matching demographics identify one patient across muscles."""
        subjects = [
            make_subject("BB", "healthy", 523, 55.5318),
            make_subject("GM", "healthy", 530, 55.5318),
            make_subject("TA", "healthy", 522, 55.5318),
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)

        patients = assign_patient_ids(load_characteristics(str(root)))

        assert patients["patient"].nunique() == 1

    def test_shared_demographics_are_treated_as_one_patient(self, tmp_path):
        """Subjects that cannot be told apart are grouped together."""
        subjects = [
            make_subject(muscle, "healthy", number, 63.9042)
            for muscle, numbers in [("BB", (380, 387)), ("TA", (380, 387))]
            for number in numbers
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)

        patients = assign_patient_ids(load_characteristics(str(root)))

        assert patients["patient"].nunique() == 1
        assert self.patient_of(patients, "BB", "healthy", 380) == self.patient_of(
            patients, "BB", "healthy", 387
        )

    def test_echogenicity_separates_subjects(self, tmp_path):
        """The same number in each block denotes different patients."""
        subjects = [
            make_subject("BB", "healthy", 5, 30.0),
            make_subject("BB", "pathological", 5, 45.0),
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)

        patients = assign_patient_ids(load_characteristics(str(root)))

        assert patients["patient"].nunique() == 2

    def test_deterministic(self, tmp_path):
        """Repeated calls assign identical patient IDs."""
        subjects = [
            make_subject("BB", "healthy", number, float(number))
            for number in range(1, 6)
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)
        characteristics = load_characteristics(str(root))

        first = assign_patient_ids(characteristics)
        second = assign_patient_ids(characteristics)

        pd.testing.assert_frame_equal(first, second)


class TestCollectExamples:
    """Tests for collect_examples."""

    def test_patients_are_resolved_across_muscles(self, tmp_path):
        """Images are grouped by patient rather than subject number."""
        subjects = [
            make_subject("BB", "healthy", 10, 30.0, frames=2),
            make_subject("TA", "healthy", 9, 30.0, frames=3),
            make_subject("TA", "healthy", 10, 70.0, sex="F", frames=1),
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)

        examples = collect_examples(str(root))

        def patient(filename):
            return examples.loc[examples["filename"] == filename, "patient"].iloc[0]

        assert len(examples) == 6
        assert examples["patient"].nunique() == 2
        # The reused number 10 is two people, and one person is numbered
        # 10 in BB but 9 in TA. Grouping on the number inverts both.
        assert patient("bb_healthy_anon_10_1") != patient("ta_healthy_anon_10_1")
        assert patient("bb_healthy_anon_10_1") == patient("ta_healthy_anon_9_1")

    def test_filename_format_is_unchanged(self, tmp_path):
        """Filenames keep the muscle and echogenicity prefixes."""
        root = write_raw_tree(
            tmp_path / "raw", [make_subject("BB", "healthy", 1, 30.0)]
        )

        examples = collect_examples(str(root))

        assert examples["filename"].tolist() == ["bb_healthy_anon_1_1"]

    def test_metadata_is_attached(self, tmp_path):
        """Muscle names and demographics accompany each image."""
        root = write_raw_tree(
            tmp_path / "raw", [make_subject("GM", "pathological", 3, 44.0, sex="F")]
        )

        row = collect_examples(str(root)).iloc[0]

        assert row["muscle"] == "gastrocnemius medialis"
        assert row["echogenicity"] == "pathological"
        assert row["age"] == pytest.approx(44.0)
        assert row["sex"] == "F"
        assert Path(row["mask_path"]).is_file()

    def test_unmatched_image_raises(self, tmp_path):
        """An image without a subject entry is reported."""
        root = write_raw_tree(
            tmp_path / "raw",
            [make_subject("BB", "healthy", 1, 30.0)],
            orphan_images=[("BB", "healthy", 999)],
        )

        with pytest.raises(ValueError, match="999"):
            collect_examples(str(root))


class TestSplitByPatient:
    """Tests for splitting the collected examples by patient."""

    def test_no_patient_overlap(self, tmp_path):
        """Patients scanned for several muscles stay in one split."""
        subjects = [
            make_subject(muscle, "healthy", number + offset, float(number), frames=2)
            for number in range(1, 11)
            for muscle, offset in [("BB", 0), ("GM", 100), ("TA", 200)]
        ]
        root = write_raw_tree(tmp_path / "raw", subjects)
        examples = collect_examples(str(root))

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(examples, groups=examples["patient"]))
        train_df = examples.iloc[train_idx].reset_index(drop=True)
        test_df = examples.iloc[test_idx].reset_index(drop=True)
        train_df, test_df = cap_train_split_by_patient(train_df, test_df, 20)

        assert examples["patient"].nunique() == 10
        assert not set(train_df["patient"]) & set(test_df["patient"])


def flatten_masks_to_a_pattern(root):
    """Replace the flat raw masks with an asymmetric pattern.

    A uniform mask survives any resize, so the masks are given a
    pattern that a wrong interpolation or a transposed axis would
    distort.

    Args:
        root: The raw dataset root created by write_raw_tree.
    """
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :3] = 255

    for path in sorted(Path(root).glob("*/*/Masks/*.png")):
        io.imsave(path, mask, check_contrast=False)


class TestProcessStmus:
    """Tests for the dataset and ground truth written by process_stmus."""

    size = 8

    @pytest.fixture
    def dataset(self, tmp_path):
        """Process a fake raw dataset and load the result from disk.

        Args:
            tmp_path: The pytest temporary directory fixture.

        Returns:
            A tuple of the loaded DatasetDict and its output directory.
        """
        subjects = [
            make_subject(muscle, "healthy", number + offset, float(number), frames=2)
            for number in range(1, 11)
            for muscle, offset in [("BB", 0), ("GM", 100), ("TA", 200)]
        ]
        raw_data_dir = tmp_path / "raw"
        root = write_raw_tree(raw_data_dir / "Polito-Radboud-DeepLearningUS", subjects)
        flatten_masks_to_a_pattern(root)

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        process_stmus(str(raw_data_dir), str(output_dir), size=self.size)

        output_path = output_dir / OUTPUT_NAME

        return load_from_disk(str(output_path)), output_path

    @pytest.mark.parametrize("split", ["test", f"test.{size}"])
    def test_ground_truth_matches_its_split(self, dataset, split):
        """Every ground-truth mask is the mask of the split it sits in.

        This is the test that would have caught ground truth being
        encoded at the original resolution while filed under the
        resized split.
        """
        data, output_path = dataset
        written = pd.read_csv(
            output_path / split / "ground_truth_segmentation.csv",
            converters={"encoded_pixels": str},
        )

        assert written["image_id"].tolist() == data[split]["filename"]

        for row, record in zip(written.to_dict("records"), data[split]):
            expected = np.asarray(record["mask"])
            decoded = rle_to_mask(
                row["encoded_pixels"], height=expected.shape[0], width=expected.shape[1]
            )
            np.testing.assert_array_equal(decoded, expected)

    def test_resized_ground_truth_is_square(self, dataset):
        """The resized split's masks are padded to the target size."""
        data, _ = dataset
        for record in data[f"test.{self.size}"]:
            assert np.asarray(record["mask"]).shape == (self.size, self.size)
