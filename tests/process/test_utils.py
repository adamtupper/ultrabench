"""Unit tests for ultrabench2.process.utils."""

import httpx
import numpy as np
import pandas as pd
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from ultrabench2.core.rle import mask_to_rle, rle_to_mask
from ultrabench2.evaluate.segmentation import SegmentationEvaluator
from ultrabench2.process.utils import (
    cap_train_split_by_patient,
    generate_classification_ground_truth_from_records,
    generate_segmentation_ground_truth_from_records,
    ground_truth_commit_operations,
    ground_truth_path_in_repo,
    resize_and_pad,
    subsample_balanced_by_class,
    subsample_by_group,
    to_3_channel,
    verify_revision_unpublished,
)


def make_patient_frame(
    n_patients: int, per_patient: int, prefix: str = "train"
) -> pd.DataFrame:
    """Build a metadata frame with a fixed number of examples per patient.

    Args:
        n_patients: The number of distinct patients.
        per_patient: The number of examples per patient.
        prefix: The prefix for the patient IDs, so that frames standing in
            for different splits hold different patients.

    Returns:
        A DataFrame with ``patient`` and ``filename`` columns.
    """
    patients = np.repeat([f"{prefix}{i:03d}" for i in range(n_patients)], per_patient)

    return pd.DataFrame(
        {
            "patient": patients,
            "filename": [f"{prefix}_f{i}" for i in range(len(patients))],
        }
    )


class TestTo3Channel:
    """Tests for to_3_channel."""

    def test_2d_input(self):
        """A 2D (H, W) array is stacked into (H, W, 3)."""
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], image)
        np.testing.assert_array_equal(result[:, :, 1], image)
        np.testing.assert_array_equal(result[:, :, 2], image)

    def test_hwx1_input(self):
        """An (H, W, 1) array is concatenated into (H, W, 3)."""
        image = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], image[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], image[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 2], image[:, :, 0])

    def test_hwx3_input_passthrough(self):
        """An (H, W, 3) array is returned unchanged."""
        image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result is image

    def test_hwx4_input(self):
        """The alpha channel of a (H, W, 4) array is dropped."""
        image = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], image[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], image[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], image[:, :, 2])

    def test_4_channel_raises_value_error(self):
        """An (H, W, 4) array raises ValueError."""
        image = np.zeros((3, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            to_3_channel(image)

    def test_dtype_preserved_float32(self):
        """Output dtype matches input dtype for float32."""
        image = np.ones((4, 4), dtype=np.float32)
        result = to_3_channel(image)
        assert result.dtype == np.float32

    def test_dtype_preserved_uint16(self):
        """Output dtype matches input dtype for uint16."""
        image = np.ones((4, 4), dtype=np.uint16)
        result = to_3_channel(image)
        assert result.dtype == np.uint16


class TestResizeAndPad:
    """Tests for resize_and_pad."""

    def test_square_image_exact_size(self):
        """A square image resized to the same size is unchanged."""
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)

    def test_tall_image_3channel(self):
        """A tall (H > W) 3-channel image fits within the target square."""
        image = np.ones((400, 200, 3), dtype=np.uint8) * 100
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)

    def test_wide_image_3channel(self):
        """A wide (W > H) 3-channel image fits within the target square."""
        image = np.ones((200, 400, 3), dtype=np.uint8) * 100
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)

    def test_2d_mask(self):
        """A 2D mask is padded to the exact target size."""
        mask = np.ones((300, 200), dtype=np.uint8)
        result = resize_and_pad(mask, 256, order=0)
        assert result.shape == (256, 256)

    def test_tall_image_centering(self):
        """Check left and right pads are equal for tall images."""
        # 100×50 → scale = 256/100 = 2.56 → new_h=256, new_w=128
        # left pad = (256 - 128) // 2 = 64
        image = np.ones((100, 50, 3), dtype=np.uint8) * 200
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)
        # Padding columns should be zero
        np.testing.assert_array_equal(result[:, :64, :], 0)
        np.testing.assert_array_equal(result[:, 192:, :], 0)

    def test_wide_image_centering(self):
        """Check top and bottom pads are equal for wide images."""
        # 50×100 → scale = 256/100 = 2.56 → new_h=128, new_w=256
        # top pad = (256 - 128) // 2 = 64
        image = np.ones((50, 100, 3), dtype=np.uint8) * 200
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)
        np.testing.assert_array_equal(result[:64, :, :], 0)
        np.testing.assert_array_equal(result[192:, :, :], 0)

    def test_dtype_preserved_uint8(self):
        """Output dtype matches uint8 input."""
        image = np.ones((300, 200, 3), dtype=np.uint8)
        result = resize_and_pad(image, 256, order=1)
        assert result.dtype == np.uint8

    def test_mask_order0_binary_values(self):
        """Nearest-neighbor mask resize preserves binary {0, 1} values."""
        mask = np.zeros((300, 200), dtype=np.uint8)
        mask[50:250, 50:150] = 1
        result = resize_and_pad(mask, 256, order=0)
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 1})


class TestCapTrainSplitByPatient:
    """Tests for cap_train_split_by_patient."""

    def test_under_cap_is_unchanged(self):
        """A training split at or under the cap is returned unchanged."""
        train_df = make_patient_frame(10, 5)
        test_df = make_patient_frame(2, 5, prefix="test")
        capped_train, expanded_test = cap_train_split_by_patient(train_df, test_df, 50)
        pd.testing.assert_frame_equal(capped_train, train_df)
        pd.testing.assert_frame_equal(expanded_test, test_df)

    def test_cap_is_respected(self):
        """The capped training split never exceeds the cap."""
        train_df = make_patient_frame(100, 7)
        test_df = make_patient_frame(10, 7, prefix="test")
        capped_train, _ = cap_train_split_by_patient(train_df, test_df, 100)
        assert len(capped_train) <= 100
        assert len(capped_train) > 100 - 7

    def test_no_patient_overlap(self):
        """No patient appears in both the training and test splits."""
        train_df = make_patient_frame(100, 3)
        test_df = make_patient_frame(10, 3, prefix="test")
        capped_train, expanded_test = cap_train_split_by_patient(train_df, test_df, 100)
        train_patients = set(capped_train["patient"])
        assert not train_patients & set(expanded_test["patient"])

    def test_overflow_moved_to_test(self):
        """Overflow examples are appended after the original test examples."""
        train_df = make_patient_frame(50, 4)
        test_df = make_patient_frame(5, 4, prefix="test")
        original_test = test_df.copy()
        capped_train, expanded_test = cap_train_split_by_patient(train_df, test_df, 100)
        assert len(capped_train) + len(expanded_test) == len(train_df) + len(test_df)
        pd.testing.assert_frame_equal(
            expanded_test.head(len(original_test)), original_test
        )

    def test_deterministic(self):
        """The same seed selects the same patients."""
        train_df = make_patient_frame(60, 5)
        test_df = make_patient_frame(5, 5, prefix="test")
        first, _ = cap_train_split_by_patient(train_df, test_df, 100)
        second, _ = cap_train_split_by_patient(train_df, test_df, 100)
        pd.testing.assert_frame_equal(first, second)


class TestSubsampleBalancedByClass:
    """Tests for subsample_balanced_by_class."""

    def test_under_cap_is_unchanged(self):
        """A frame at or under the cap is returned unchanged."""
        df = pd.DataFrame({"label": [0] * 30 + [1] * 20})
        result = subsample_balanced_by_class(df, 100, label_col="label")
        pd.testing.assert_frame_equal(result, df)

    def test_removals_come_from_the_largest_class(self):
        """Only the over-represented class shrinks."""
        df = pd.DataFrame({"label": [0] * 800 + [1] * 400})
        result = subsample_balanced_by_class(df, 1000, label_col="label")
        assert len(result) == 1000
        assert result["label"].value_counts().to_dict() == {0: 600, 1: 400}

    def test_skewed_input_is_balanced(self):
        """A heavily skewed input is balanced as far as the cap allows."""
        df = pd.DataFrame({"label": [0] * 2000 + [1] * 879})
        result = subsample_balanced_by_class(df, 1000, label_col="label")
        assert result["label"].value_counts().to_dict() == {0: 500, 1: 500}

    def test_small_classes_kept_whole(self):
        """Classes smaller than the per-class quota are kept whole."""
        df = pd.DataFrame({"label": [0] * 5 + [1] * 10 + [2] * 1000})
        result = subsample_balanced_by_class(df, 100, label_col="label")
        counts = result["label"].value_counts().to_dict()
        assert len(result) == 100
        assert counts[0] == 5
        assert counts[1] == 10
        assert counts[2] == 85

    def test_uniform_input_loses_equal_shares(self):
        """An already uniform input loses an equal share from each class."""
        df = pd.DataFrame({"label": [0] * 600 + [1] * 600})
        result = subsample_balanced_by_class(df, 1000, label_col="label")
        assert result["label"].value_counts().to_dict() == {0: 500, 1: 500}

    def test_input_order_preserved(self):
        """The subsample keeps the row order of the input."""
        df = pd.DataFrame({"label": [0] * 800 + [1] * 400, "position": range(1200)})
        result = subsample_balanced_by_class(df, 1000, label_col="label")
        assert result["position"].is_monotonic_increasing

    def test_deterministic(self):
        """The same seed selects the same examples."""
        df = pd.DataFrame({"label": [0] * 800 + [1] * 400, "position": range(1200)})
        first = subsample_balanced_by_class(df, 1000, label_col="label")
        second = subsample_balanced_by_class(df, 1000, label_col="label")
        pd.testing.assert_frame_equal(first, second)


class TestSubsampleByGroup:
    """Tests for subsample_by_group."""

    @staticmethod
    def make_sequence_frame(n_patients: int, n_frames: int) -> pd.DataFrame:
        """Build a frame-level metadata frame with two views per patient.

        Args:
            n_patients: The number of distinct patients.
            n_frames: The number of frames per (patient, view) sequence.

        Returns:
            A DataFrame with ``patient``, ``view`` and ``frame`` columns.
        """
        rows = [
            {"patient": patient, "view": view, "frame": frame}
            for patient in range(n_patients)
            for view in ["2CH", "4CH"]
            for frame in range(n_frames)
        ]

        return pd.DataFrame(rows)

    def test_under_cap_is_unchanged(self):
        """A frame at or under the cap is returned unchanged."""
        df = self.make_sequence_frame(10, 5)
        result = subsample_by_group(df, 1000, group_cols=["patient", "view"])
        pd.testing.assert_frame_equal(result, df)

    def test_cap_is_met_exactly(self):
        """The subsample holds exactly the requested number of examples."""
        df = self.make_sequence_frame(450, 34)
        result = subsample_by_group(df, 1000, group_cols=["patient", "view"])
        assert len(result) == 1000

    def test_every_group_is_represented(self):
        """All patients and views survive the subsample."""
        df = self.make_sequence_frame(450, 34)
        result = subsample_by_group(df, 1000, group_cols=["patient", "view"])
        assert result["patient"].nunique() == 450
        assert result["view"].nunique() == 2

    def test_no_group_exceeds_its_quota(self):
        """Groups hold either the base quota or one extra example."""
        df = self.make_sequence_frame(450, 34)
        result = subsample_by_group(df, 1000, group_cols=["patient", "view"])
        sizes = result.groupby(["patient", "view"]).size()
        assert set(sizes.unique()).issubset({1, 2})

    def test_small_groups_kept_whole(self):
        """A group with fewer examples than its quota is kept whole."""
        df = pd.DataFrame({"group": ["a"] * 2 + ["b"] * 100, "position": range(102)})
        result = subsample_by_group(df, 50, group_cols=["group"])
        assert (result["group"] == "a").sum() == 2

    def test_deterministic(self):
        """The same seed selects the same examples."""
        df = self.make_sequence_frame(100, 20)
        first = subsample_by_group(df, 500, group_cols=["patient", "view"])
        second = subsample_by_group(df, 500, group_cols=["patient", "view"])
        pd.testing.assert_frame_equal(first, second)


class TestVerifyRevisionUnpublished:
    """Tests for verify_revision_unpublished."""

    @staticmethod
    def patch_tags(monkeypatch, tags: list[str] | None):
        """Patch the Hub API to report a given set of tags.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
            tags: The tag names to report, or None to simulate a missing
                repository.
        """

        def list_repo_refs(self, repo_id, repo_type=None, **kwargs):
            if tags is None:
                response = httpx.Response(
                    404, request=httpx.Request("GET", f"https://hf.co/{repo_id}")
                )
                raise RepositoryNotFoundError(repo_id, response=response)
            refs = type("GitRefs", (), {})()
            refs.tags = [type("GitRefInfo", (), {"name": tag})() for tag in tags]
            return refs

        monkeypatch.setattr(
            "ultrabench2.process.utils.HfApi.list_repo_refs", list_repo_refs
        )

    def test_raises_when_revision_exists(self, monkeypatch):
        """An already published revision is rejected."""
        self.patch_tags(monkeypatch, ["0.1", "0.2"])
        with pytest.raises(ValueError, match="already been published"):
            verify_revision_unpublished("UltraBench/TN3K", "0.2")

    def test_passes_when_revision_is_new(self, monkeypatch):
        """A revision that has not been published yet is accepted."""
        self.patch_tags(monkeypatch, ["0.1"])
        verify_revision_unpublished("UltraBench/TN3K", "0.2")

    def test_passes_when_repo_is_missing(self, monkeypatch):
        """A repository that does not exist yet is accepted."""
        self.patch_tags(monkeypatch, None)
        verify_revision_unpublished("UltraBench/TN3K", "0.2")


class TestGenerateClassificationGroundTruth:
    """Tests for generate_classification_ground_truth_from_records."""

    records = [
        {"filename": "p1_0", "label": 0, "patient": 1, "view": "2CH"},
        {"filename": "p1_1", "label": 0, "patient": 1, "view": "2CH"},
        {"filename": "p2_0", "label": 1, "patient": 2, "view": "4CH"},
    ]

    def test_image_level_csv_has_no_sequence_column(self, tmp_path):
        """Without sequence keys, only image_id and label are written."""
        output_path = tmp_path / "gt.csv"
        generate_classification_ground_truth_from_records(self.records, output_path)

        written = pd.read_csv(output_path)
        assert list(written.columns) == ["image_id", "label"]
        assert written["image_id"].tolist() == ["p1_0", "p1_1", "p2_0"]

    def test_sequence_keys_add_sequence_id(self, tmp_path):
        """Sequence keys are joined with underscores into sequence_id."""
        output_path = tmp_path / "gt.csv"
        generate_classification_ground_truth_from_records(
            self.records, output_path, sequence_keys=["patient", "view"]
        )

        written = pd.read_csv(output_path)
        assert list(written.columns) == ["image_id", "label", "sequence_id"]
        assert written["sequence_id"].tolist() == ["1_2CH", "1_2CH", "2_4CH"]

    def test_missing_sequence_key_raises(self, tmp_path):
        """A record missing a sequence key is rejected."""
        with pytest.raises(KeyError, match="video"):
            generate_classification_ground_truth_from_records(
                self.records, tmp_path / "gt.csv", sequence_keys=["video"]
            )


def make_mask(size: int, value: int = 1) -> np.ndarray:
    """Build a mask with a square of one class in its top-left corner.

    Args:
        size: The height and width of the mask.
        value: The class ID to fill the square with.

    Returns:
        A uint8 array of shape (size, size).
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[: size // 2, : size // 2] = value

    return mask


class TestGenerateSegmentationGroundTruth:
    """Tests for generate_segmentation_ground_truth_from_records."""

    records = [
        {"filename": "a", "mask": make_mask(8, value=1)},
        {"filename": "b", "mask": make_mask(8, value=2)},
    ]

    def test_csv_columns_and_row_order(self, tmp_path):
        """One row per (image, class), images in order, classes within them."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1, 2], output_path, image_size=8
        )

        written = pd.read_csv(output_path)
        assert list(written.columns) == ["image_id", "class_id", "encoded_pixels"]
        assert written["image_id"].tolist() == ["a", "a", "b", "b"]
        assert written["class_id"].tolist() == [1, 2, 1, 2]

    def test_masks_round_trip(self, tmp_path):
        """Each encoded mask decodes back to that class of the record."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1, 2], output_path, image_size=8
        )

        written = pd.read_csv(output_path, converters={"encoded_pixels": str})
        for row in written.to_dict("records"):
            record = next(r for r in self.records if r["filename"] == row["image_id"])
            expected = np.asarray(record["mask"] == row["class_id"], dtype=np.uint8)
            decoded = rle_to_mask(row["encoded_pixels"], height=8, width=8)
            np.testing.assert_array_equal(decoded, expected)

    def test_empty_class_is_an_empty_string(self, tmp_path):
        """A class absent from an image encodes to an empty string."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1], output_path, image_size=8
        )

        written = pd.read_csv(output_path, converters={"encoded_pixels": str})
        assert written.loc[written["image_id"] == "b", "encoded_pixels"].item() == ""

    def test_missing_mask_is_skipped(self, tmp_path):
        """A record without a mask contributes no rows."""
        output_path = tmp_path / "gt.csv"
        records = [*self.records, {"filename": "c", "mask": None}]
        generate_segmentation_ground_truth_from_records(
            records, "mask", [1], output_path, image_size=8
        )

        written = pd.read_csv(output_path)
        assert written["image_id"].tolist() == ["a", "b"]

    def test_wrong_size_mask_raises(self, tmp_path):
        """A mask that is not at the split's resolution is rejected."""
        records = [{"filename": "a", "mask": make_mask(4)}]
        with pytest.raises(ValueError, match=r"expected \(8, 8\)"):
            generate_segmentation_ground_truth_from_records(
                records, "mask", [1], tmp_path / "gt.csv", image_size=8
            )

    def test_non_square_mask_raises(self, tmp_path):
        """A mask that was not padded to a square is rejected."""
        records = [{"filename": "a", "mask": np.zeros((4, 8), dtype=np.uint8)}]
        with pytest.raises(ValueError, match="has shape"):
            generate_segmentation_ground_truth_from_records(
                records, "mask", [1], tmp_path / "gt.csv", image_size=8
            )

    def test_none_image_size_accepts_mixed_shapes(self, tmp_path):
        """The original-resolution split holds masks of differing sizes."""
        output_path = tmp_path / "gt.csv"
        records = [
            {"filename": "a", "mask": make_mask(4)},
            {"filename": "b", "mask": make_mask(6)},
        ]
        generate_segmentation_ground_truth_from_records(
            records, "mask", [1], output_path, image_size=None
        )

        written = pd.read_csv(output_path)
        assert written["image_id"].tolist() == ["a", "b"]

    def test_accepts_an_iterator(self, tmp_path):
        """Records may be generated lazily rather than held in memory."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            (r for r in self.records), "mask", [1], output_path, image_size=8
        )

        written = pd.read_csv(output_path)
        assert written["image_id"].tolist() == ["a", "b"]

    def test_evaluator_scores_the_written_csv(self, tmp_path):
        """The evaluation layer can score what the dataset layer writes.

        The ground truth and the predictions are generated from the same
        masks here, so a perfect score is the only correct answer. It
        fails if the columns, the class IDs or the mask resolution of
        the two layers disagree.
        """
        ground_truth_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1, 2], ground_truth_path, image_size=8
        )

        predictions_path = tmp_path / "predictions.csv"
        pd.DataFrame(
            [
                {
                    "image_id": r["filename"],
                    "class_id": class_id,
                    "encoded_pixels": mask_to_rle(
                        np.asarray(r["mask"] == class_id, dtype=np.uint8)
                    ),
                }
                for r in self.records
                for class_id in (1, 2)
            ]
        ).to_csv(predictions_path, index=False)

        metrics = SegmentationEvaluator(
            ground_truth_path,
            predictions_path,
            tmp_path / "results",
            image_height=8,
            image_width=8,
        ).evaluate()

        assert metrics["positive_dice"] == pytest.approx(1.0)

    def test_evaluator_penalises_a_wrong_prediction(self, tmp_path):
        """A prediction that misses the mask scores below a perfect one."""
        ground_truth_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1], ground_truth_path, image_size=8
        )

        predictions_path = tmp_path / "predictions.csv"
        pd.DataFrame(
            [
                {
                    "image_id": r["filename"],
                    "class_id": 1,
                    "encoded_pixels": mask_to_rle(np.zeros((8, 8), dtype=np.uint8)),
                }
                for r in self.records
            ]
        ).to_csv(predictions_path, index=False)

        metrics = SegmentationEvaluator(
            ground_truth_path,
            predictions_path,
            tmp_path / "results",
            image_height=8,
            image_width=8,
        ).evaluate()

        assert metrics["positive_dice"] < 1.0


class TestGroundTruthPathInRepo:
    """Tests for ground_truth_path_in_repo and the commit operations."""

    def test_original_resolution_split_has_no_size(self):
        """The full-resolution CSV is named after the test split."""
        assert (
            ground_truth_path_in_repo("segmentation")
            == "test-ground-truth-segmentation.csv"
        )

    def test_resized_split_carries_its_size(self):
        """The resized CSV names the split whose resolution it holds."""
        assert (
            ground_truth_path_in_repo("classification", size=256)
            == "test.256-ground-truth-classification.csv"
        )

    def test_subset_is_prefixed(self):
        """A dataset with subsets prefixes the subset the CSV covers."""
        assert (
            ground_truth_path_in_repo("classification", size=256, prefix="fetal")
            == "fetal-test.256-ground-truth-classification.csv"
        )

    def test_every_task_is_published_for_both_splits(self, tmp_path):
        """Each task yields one operation per test split."""
        tasks = {
            "classification": "ground_truth_classification.csv",
            "segmentation": "ground_truth_segmentation.csv",
        }
        for split in ("test", "test.256"):
            (tmp_path / split).mkdir()
            for filename in tasks.values():
                (tmp_path / split / filename).touch()

        operations = ground_truth_commit_operations(tmp_path, tasks, 256)

        assert [op.path_in_repo for op in operations] == [
            "test-ground-truth-classification.csv",
            "test-ground-truth-segmentation.csv",
            "test.256-ground-truth-classification.csv",
            "test.256-ground-truth-segmentation.csv",
        ]
        assert [op.path_or_fileobj for op in operations] == [
            str(tmp_path / "test" / "ground_truth_classification.csv"),
            str(tmp_path / "test" / "ground_truth_segmentation.csv"),
            str(tmp_path / "test.256" / "ground_truth_classification.csv"),
            str(tmp_path / "test.256" / "ground_truth_segmentation.csv"),
        ]
