"""Shared fixtures for the dataset loader tests."""

import csv

import numpy as np
import pytest
import skimage.io as io


IMAGE_SHAPE = (220, 300)
"""The height and width of a fixture image.

Deliberately not square, so that resizing to a square test split is
visible, and large enough that the bright block survives the ten
erosions and 5,000 pixel floor the scan-mask helpers apply.
"""


def write_image(path, seed: int, shape: tuple[int, int] = IMAGE_SHAPE) -> None:
    """Write a synthetic ultrasound-like image.

    The image holds a bright off-centre block on a dark ground, so a
    wrong resize or a transposed axis changes the result visibly.

    Args:
        path: The file to write.
        seed: Varies the block position between images.
        shape: The height and width of the image.
    """
    height, width = shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    top = 30 + (seed % 3)
    image[top : top + 150, 40 : 40 + 200] = 200
    image[top + 20 : top + 80, 70:170] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(path), image, check_contrast=False)


def write_manifests(directory, name: str, train: list[str], test: list[str]) -> None:
    """Write a dataset's train and test manifests.

    Args:
        directory: The manifest root the loaders read from.
        name: The registry name of the dataset.
        train: The names belonging to the train split.
        test: The names belonging to the test split.
    """
    target = directory / name
    target.mkdir(parents=True, exist_ok=True)
    for split, names in (("train", train), ("test", test)):
        with (target / f"{split}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename"])
            writer.writerows([entry] for entry in names)


@pytest.fixture
def gbcu_raw(tmp_path, monkeypatch):
    """Build a fake GBCU raw tree matching its split manifests.

    Args:
        tmp_path: The pytest temporary directory fixture.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The raw data directory.
    """
    raw = tmp_path / "raw"

    train = [(f"im{i:05d}", i % 3) for i in range(4)]
    test = [(f"im{i:05d}", i % 3) for i in range(4, 6)]
    surplus = [("im00099", 0)]

    for name, label in (*train, *surplus):
        write_image(raw / "imgs" / f"{name}.jpg", seed=label)
    for name, label in test:
        write_image(raw / "imgs" / f"{name}.jpg", seed=label)

    (raw / "train.txt").write_text(
        "".join(f"{name}.jpg,{label}\n" for name, label in (*train, *surplus))
    )
    (raw / "test.txt").write_text(
        "".join(f"{name}.jpg,{label}\n" for name, label in test)
    )

    write_manifests(
        tmp_path / "manifests",
        "GBCU",
        [name for name, _ in train],
        [name for name, _ in test],
    )
    monkeypatch.setattr(
        "ultrabench.datasets.utils.MANIFEST_DIR", tmp_path / "manifests"
    )

    return raw


@pytest.fixture
def open_kidney_raw(tmp_path, monkeypatch):
    """Build a fake Open Kidney raw tree matching its manifests.

    The metadata carries a duplicate `file_id`, so the fixture also
    exercises the de-duplication that runs before partitioning.

    Args:
        tmp_path: The pytest temporary directory fixture.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The raw data directory.
    """
    raw = tmp_path / "raw"
    names = [f"{i}_IM-{i:04d}_anon" for i in range(5)]
    duplicate = "9_IM-0001_anon"

    rows = []
    for i, name in enumerate([*names, duplicate]):
        write_image(raw / "images" / f"{name}.png", seed=i)
        for kind, value in (("capsule", 1), ("regions", 1 + i % 3)):
            mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
            mask[40:190, 50:250] = value
            path = raw / "labels" / "reviewed_masks_1" / kind / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(str(path), mask, check_contrast=False)

        rows.append(
            {
                "filename": f"{name}.png",
                "file_attributes": str(
                    {
                        "Quality": "good",
                        "View": "longitudinal" if i % 2 else "transverse",
                        "Comments": "native kidney",
                    }
                ),
                "region_attributes": "{}",
                "region_shape_attributes": "{}",
            }
        )

    labels = raw / "labels" / "reviewed_labels_1.csv"
    labels.parent.mkdir(parents=True, exist_ok=True)
    with labels.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_manifests(tmp_path / "manifests", "OpenKidney", names[:3], names[3:])
    monkeypatch.setattr(
        "ultrabench.datasets.utils.MANIFEST_DIR", tmp_path / "manifests"
    )

    return raw


@pytest.fixture
def mmotu_raw(tmp_path, monkeypatch):
    """Build a fake MMOTU-2D raw tree matching its manifests.

    The class label files use the two-space separator the real dataset
    ships, so the fixture exercises the parser rather than bypassing it.

    Args:
        tmp_path: The pytest temporary directory fixture.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The raw data directory.
    """
    raw = tmp_path / "raw"
    otu = raw / "OTU_2d"
    splits = {"train": ["1", "2", "10"], "val": ["3", "4"]}

    for prefix, names in splits.items():
        (otu / "images").mkdir(parents=True, exist_ok=True)
        (otu / "annotations").mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names):
            write_image(otu / "images" / f"{name}.JPG", seed=i)
            mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
            mask[40:190, 50:250] = 1
            io.imsave(
                str(otu / "annotations" / f"{name}_binary.PNG"),
                mask,
                check_contrast=False,
            )

        (otu / f"{prefix}.txt").write_text("\n".join(names) + "\n")
        (otu / f"{prefix}_cls.txt").write_text(
            "".join(f"{name}.JPG  {i % 8}\n" for i, name in enumerate(names))
        )

    write_manifests(tmp_path / "manifests", "MMOTU", splits["train"], splits["val"])
    monkeypatch.setattr(
        "ultrabench.datasets.utils.MANIFEST_DIR", tmp_path / "manifests"
    )

    return raw


@pytest.fixture
def covid_blues_raw(tmp_path, monkeypatch):
    """Build a fake COVID-BLUES raw tree matching its manifests.

    Decoding real video is avoided by stubbing frame extraction, so the
    filename parsing, clinical merge and per-video partitioning all run
    against the real code.

    Args:
        tmp_path: The pytest temporary directory fixture.
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The directory holding the `covid_blues` folder.
    """
    raw = tmp_path / "raw"
    inner = raw / "covid_blues"
    (inner / "lus_videos").mkdir(parents=True)

    videos = [(patient, point) for patient in (1, 2, 3) for point in ("L1", "R1")]
    for patient, point in videos:
        (inner / "lus_videos" / f"patient_{patient}_{point}.mp4").write_bytes(b"")

    with (inner / "clinical_variables.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "cov_test"])
        writer.writerows([patient, patient % 2] for patient in (1, 2, 3))

    def fake_extract_frames(video_path):
        """Return three synthetic frames for any video.

        Args:
            video_path: The video that would have been decoded.

        Returns:
            An array of three RGB frames.
        """
        frames = np.zeros((3, *IMAGE_SHAPE, 3), dtype=np.uint8)
        for i in range(3):
            frames[i, 30:180, 40:240] = 180 + i * 20

        return frames

    monkeypatch.setattr(
        "ultrabench.process.covid_blues.extract_frames", fake_extract_frames
    )
    write_manifests(
        tmp_path / "manifests",
        "COVID-BLUES",
        ["patient_1_L1", "patient_1_R1", "patient_2_L1", "patient_2_R1"],
        ["patient_3_L1", "patient_3_R1"],
    )
    monkeypatch.setattr(
        "ultrabench.datasets.utils.MANIFEST_DIR", tmp_path / "manifests"
    )

    return raw
