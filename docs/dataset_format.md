---
title: Datasets
nav_order: 3
has_children: true
---

# Dataset Format

UltraBench provides a standardized preprocessing pipeline that converts raw ultrasound datasets into a consistent directory structure.

## Loading Datasets

`ultrabench.load_dataset` loads any dataset in the collection through one interface:

```python
import ultrabench

dataset = ultrabench.load_dataset(
    name="AUL",          # the dataset to load
    path=None,           # the raw data directory, for a local dataset
    subset=None,         # the subset, for a dataset that has more than one
    split=None,          # the split, or None for all of them
    cache_dir=None,      # where processed datasets are cached
    revision=None,       # the revision to pin
)
```

It returns a Hugging Face `DatasetDict`, or a `Dataset` when you name a split.

Datasets reach you in one of three ways, and the interface is the same for all of them:

| Type | Datasets | How it loads |
|------|----------|--------------|
| Processed | AUL, BUS-BRA, CAMUS, FetalPlanes, MASLD, PSFHS, STMUS, TN3K | Downloaded ready to use. |
| Raw | COVID-BLUES, MMOTU | The original files are downloaded and preprocessed on your machine the first time. |
| Local | GBCU, OpenKidney | Neither form may be republished, so you download the raw data and pass its directory as `path`. |

```python
# A processed dataset
data = ultrabench.load_dataset(name="AUL", revision="0.4")

# A dataset with more than one subset
data = ultrabench.load_dataset(name="CAMUS", subset="classification")

# A local dataset
data = ultrabench.load_dataset(name="GBCU", path="path/to/raw/data")

# A single split
train_data = ultrabench.load_dataset(name="AUL", split="train")
```

Every dataset publishes the same three splits: `train`, `test`, and `test.256`. Passing an unknown name, subset, or split raises a `ValueError` listing the valid values, and a local dataset with no `path` raises a `ValueError` naming where to get the data.

### Caching

Raw and local datasets are preprocessed on your machine, so the result is cached and only the first call pays for it. The cache directory is the first of:

1. The `cache_dir` argument.
2. The `ULTRABENCH_CACHE` environment variable.
3. `~/.ultrabench`.

Each entry lives at `<cache>/<dataset>/<key>/`, where `key` fingerprints the package version, the dataset revision, the image size, and the raw data path. Changing any of these produces a new entry rather than a stale hit, so a cache never has to be cleared by hand.

`ULTRABENCH_CACHE` governs this cache only. When you pass no cache directory, Hugging Face downloads stay wherever `HF_HOME` puts them.

Read a cache entry with `datasets.load_from_disk`, never `datasets.load_dataset`. The latter silently misreads the layout rather than failing.

### Split Manifests

The splits of the raw and local datasets are committed to `src/ultrabench/datasets/manifests/`, one CSV per split. Preprocessing partitions the raw examples by these files rather than re-running a seeded splitter, so the splits cannot shift with a library upgrade or a change to the raw data, and results stay comparable with the published revision.

An example listed in neither manifest is dropped, which is how the training splits are capped and how duplicate images are removed. An entry with no matching raw example raises a `ValueError`, which is what catches raw data that has changed. COVID-BLUES is listed one video per row rather than one frame, because every frame of a video belongs to the same split.

Regenerate the manifests only when a new revision intentionally re-splits a dataset:

```bash
uv run python scripts/generate_split_manifests.py path/to/processed/datasets
```

## Downloading and Preprocessing

`ultrabench.load_dataset` is the only interface users need. It downloads the processed datasets from the Hugging Face Hub, and downloads (or reads from `path`) and preprocesses the raw and local datasets on first use, caching the result.

Downloading the raw data, processing it and publishing it to the Hub are maintainer tasks. The scripts for them, and how to use them, are in [`scripts/README.md`](https://github.com/adamtupper/ultrabench/blob/main/scripts/README.md).

## Training Split Size

Every published `train` split holds a maximum of 1,000 examples, so that results are comparable across datasets rather than confounded by training-set size. Datasets that already hold fewer than 1,000 training examples are left unchanged. Validation data is carved out of the `train` split at training time, so the effective number of examples used to fit a model is smaller again (800 at the default 80:20 train/validation ratio).

Datasets over the cap are reduced in one of three ways:

| Rule | Datasets | Behaviour |
|------|----------|-----------|
| Move whole patients to `test` | BUS-BRA, FetalPlanes, STMUS | Patients are kept whole and moved from `train` into `test` until the cap is met, so the `test` split grows and no patient appears in both splits. |
| Move images to `test` | PSFHS | Images are moved individually, because the dataset publishes no patient identifiers and UltraBench defines its split itself. |
| Discard the surplus | CAMUS, GBCU, TN3K | The surplus examples are dropped and the `test` split is unchanged. This applies where the original `test` split is patient-disjoint but no patient identifiers are published (GBCU, TN3K), and where frames are subsampled within each sequence to preserve patient diversity (CAMUS). |

Where examples are discarded from a labelled dataset, they are taken from the largest classes first, which makes the class distribution more uniform and keeps the rarer (often more clinically important) classes intact.

## Dataset Revisions

Published datasets are versioned with Git tags on the Hugging Face Hub. Pin a revision so that an experiment stays reproducible when a dataset changes:

```python
import ultrabench

train_data = ultrabench.load_dataset(name="TN3K", revision="0.4", split="train")
```

`revision` pins the published dataset for a processed dataset, and the published raw files for a raw dataset. A local dataset has no revision, so passing one raises a `UserWarning` and the argument is ignored; what fixes its splits instead is the manifest committed alongside the code.

| Revision | Description |
|----------|-------------|
| `0.1` | The initial UltraBench v2 splits. |
| `0.2` | The `train` splits are capped at 1,000 examples. CAMUS is also split into `classification` and `segmentation` subsets. |
| `0.3` | The ground-truth CSVs of the sequence-level classification tasks (CAMUS image quality, COVID-BLUES, MASLD) gain a `sequence_id` column. The images are unchanged. |
| `0.4` | Each test split gets its own ground-truth CSVs, with the segmentation masks encoded at the resolution of that split. Earlier revisions publish full-resolution masks under the `test.256` name. The images are unchanged. From this revision every dataset in the collection shares one revision number. |

Each task configuration in `src/ultrabench/evaluate/task_configs/` pins the revision its task is scored against. Pass the same `revision` to `load_dataset` when you train, so your predictions and the ground truth come from the same revision.

The maintainer scripts refuse to publish a revision that is already tagged, so re-running `--publish` cannot overwrite a released revision. Create the tag once the datasets are pushed:

```bash
hf repos tag create UltraBench/TN3K 0.2 --type dataset -m "Cap training splits at 1,000 samples."
```

## Output Directory Structure

A processed dataset is a Hugging Face `DatasetDict` written with `save_to_disk`. The `load_dataset` cache entries and the output of `scripts/process_dataset.py` (under `<output_dir>/UltraBench/<Dataset>/`) share this layout:

```
UltraBench/AUL/
├── dataset_dict.json
├── version.txt              # Package version and build date
├── train/                   # Original-size training images
├── test/                    # Original-size test images
│   └── ground_truth_*.csv   # One per task, at this resolution
└── test.256/                # Test images resized to 256x256
    └── ground_truth_*.csv   # One per task, at this resolution
```

A dataset with more than one subset nests each subset one level deeper, for example `UltraBench/CAMUS/classification/`.

Read this layout with `datasets.load_from_disk`. `datasets.load_dataset` silently misreads it.

Loading a raw or local dataset writes the same layout into the cache, so a dataset you preprocess yourself can also be scored by the evaluation layer.

## Images and Masks

- **Images**: All images are saved as 3-channel (RGB) PNG files.
- **Test images**: Resized to fit within `size x size` pixels (preserving aspect ratio) then center-padded with black pixels to reach exactly `size x size`.
- **Masks**: Saved as single-channel PNG files (grayscale).
- **Scan masks**: Binary masks that indicate the ultrasound scan region (non-zero = scan area). Present in all datasets.

## Example: PyTorch + Albumentations

Datasets carry PIL images, so a transform is applied with `with_transform`:

```python
import albumentations as A
import numpy as np
import ultrabench
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
    [A.Normalize(mean=0.0, std=1.0), ToTensorV2()],
    additional_targets={"mask": "mask", "scan_mask": "mask"},
)


def apply(batch):
    """Apply the transform to a batch of examples."""
    return {
        "image": [
            transform(image=np.array(image))["image"] for image in batch["image"]
        ],
        "target": batch["mass_mask"],
    }


train_data = ultrabench.load_dataset(name="AUL", split="train")
train_data = train_data.with_transform(apply)
```

The example training scripts do this for every task in [`examples/common.py`](../examples/common.py), including the train/validation split and the resize that matches the `test.256` split.
