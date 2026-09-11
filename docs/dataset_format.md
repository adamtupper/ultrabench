# Dataset Format

UltraBench provides a standardized preprocessing pipeline that converts raw ultrasound datasets into a consistent directory structure.

## Loading Datasets

`ultrabench2.load_dataset` loads any dataset in the collection through one interface:

```python
import ultrabench2

dataset = ultrabench2.load_dataset(
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
data = ultrabench2.load_dataset(name="AUL", revision="0.4")

# A dataset with more than one subset
data = ultrabench2.load_dataset(name="CAMUS", subset="classification")

# A local dataset
data = ultrabench2.load_dataset(name="GBCU", path="path/to/raw/data")

# A single split
train_data = ultrabench2.load_dataset(name="AUL", split="train")
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

The splits of the raw and local datasets are committed to `src/ultrabench2/datasets/manifests/`, one CSV per split. Preprocessing partitions the raw examples by these files rather than re-running a seeded splitter, so the splits cannot shift with a library upgrade or a change to the raw data, and results stay comparable with the published revision.

An example listed in neither manifest is dropped, which is how the training splits are capped and how duplicate images are removed. An entry with no matching raw example raises a `ValueError`, which is what catches raw data that has changed. COVID-BLUES is listed one video per row rather than one frame, because every frame of a video belongs to the same split.

Regenerate the manifests only when a new revision intentionally re-splits a dataset:

```bash
uv run python scripts/generate_split_manifests.py path/to/processed/datasets
```

## Downloading Datasets

Some datasets can be downloaded automatically using the `ultrabench2 download` command:

```bash
ultrabench2 download <dataset> <output_dir>
```

Downloadable datasets:

| Dataset | Command |
|---------|---------|
| AUL | `ultrabench2 download aul <output_dir>` |
| CAMUS | `ultrabench2 download camus <output_dir>` |
| MASLD | `ultrabench2 download masld <output_dir>` |
| PSFHS | `ultrabench2 download psfhs <output_dir>` |

For datasets not listed above, refer to their respective sources and download them manually before running the preprocessing step.

## Preprocessing Datasets

Once the raw data is available, preprocess it into the standardized format using the `ultrabench2 process` command:

```bash
ultrabench2 process <dataset> <raw_dir> <output_dir> [--size 256] [--publish]
```

- `<raw_dir>`: Path to the downloaded raw dataset directory.
- `<output_dir>`: Path where the preprocessed dataset will be saved.
- `--size`: Target height and width (in pixels) for test images (default: `256`).
- `--publish`: Push the processed dataset to the Hugging Face Hub.

All 12 datasets are supported:

| Dataset | Command |
|---------|---------|
| AUL | `ultrabench2 process aul <raw_dir> <output_dir>` |
| BUS-BRA | `ultrabench2 process busbra <raw_dir> <output_dir>` |
| CAMUS | `ultrabench2 process camus <raw_dir> <output_dir>` |
| COVID-BLUES | `ultrabench2 process covid-blues <raw_dir> <output_dir>` |
| Fetal Planes | `ultrabench2 process fetal-planes <raw_dir> <output_dir>` |
| GBCU | `ultrabench2 process gbcu <raw_dir> <output_dir>` |
| MASLD | `ultrabench2 process masld <raw_file> <output_dir>` |
| MMOTU | `ultrabench2 process mmotu <raw_dir> <output_dir>` |
| Open Kidney | `ultrabench2 process open-kidney <raw_dir> <output_dir>` |
| PSFHS | `ultrabench2 process psfhs <raw_dir> <output_dir>` |
| STMUS | `ultrabench2 process stmus <raw_dir> <output_dir>` |
| TN3K | `ultrabench2 process tn3k <raw_dir> <output_dir>` |

Most users do not need this command. `ultrabench2.load_dataset` preprocesses the raw and local datasets for you and caches the result; use `ultrabench2 process` when you are publishing a dataset or want the output on disk.

## Publishing Raw Data

COVID-BLUES and MMOTU may be redistributed unmodified but not in a preprocessed form, so their original files are published instead and `ultrabench2.load_dataset` preprocesses them locally. Publish them with:

```bash
ultrabench2 publish-raw <dataset> <raw_dir>
```

Only the files the processing code reads are uploaded, and the layout on the Hub is the layout `ultrabench2 process` expects. Tag the revision afterwards, as the command prints.

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
import ultrabench2

train_data = ultrabench2.load_dataset(name="TN3K", revision="0.4", split="train")
```

`revision` pins the published dataset for a processed dataset, and the published raw files for a raw dataset. A local dataset has no revision, so passing one raises a `UserWarning` and the argument is ignored; what fixes its splits instead is the manifest committed alongside the code.

| Revision | Description |
|----------|-------------|
| `0.1` | The initial UltraBench 2 splits. |
| `0.2` | The `train` splits are capped at 1,000 examples. CAMUS is also split into `classification` and `segmentation` subsets. |
| `0.3` | The ground-truth CSVs of the sequence-level classification tasks (CAMUS image quality, COVID-BLUES, MASLD) gain a `sequence_id` column. The images are unchanged. |
| `0.4` | Each test split gets its own ground-truth CSVs, with the segmentation masks encoded at the resolution of that split. Earlier revisions publish full-resolution masks under the `test.256` name. The images are unchanged. From this revision every dataset in the collection shares one revision number. |

Set `revision` in a task configuration (`src/ultrabench2/training/config/task/*.yaml`) to pin the revision a training run uses, and pass `--revision` to `scripts/slurm/download_dataset.py` to pre-cache the matching revision.

The processing code refuses to publish a revision that is already tagged, so re-running `--publish` cannot overwrite a released revision. Create the tag once the datasets are pushed:

```bash
hf repos tag create UltraBench/TN3K 0.2 --type dataset -m "Cap training splits at 1,000 samples."
```

## Output Directory Structure

`ultrabench2 process` writes a Hugging Face `DatasetDict` with `save_to_disk`, under `<output_dir>/UltraBench/<Dataset>/`:

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
import ultrabench2
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


train_data = ultrabench2.load_dataset(name="AUL", split="train")
train_data = train_data.with_transform(apply)
```

The training layer does this for every task in [`src/ultrabench2/training/data/utils.py`](../src/ultrabench2/training/data/utils.py), including the train/validation split and the sequence pooling some tasks need.
