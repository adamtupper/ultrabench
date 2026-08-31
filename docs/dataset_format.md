# Dataset Format

UltraBench provides a standardized preprocessing pipeline that converts raw ultrasound datasets into a consistent directory structure.

## Downloading Datasets

Some datasets can be downloaded automatically using the `ultrabench download` command:

```bash
ultrabench download <dataset> <output_dir>
```

Downloadable datasets:

| Dataset | Command |
|---------|---------|
| AUL | `ultrabench download aul <output_dir>` |
| Butterfly | `ultrabench download butterfly <output_dir>` |
| CAMUS | `ultrabench download camus <output_dir>` |
| MASLD | `ultrabench download masld <output_dir>` |
| POCUS | `ultrabench download pocus <output_dir>` |
| PSFHS | `ultrabench download psfhs <output_dir>` |

For datasets not listed above, refer to their respective sources and download them manually before running the preprocessing step.

## Preprocessing Datasets

Once the raw data is available, preprocess it into the standardized format using the `ultrabench process` command:

```bash
ultrabench process <dataset> <raw_dir> <output_dir> [--size 256] [--publish]
```

- `<raw_dir>`: Path to the downloaded raw dataset directory.
- `<output_dir>`: Path where the preprocessed dataset will be saved.
- `--size`: Target height and width (in pixels) for test images (default: `256`).
- `--publish`: Push the processed dataset to the Hugging Face Hub.

All 10 datasets are supported:

| Dataset | Command |
|---------|---------|
| AUL | `ultrabench process aul <raw_dir> <output_dir>` |
| Butterfly | `ultrabench process butterfly <raw_dir> <output_dir>` |
| CAMUS | `ultrabench process camus <raw_dir> <output_dir>` |
| MASLD | `ultrabench process MASLD <raw_dir> <output_dir>` |
| GBCU | `ultrabench process gbcu <raw_dir> <output_dir>` |
| MMOTU | `ultrabench process mmotu <raw_dir> <output_dir>` |
| Open Kidney | `ultrabench process open-kidney <raw_dir> <output_dir>` |
| POCUS | `ultrabench process pocus <raw_dir> <output_dir>` |
| PSFHS | `ultrabench process psfhs <raw_dir> <output_dir>` |
| Stanford Thyroid | `ultrabench process stanford-thyroid <raw_dir> <output_dir>` |

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
from datasets import load_dataset

train_data = load_dataset("UltraBench/TN3K", revision="0.2", split="train")
```

| Revision | Description |
|----------|-------------|
| `0.1` | The initial UltraBench 2 splits. |
| `0.2` | The `train` splits are capped at 1,000 examples. CAMUS is also split into `classification` and `segmentation` subsets. |
| `0.3` | The ground-truth CSVs of the sequence-level classification tasks (CAMUS image quality, COVID-BLUES, MASLD) gain a `sequence_id` column. The images are unchanged. |

Set `revision` in a task configuration (`src/ultrabench/training/config/task/*.yaml`) to pin the revision a training run uses, and pass `--revision` to `scripts/slurm/download_dataset.py` to pre-cache the matching revision.

The processing code refuses to publish a revision that is already tagged, so re-running `--publish` cannot overwrite a released revision. Create the tag once the datasets are pushed:

```bash
hf repos tag create UltraBench/TN3K 0.2 --type dataset -m "Cap training splits at 1,000 samples."
```

## Output Directory Structure

Each preprocessed dataset is saved under `<output_dir>/<dataset>_v{version}/` with the following layout:

```
<dataset>_v{version}/
├── version.txt              # Dataset version string
├── train_val/               # Training and validation split
│   ├── images/              # Input images (3-channel PNG)
│   ├── masks/
│   │   └── <type>/          # Task-specific mask subdirectories
│   └── train_val.json       # Metadata for all train/val examples
└── test_256/                # Test split (suffix reflects --size, e.g. test_256)
    ├── images/              # Input images resized to 256×256 (3-channel PNG)
    ├── masks/
    │   └── <type>/          # Task-specific mask subdirectories
    └── test.json            # Metadata for all test examples
```

All file paths stored in the JSON metadata are relative to their split directory (e.g., `"images/foo.png"` resolves to `train_val/images/foo.png`).

Processing also writes a ground-truth CSV for the test split of each task, which is published alongside the images on the Hugging Face Hub and read by the evaluation layer. See [the evaluation documentation](evaluation.md) for its columns.

## Images and Masks

- **Images**: All images are saved as 3-channel (RGB) PNG files.
- **Test images**: Resized to fit within `size x size` pixels (preserving aspect ratio) then center-padded with black pixels to reach exactly `size x size`.
- **Masks**: Saved as single-channel PNG files (grayscale).
- **Scan masks**: Binary masks that indicate the ultrasound scan region (non-zero = scan area). Present in all datasets.

## JSON Metadata Format

Each split directory contains a single JSON file containing an array of objects, one per example. All file paths are relative to the split directory.

### Classification example (`aul_mass`)

```json
[
    {
        "pathology": "Benign",
        "label": 1,
        "image": "images/benign_65.png",
        "scan_mask": "masks/scan/benign_65.png",
        "liver_mask": "masks/liver/benign_65.png",
        "mass_mask": "masks/mass/benign_65.png"
    },
    ...
]
```

- `label`: Integer class index.
- `pathology`: Human-readable class name.
- `image`: Path to the input image.
- `scan_mask`: Path to the ultrasound scan region mask.
- Additional task-specific mask keys (e.g., `liver_mask`, `mass_mask`) vary by dataset.

### Segmentation example (`psfhs`)

```json
[
    {
        "image": "images/03795.png",
        "scan_mask": "masks/scan/03795.png",
        "psfh_mask": "masks/psfh/03795.png"
    },
    ...
]
```

- `image`: Path to the input image.
- `scan_mask`: Path to the ultrasound scan region mask.
- Task-specific mask key (e.g., `psfh_mask`) contains the segmentation target.

## Example: PyTorch + Albumentations

The file [`examples/pytorch_datasets_with_albumentations.py`](../examples/pytorch_datasets_with_albumentations.py) provides ready-to-use `ClassificationDataset` and `SegmentationDataset` classes.

### Classification

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from examples.pytorch_datasets_with_albumentations import ClassificationDataset

transform = A.Compose(
    [
        A.Normalize(mean=0.0, std=1.0),  # e.g., normalize to [0, 1]
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask", "scan_mask": "mask"},
)

train_dataset = ClassificationDataset(
    root_dir="path/to/aul_mass_v1.0.1/train_val/",
    transform=transform,
)
test_dataset = ClassificationDataset(
    root_dir="path/to/aul_mass_v1.0.1/test_256/",
    transform=transform,
)
```

### Segmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from examples.pytorch_datasets_with_albumentations import SegmentationDataset

transform = A.Compose(
    [
        A.Normalize(mean=0.0, std=1.0),  # e.g., normalize to [0, 1]
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask", "scan_mask": "mask"},
)

train_dataset = SegmentationDataset(
    root_dir="path/to/psfhs_v1.0.1/train_val/",
    transform=transform,
    mask_key="psfh_mask",
)
test_dataset = SegmentationDataset(
    root_dir="path/to/psfhs_v1.0.1/test_256/",
    transform=transform,
    mask_key="psfh_mask",
)
```
