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
| Fatty Liver | `ultrabench download fatty-liver <output_dir>` |
| POCUS | `ultrabench download pocus <output_dir>` |
| PSFHS | `ultrabench download psfhs <output_dir>` |

For datasets not listed above, refer to their respective sources and download them manually before running the preprocessing step.

## Preprocessing Datasets

Once the raw data is available, preprocess it into the standardized format using the `ultrabench process` command:

```bash
ultrabench process <dataset> <raw_dir> <output_dir> [--size 256]
```

- `<raw_dir>`: Path to the downloaded raw dataset directory.
- `<output_dir>`: Path where the preprocessed dataset will be saved.
- `--size`: Target height and width (in pixels) for test images (default: `256`).

All 10 datasets are supported:

| Dataset | Command |
|---------|---------|
| AUL | `ultrabench process aul <raw_dir> <output_dir>` |
| Butterfly | `ultrabench process butterfly <raw_dir> <output_dir>` |
| CAMUS | `ultrabench process camus <raw_dir> <output_dir>` |
| Fatty Liver | `ultrabench process fatty-liver <raw_dir> <output_dir>` |
| GBCU | `ultrabench process gbcu <raw_dir> <output_dir>` |
| MMOTU | `ultrabench process mmotu <raw_dir> <output_dir>` |
| Open Kidney | `ultrabench process open-kidney <raw_dir> <output_dir>` |
| POCUS | `ultrabench process pocus <raw_dir> <output_dir>` |
| PSFHS | `ultrabench process psfhs <raw_dir> <output_dir>` |
| Stanford Thyroid | `ultrabench process stanford-thyroid <raw_dir> <output_dir>` |

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
