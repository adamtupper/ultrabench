# CAMUS Dataset

The Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS) dataset consists consists of apical four-chamber and two-chamber view cardiac ultrasound sequences from 500 patients for a total of 19,232 images. Each image is accompanied by segmentation masks for the left ventricle (LV) endocardium, left ventricle (LV) myocardium, and the left atrium (LA). Each image is also labeled according to the quality of the scan (*poor*, *medium*, and *good*). This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for cardiac structure segmentation and image quality assessment.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> S. Leclerc et al., "Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography," in IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, Sept. 2019, doi: 10.1109/TMI.2019.2900516

## Subsets

The dataset has two subsets, which differ only in their `train` split:

- `classification`: All 17,264 training frames. Image quality is a sequence-level label and the training split holds only 900 sequences (450 patients x 2 views), which is already below the 1,000-example training cap, so the frames are not subsampled.
- `segmentation`: 1,000 training frames, drawn uniformly at random from within each sequence. All 450 training patients and both views are retained, so each sequence contributes one or two frames.

Both subsets share the same `test` and `test.256` splits.

## Splits

The dataset is split according to the original `train` and `test` splits with 450 and 50 patients, respectively:

- `train`: The 450 training patients. 17,264 frames in the `classification` subset and 1,000 frames in the `segmentation` subset, in the original sizes.
- `test`: The remaining 50 patients (1,968 frames) with images in the original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.3`: The classification ground-truth CSV gains a `sequence_id` column so that image quality is evaluated per sequence. The images, masks, and splits are unchanged.
- `0.2`: Split into the `classification` and `segmentation` subsets described above. The `segmentation` training split is subsampled to 1,000 frames and the surplus frames are discarded rather than moved to the `test` split.
- `0.1`: A single unnamed configuration holding all 17,264 training frames.

## Features

All splits of both subsets have the following features:

| Feature             | Feature Type      | Description                                                                                       |
| ------------------- | ----------------- | ------------------------------------------------------------------------------------------------- |
| `patient`           | `Value("string")` | Patient ID.                                                                                       |
| `view`              | `Value("string")` | View of the sequence (2CH or 4CH).                                                                |
| `frame`             | `Value("int32")`  | Frame number of the image in the sequence.                                                        |
| `sex`               | `Value("string")` | Sex of the patient (M or F).                                                                      |
| `age`               | `Value("int32")`  | Age of the patient.                                                                               |
| `image_quality`     | `ClassLabel`      | Quality of the image (poor = 0, medium = 1, good = 2).                                            |
| `ejection_fraction` | `Value(float32)`  | Ejection fraction.                                                                                |
| `end_systole`       | `Value(bool)`     | End-Systole / Systolic frame.                                                                     |
| `end_diastole`      | `Value(bool)`     | End-Diastole / Diastolic frame.                                                                   |
| `frame_rate`        | `Value(float32)`  | Frame rate of the sequence.                                                                       |
| `image`             | `Image`           | 3-channel ultrasound image.                                                                       |
| `mask`              | `Image`           | Single-channel, multi-class mask (background = 0, LV endocardium = 1, LV myocardium = 2, LA = 3). |
| `scan_mask`         | `Image`           | The single-channel, binary ultrasound scan masks generated using morphological operations.        |

## Tasks

The CAMUS dataset supports two tasks, one per subset:

- Left ventricle (LV) endocardium, left ventricle (LV) myocardium, and left atrium (LA) segmentation (`segmentation` subset).
- Image quality assessment (`classification` subset).

## Evaluation

Segmentation is evaluated per image. Image quality assessment is evaluated per **sequence**, since the quality grade describes the whole acquisition rather than an individual frame. A sequence is one `(patient, view)` pair, so the `test` split holds 100 sequences (50 patients x 2 views) across its 1,968 frames.

Generate one prediction per sequence by averaging the predicted probabilities over every frame in the sequence, then taking the argmax. The ground-truth CSV carries a `sequence_id` column (`{patient}_{view}`, for example `313_2CH`) that identifies which frames belong together, so the predictions CSV stays per image:

```bash
python -m ultrabench.evaluate task=classification aggregate_by=sequence_id \
  ground_truth=classification-test-ground-truth-classification.csv \
  predictions=predictions.csv
```

See [the evaluation documentation](../evaluation.md) for the CSV format.

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index). You must name a subset, since the dataset has no default configuration:

```python
from datasets import load_dataset

# Cardiac structure segmentation
train_data = datasets.load_dataset(
    "UltraBench/CAMUS", name="segmentation", revision="0.3", split="train"
)
test_data = datasets.load_dataset(
    "UltraBench/CAMUS", name="segmentation", revision="0.3", split="test.256"
)

# Image quality assessment
train_data = datasets.load_dataset(
    "UltraBench/CAMUS", name="classification", revision="0.3", split="train"
)
test_data = datasets.load_dataset(
    "UltraBench/CAMUS", name="classification", revision="0.3", split="test.256"
)
```