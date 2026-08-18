# Fetal Planes Dataset

Fetal Planes is a dataset of 12,400 routinely acquired maternal-fetal screening ultrasound images from 1,792 patients. The images are divided into 6 classes: four of the most widely used fetal anatomical planes (Abdomen, Brain, Femur and Thorax), the mother’s cervix (widely used for prematurity screening) and a general category to include any other less common image planes. The fetal brain images are further categorized into the 3 most common fetal brain planes (Trans-thalamic, Trans-cerebellum, Trans-ventricular and Other to judge fine grain categorization performance. This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for maternal-fetal screening tasks.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original source:
>
> Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci Rep 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5

Following the original dataset, there are two subsets for the dataset: `fetal` and `fetal_brain`. These subsets contain all images and the subset of fetal brain images, respectively. 

## Splits

The dataset is split according to the original `train` and `test` splits, with each subset's `train` split capped at a maximum of 1,000 examples. Whole patients are moved from `train` into `test` until the cap is met, so each subset remains split by patient.

The two subsets are capped independently, so each gets the full training budget. A patient may therefore be in the `train` split of one subset and the `test` split of the other.

### `fetal` Subset

- `train`: 1,000 images from 115 patients in their original sizes.
- `test`: 11,400 images from 1,677 patients in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

### `fetal_brain` Subset

- `train`: 1,000 images from 341 patients in their original sizes.
- `test`: 2,092 images from 741 patients in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.2`: Each subset's `train` split is capped at 1,000 images, with the overflow patients moved into the corresponding `test` split.
- `0.1`: The original splits (`fetal`: 7,129 train and 5,271 test images; `fetal_brain`: 1,620 train and 1,472 test images).

## Features

### `fetal` Subset

All splits have the following features:

| Feature       | Feature Type      | Description                                                                                                          |
| ------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `filename`    | `Value("string")` | Original image filename (without the file extension).                                                                |
| `patient`     | `Value("int32")`  | Patient ID.                                                                                                          |
| `plane`       | `ClassLabel`      | Fetal plane (Fetal abdomen = 0, Fetal brain = 1, Fetal femur = 2, Fetal thorax = 3, Maternal cervix = 4, Other = 5). |
| `brain_plane` | `Value("string")` | Fetal brain plane ("Not A Brain", "Trans-thalamic", "Trans-cerebellum", "Trans-ventricular", "Other").               |
| `operator`    | `Value("string")` | Operator ID ("Op. 1", "Op. 3", "Op. 2", "Other").                                                                    |
| `machine`     | `Value("string")` | Ultrasound machine ("Aloka", "Voluson E6", "Voluson S10", "Other").                                                  | 
| `image`       | `Image`           | 3-channel ultrasound image.                                                                                          |
| `scan_mask`   | `Image`           | Single-channel, binary ultrasound scan mask generated using morphological operations.                                |


### `fetal_brain` Subset

All splits have the following features:

| Feature       | Feature Type      | Description                                                                                     |
| ------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| `filename`    | `Value("string")` | Original image filename (without the file extension).                                           |
| `patient`     | `Value("int32")`  | Patient ID.                                                                                     |
| `brain_plane` | `ClassLabel     ` | Fetal brain plane (Trans-thalamic = 0, Trans-cerebellum = 1, Trans-ventricular = 2, Other = 3). |
| `operator`    | `Value("string")` | Operator ID ("Op. 1", "Op. 3", "Op. 2", "Other").                                               |
| `machine`     | `Value("string")` | Ultrasound machine ("Aloka", "Voluson E6", "Voluson S10", "Other").                             | 
| `image`       | `Image`           | 3-channel ultrasound image.                                                                     |
| `scan_mask`   | `Image`           | Single-channel, binary ultrasound scan mask generated using morphological operations.           |

## Tasks

The Fetal Planes dataset supports two tasks:

- Fetal plane classification
- Fine-grained fetal brain plane classification.

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

# Fetal plane classification
train_data = datasets.load_dataset(
    "UltraBench/FetalPlanes", name="fetal", revision="0.2", split="train"
)
test_data = datasets.load_dataset(
    "UltraBench/FetalPlanes", name="fetal", revision="0.2", split="test.256"
)

# Fetal brain plane classification
train_data = datasets.load_dataset(
    "UltraBench/FetalPlanes", name="fetal_brain", revision="0.2", split="train"
)
test_data = datasets.load_dataset(
    "UltraBench/FetalPlanes", name="fetal_brain", revision="0.2", split="test.256"
)
```
