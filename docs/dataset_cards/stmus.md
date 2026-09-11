# Segmentation of Transverse Musculoskeletal Ultrasound Images (STMUS) Dataset

STMUS ([Mendeley Data](https://data.mendeley.com/datasets/3jykz7wz8d/1)) is a dataset for transverse musculoskeletal ultrasound segmentation. It consists of 8,169 images of three different muscles (the tibialis anterior, gastrocnemius medialis, and biceps brachii) from 1,223 patients. Each image is accompanied by an echogenicity level label, patient characteristics, and a binary muscle segmentation mask. This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for muscle segmentation.

The subject numbers in the original filenames are local to each muscle and echogenicity directory, so patients are identified here by joining the per-muscle characteristics spreadsheets. Two subjects whose characteristics are identical are treated as a single patient, so that their images are never split across the two splits. The original paper reports 3,917 images from 1,283 subjects. Those figures describe the authors' own train, validation, and test subset, and count each muscle scanned for a patient as a separate subject. This version includes every released image.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original works:
>
> Marzola, F., van Alfen, N., Doorduin, J., & Meiburger, K. M. (2021). Deep learning segmentation of transverse musculoskeletal ultrasound images for neuromuscular disease assessment. Computers in Biology and Medicine, 135, 104623. https://doi.org/10.1016/j.compbiomed.2021.104623


## Splits

The original dataset has been divided into `train` and `test` splits by patient:

- `train`: 1,000 images from 151 patients, in their original resolution.
- `test`: The remaining 7,169 images from 1,072 patients, in their original resolution.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

There is no patient overlap between the `train` and `test` splits.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.3`: Patients are identified from the per-muscle characteristics spreadsheets rather than the subject numbers in the filenames, which are local to each directory. This corrects the patient count from 1,363 to 1,223 and removes the patient overlap between the splits, which previously affected 253 of the 296 `train` patients. The `train` split becomes 1,000 images from 151 patients and the `test` split 7,169 images from 1,072 patients. The `age`, `height`, `weight`, and `sex` features are added.
- `0.2`: The `train` split is capped at 1,000 images (322 of the 2,122 training patients). Whole patients are moved into the `test` split until the cap is met, so the `test` split grows from 1,636 to 7,169 images.
- `0.1`: The original 80:20 split by patient, with 6,533 training and 1,636 test images.

## Features

All splits have the following features:

| Feature        | Feature Type      | Description                                                                               |
| -------------- | ----------------- | ----------------------------------------------------------------------------------------- |
| `filename`     | `Value("string")` | Original image filenames (without the file extension) prefixed by muscle (bb, gm, or ta) and echogenicity (healthy or pathological) |
| `patient`      | `Value("string")` | The patient ID                                                                            |
| `muscle`       | `Value("string")` | The muscle in the image (biceps brachii, gastrocnemius medialis, or tibialis anterior)    |
| `echogenicity` | `Value("string")` | The echogenicity level (healthy or pathological)                                          |
| `age`          | `Value("float32")`| The age of the patient in years                                                           |
| `height`       | `Value("float32")`| The height of the patient in cm                                                  |
| `weight`       | `Value("float32")`| The weight of the patient in kg                                                    |
| `sex`          | `Value("string")` | The sex of the patient (M or F)                                                           |
| `image`        | `Image`           | 3-channel ultrasound images                                                               |
| `mask`         | `Image`           | Single-channel, binary muscle segmentation mask (background = 0, muscle = 1)              |
| `scan_mask`    | `Image`           | Single-channel, binary ultrasound scan mask (all ones; the scan fills the entire image for linear-probe acquisitions) |

The patient characteristics are taken from the `characteristics.xlsx` spreadsheets in the original dataset, which record them once per patient.

## Tasks

The STMUS dataset supports one task:

- Binary muscle segmentation

## Usage

Load this dataset with `ultrabench2.load_dataset`, which returns a Hugging Face [`DatasetDict`](https://huggingface.co/docs/datasets/index):

```python
import ultrabench2

train_data = ultrabench2.load_dataset(
    name="STMUS",
    split="train",
    revision="0.4",
)
test_data = ultrabench2.load_dataset(
    name="STMUS",
    split="test.256",
    revision="0.4",
)
```
