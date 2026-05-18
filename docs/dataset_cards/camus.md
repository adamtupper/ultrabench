# CAMUS Dataset

The Cardiac Acquisitions for Multi-structure Ultrasound Segmentation (CAMUS) dataset consists consists of apical four-chamber and two-chamber view cardiac ultrasound sequences from 500 patients for a total of 19,232 images. Each image is accompanied by segmentation masks for the left ventricle (LV) endocardium, left ventricle (LV) myocardium, and the left atrium (LA). Each image is also labeled according to the quality of the scan (*poor*, *medium*, and *good*). This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for cardiac structure segmentation and image quality assessment.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> S. Leclerc et al., "Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography," in IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, Sept. 2019, doi: 10.1109/TMI.2019.2900516

## Splits

The dataset is split according to the original `train` and `test` splits with 450 and 50 patients, respectively (an 80:20 split):

- `train`: 80% of the patients with the images in the original sizes.
- `test`: The remaining 20% of patients with images in the original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

## Features

All splits have the following features:

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

The CAMUS dataset supports two tasks:

- Left ventricle (LV) endocardium, left ventricle (LV) myocardium, and left atrium (LA) segmentation.
- Image quality assessment.

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/CAMUS", split="train")
test_data = datasets.load_dataset("UltraBench/CAMUS", split="test.256")
```