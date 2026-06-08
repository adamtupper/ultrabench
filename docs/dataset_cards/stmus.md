# Segmentation of Transverse Musculoskeletal Ultrasound Images (STMUS) Dataset

STMUS ([Mendeley Data](https://data.mendeley.com/datasets/3jykz7wz8d/1)) is a dataset for transverse musculoskeletal ultrasound segmentation. It consists of 8,169 images of three different muscles (the tibialis anterior, gastrocnemius medialis, and biceps brachii) from 1,363 patients. Each image is accompanied by an echogenicity level label and binary muscle segmentation mask. This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for muscle segmentation.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original works:
>
> Marzola, F., van Alfen, N., Doorduin, J., & Meiburger, K. M. (2021). Deep learning segmentation of transverse musculoskeletal ultrasound images for neuromuscular disease assessment. Computers in Biology and Medicine, 135, 104623. https://doi.org/10.1016/j.compbiomed.2021.104623


## Splits

The original dataset has been divided into `train` and `test` splits by patient:

- `train`: 80% of the patients with the images in their original resolution.
- `test`: The remaining 20% of patients with the images in their original resolution.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

## Features

All splits have the following features:

| Feature        | Feature Type      | Description                                                                               |
| -------------- | ----------------- | ----------------------------------------------------------------------------------------- |
| `filename`     | `Value("string")` | Original image filenames (without the file extension) prefixed by muscle (bb, gm, or ta) and echogenicity (healthy or pathological) |
| `patient`      | `Value("string")` | The patient ID                                                                            |
| `muscle`       | `Value("string")` | The muscle in the image (biceps brachii, gastrocnemius medialis, or tibialis anterior)    |
| `echogenicity` | `Value("string")` | The echogenicity level (healthy or pathological)                                          |
| `image`        | `Image`           | 3-channel ultrasound images                                                               |
| `mask`         | `Image`           | Single-channel, binary muscle segmentation mask (background = 0, muscle = 1)              |
| `scan_mask`    | `Image`           | Single-channel, binary ultrasound scan mask (all ones; the scan fills the entire image for linear-probe acquisitions) |

The original dataset contains additional characteristic information per patient (age, height, weight, and sex) that are not included here.

## Tasks

The STMUS dataset supports one task:

- Binary muscle segmentation

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/STMUS", split="train")
test_data = datasets.load_dataset("UltraBench/STMUS", split="test")
```
