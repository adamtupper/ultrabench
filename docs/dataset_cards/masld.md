# MASLD Dataset

The MALSD dataset consists of 550 liver ultrasound images from 55 patients from the *Dataset of B-mode fatty liver ultrasound images*. 38 of the 55 patients are diagnosed with Metabolic dysfunction-associated steatotic liver disease (MASLD), formerly known as nonalcoholic fatty liver disease (NAFLD). MASLD is defined as >5 % of hepatocytes having fatty infiltration. This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for MALSD classification.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> M. Byra, G. Styczynski, C. Szmigielski, P. Kalinowski. Ł. Michałowski. R. Paluszkiewicz. B. Ziarkiewicz-Wróblewska, K. Zieniewicz. P. Sobieraj, A. Nowicki. Transfer learning with deep convolutional neural network for liver steatosis assessment in ultrasound images. International Journal of Computer Assisted Radiology and Surgery, 2018. DOI: 10.1007/s11548-018-1843-2

## Splits

The patients of original dataset have been divided into `train` and `test` splits using a 80:20 split:

- `train`: 80% of the patients with the images in their original sizes.
- `test`: The remaining 20% of patients with the images in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

## Features

All splits have the following features:

| Feature      | Feature Type       | Description                                                                           |
| ------------ | ------------------ | ------------------------------------------------------------------------------------- |
| `patient`    | `Value("int32")`   | Patient ID.                                                                           |
| `frame`      | `Value("int32")`   | Frame index.                                                                          |
| `image`      | `Image`            | 3-channel ultrasound image.                                                           |
| `scan_mask`  | `Image`            | Single-channel, binary ultrasound scan mask generated using morphological operations. |
| `percentage` | `Value("float32")` | Percentage of hepatocytes with steatosis.                                             |
| `label`      | `ClassLabel`       | MASLD (negative = 0, positive = 1)                                                    |

## Tasks

The MASLD dataset supports one task:

- MASLD classification

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/MASLD", split="train")
test_data = datasets.load_dataset("UltraBench/MASLD", split="test.256")
```