# Annotated Ultrasound Liver (AUL) Dataset

The Annotated Ultrasound Liver (AUL) dataset consists of 735 liver ultrasound images with accompanying liver and mass segmentation masks and image labels (*normal*, *benign*, *malignant*). This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for liver and mass segmentation, and liver mass classification.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> Yiming Xu, Bowen Zheng, Xiaohong Liu, Tao Wu, Jinxiu Ju, Shijie Wang, Yufan Lian, Hongjun Zhang, Tong Liang, Ye Sang, Rui Jiang, Guangyu Wang, Jie Ren, Ting Chen, Improving artificial intelligence pipeline for liver malignancy diagnosis using ultrasound images and video frames, Briefings in Bioinformatics, Volume 24, Issue 1, January 2023, bbac569, https://doi.org/10.1093/bib/bbac569

## Splits

The original dataset has been divided into `train` and `test` splits using a stratified 80:20 split:

- `train`: 80% of the images in their original sizes.
- `test`: The remaining 20% of images in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                                                        |
| ------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| `filename`   | `Value("string")` | The original filenames for the image (without the file extension).                                                 |
| `image`      | `Image`           | The 3-channel ultrasound images.                                                                                   |
| `scan_mask`  | `Image`           | The single-channel, binary ultrasound scan masks. Missing scan masks are generated using morphological operations. |
| `liver_mask` | `Image`           | The single-channel, binary liver masks.                                                                            |
| `mass_mask`  | `Image`           | The single-channel, binary liver mass masks.                                                                       |
| `label`      | `ClassLabel`      | The type of the mass (normal/no mass = 0, benign = 1, malignant = 2)                                               |

## Tasks

The AUL dataset supports three tasks:

- Liver segmentation
- Liver mass segmentation
- Liver mass classification

Note that  image `malignant_375` is missing a liver segmentation mask, so there is one less training example for this task.

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/AUL", split="train")

# Optional, if training for liver segmentation
train_data = train_data.filter(lambda x: x["liver_mask"] is not None)

test_data = datasets.load_dataset("UltraBench/AUL", split="test.256")
```