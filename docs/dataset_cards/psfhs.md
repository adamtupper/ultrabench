---
title: PSFHS
parent: Datasets
nav_order: 10
---

# Pubic Symphysis and Fetal Head Segmentation (PSFHS) Dataset

The PSFHS dataset is a dataset for fetal head and pubic symphysis segmentation with 1,358 images from 1,124 patients. Each image is accompanied by pixel-level segmentation masks for the fetal head and pubic symphysis, supporting a three-class image segmentation task (_background_, _pubic symphysis_, and _fetal head_). This version of the dataset was created for UltraBench to support the evaluation of ultrasound models for pubic symphysis and fetal head segmentation.

{: .important }
> When using this dataset, please ensure that you cite the original works:
>
> Chen, G., Bai, J., Ou, Z. et al. PSFHS: Intrapartum ultrasound image dataset for AI-based segmentation of pubic symphysis and fetal head. Sci Data 11, 436 (2024). https://doi-org.acces.bibl.ulaval.ca/10.1038/s41597-024-03266-4
>
> Bai, J. (2024). PSFHS [Data set]. In PSFHS: Intrapartum ultrasound image dataset for AI-based segmentation of pubic symphysis and fetal head (Version V1). 26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), Vancouver/Canada. Zenodo. https://doi.org/10.5281/zenodo.10969427

## Splits

The original dataset has been divided into `train` and `test` splits:

- `train`: 1,000 images in their original 256 x 256 px resolution.
- `test`: The remaining 358 images in their original 256 x 256 px resolution.
- `test.256`: The same 358 images in their original 256 x 256 px resolution.

Due to the lack of patient identifiers, there may be some patient overlap between the `train` and `test` splits.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.2`: The `train` split is capped at 1,000 images. The 86 images that no longer fit are assigned to the `test` split, which grows from 272 to 358 images.
- `0.1`: The original 80:20 split of 1,086 training and 272 test images.

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                                                |
| ------------ | ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `filename`   | `Value("string")` | Original image filenames (without the file extension).                                                     |
| `image`      | `Image`           | 3-channel ultrasound images.                                                                               |
| `scan_mask`  | `Image`           | Single-channel, binary ultrasound scan masks generated using morphological operations.                     |
| `psfh_mask`  | `Image`           | Single-channel, multi-class PS-FH segmentation mask (background = 0, pubic symphysis = 1, fetal head = 2). |

## Tasks

The PSFHS dataset supports one task:

- Multi-class PS-FH segmentation

## Usage

Load this dataset with `ultrabench.load_dataset`, which returns a Hugging Face [`DatasetDict`](https://huggingface.co/docs/datasets/index):

```python
import ultrabench

train_data = ultrabench.load_dataset(
    name="PSFHS",
    split="train",
    revision="0.4",
)
test_data = ultrabench.load_dataset(
    name="PSFHS",
    split="test.256",
    revision="0.4",
)
```
