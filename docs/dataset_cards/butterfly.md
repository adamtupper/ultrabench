# Butterfly Dataset

> [!CAUTION]
> This dataset must not be made public because officially it may only be used may be used for the MIT GrandHack 2018. See the [GitHub repository](https://github.com/ButterflyNetwork/MITGrandHack2018) for more details.

The Butterfly dataset is a multi-region ultrasound image dataset with 34,325 training and validation images, and 6,751 test images. The images are labeled according to the organ in the image (one of nine classes):

- Morison’s pouch
- Bladder
- Heart (PLAX view)
- Heart (4-chamber view)
- Heart (2-chamber view)
- IVC
- Carotid artery
- Lungs
- Thyroid

This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for organ classification.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original source:
>
> Butterfly Network. ButterflyNetwork/MITGrandHack2018, 2018. URL https://github.com/ButterflyNetwork/MITGrandHack2018

## Splits

The dataset is split according to the original `train` and `test` splits:

- `train`: The 34,325 training and validation images in their original sizes.
- `test`: The 6,751 test images in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                                                               |
| ------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `filename`   | `Value("string")` | Original image filenames prefixed by the patient ID and label (without the file extension).                               |
| `patient`    | `Value("int32")`  | Patient ID.                                                                                                               |
| `label`      | `ClassLabel`      | Organ labels (carotid = 0, 2ch = 1, lungs = 2, ivc = 3, 4ch = 4, bladder = 5, thyroid = 6, plax = 7, morisons_pouch = 8). |
| `image`      | `Image`           | 3-channel ultrasound images.                                                                                              |
| `scan_mask`  | `Image`           | Single-channel, binary ultrasound scan masks generated using morphological operations.                                    |

## Tasks

The Butterfly dataset supports one task:

- Multi-class organ classification

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/Butterfly", split="train")
test_data = datasets.load_dataset("UltraBench/Butterfly", split="test")
```
