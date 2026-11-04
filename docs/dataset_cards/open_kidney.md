# Open Kidney Dataset

> [!CAUTION]
> This dataset must not be made public as it is licensed under a Research Usage Agreement. See [here](https://rsingla.ca/kidneyUS/) for more details.

The Open Kidney Ultrasound dataset consists of 514 B-mode kidney ultrasound images, each from a distinct patient. Each image has two segmentation masks for two separate semantic segmentation tasks: kidney capsule and fine-grain kidney regions segmentation. The regions segmentation task is very difficult due to the limited amount of data. This version of the dataset was created for UltraBench to support the evaluation of ultrasound models for kidney segmentation.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original source:
>
> Singla, R. et al. (2023). The Open Kidney Ultrasound Data Set. In: Simplifying Medical Ultrasound. ASMUS 2023. Lecture Notes in Computer Science, vol 14337. Springer, Cham. https://doi.org/10.1007/978-3-031-44521-7_15

## Splits

The dataset is split into to the original `train` and `test` splits using an 80:20 split. To minimize distribution drift between the splits, we stratify by view (*transverse*, *longitudinal*, and *other*):

- `train`: 80% of images in their original resolution.
- `test`: The remaining 20% of images in their original resolution.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

## Features

All splits have the following features:

| Feature        | Feature Type      | Description                                                                                                                 |
| -------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `filename`     | `Value("string")` | Original image filenames (without the file extension).                                                                      |
| `transplant`   | `Value("bool")`   | Whether or not the kidney is transplanted.                                                                                  |
| `view`         | `ClassLabel`      | The view of the kidney (longitudinal = 0, transverse = 1, other =2).                                                        |
| `quality`      | `Value("string")` | The quality of the image ((unsatisfactory, poor, fair, or good).                                                            |
| `image`        | `Image`           | 3-channel ultrasound images.                                                                                                |
| `capsule_mask` | `Image`           | Single-channel, binary mask of the kidney capsule.                                                                          |
| `regions_mask` | `Image`           | Single-channel, multi-class mask of the kidney regions (background = 0, central echo complex = 1, medulla = 2, cortex = 3). |
| `scan_mask`    | `Image`           | Single-channel, binary ultrasound scan masks generated using morphological operations.                                      |

## Tasks

The Open Kidney dataset supports two tasks:

- Binary kidney capsule segmentation
- Multi-class kidney regions segmentation

## Usage

Load this dataset with `ultrabench.load_dataset`, which returns a Hugging Face [`DatasetDict`](https://huggingface.co/docs/datasets/index):

```python
import ultrabench

train_data = ultrabench.load_dataset(
    name="OpenKidney",
    path="path/to/raw/data",
    split="train",
)
test_data = ultrabench.load_dataset(
    name="OpenKidney",
    path="path/to/raw/data",
    split="test.256",
)
```

OpenKidney cannot be republished, so download it from [its original source](https://rsingla.ca/kidneyUS/) first and pass the directory as `path`. The first call preprocesses it and caches the result in `$ULTRABENCH_CACHE`, or `~/.ultrabench` if that is not set, so later calls are immediate.
