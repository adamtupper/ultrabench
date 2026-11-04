# MASLD Dataset

The MALSD dataset consists of 550 liver ultrasound images from 55 patients from the *Dataset of B-mode fatty liver ultrasound images*. 38 of the 55 patients are diagnosed with Metabolic dysfunction-associated steatotic liver disease (MASLD), formerly known as nonalcoholic fatty liver disease (NAFLD). MASLD is defined as >5 % of hepatocytes having fatty infiltration. This version of the dataset was created for UltraBench to support the evaluation of ultrasound models for MALSD classification.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> M. Byra, G. Styczynski, C. Szmigielski, P. Kalinowski. Ł. Michałowski. R. Paluszkiewicz. B. Ziarkiewicz-Wróblewska, K. Zieniewicz. P. Sobieraj, A. Nowicki. Transfer learning with deep convolutional neural network for liver steatosis assessment in ultrasound images. International Journal of Computer Assisted Radiology and Surgery, 2018. DOI: 10.1007/s11548-018-1843-2

## Splits

The patients of original dataset have been divided into `train` and `test` splits using a 80:20 split:

- `train`: 80% of the patients with the images in their original sizes.
- `test`: The remaining 20% of patients with the images in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.3`: The ground-truth CSV gains a `sequence_id` column so that MASLD classification is evaluated per patient. The images and splits are unchanged.
- `0.1`: The initial UltraBench v2 splits.

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

## Evaluation

The MASLD diagnosis describes the patient, not an individual image, so the task is evaluated per **patient**. All 10 images of a patient average into one prediction, leaving 11 predictions on the `test` split.

Generate one prediction per patient by averaging the predicted probabilities over the patient's images, then taking the argmax. The ground-truth CSV carries a `sequence_id` column (the patient ID) that identifies which images belong together, so the predictions CSV stays per image:

```bash
python -m ultrabench.evaluate task=classification aggregate_by=sequence_id \
  ground_truth=test-ground-truth-classification.csv \
  predictions=predictions.csv
```

The `test` split holds only 11 patients, so the balanced accuracy moves in coarse steps. Average over several seeds before comparing models.

See [the evaluation documentation](../evaluation.md) for the CSV format.

## Usage

Load this dataset with `ultrabench.load_dataset`, which returns a Hugging Face [`DatasetDict`](https://huggingface.co/docs/datasets/index):

```python
import ultrabench

train_data = ultrabench.load_dataset(
    name="MASLD",
    split="train",
    revision="0.4",
)
test_data = ultrabench.load_dataset(
    name="MASLD",
    split="test.256",
    revision="0.4",
)
```