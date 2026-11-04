# COVID-BLUES Dataset

> [!CAUTION]
> This dataset must not be made public as it is licensed under a CC-BY-NC-ND-4.0 license.

COVID-BLUES ([GitHub](https://github.com/NinaWie/COVID-BLUES), [Hugging Face](https://huggingface.co/datasets/jannisborn/COVID-BLUES)) is a dataset of lung ultrasound 362 videos from 63 suspected COVID-19 patients (30 negative and 33 positive cases). There are a total of 31,746 frames across the videos. This version of the dataset was created for UltraBench to support the evaluation of ultrasound models for COVID-19 classification.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original article:
>
> Wiedemann, N., Boer, D. de K., Richter, M., van de Weijer, S., Buhre, C., Eggert, F. A. M., Aarnoudse, S., Grevendonk, L., Röber, S., Remie, C. M. E., Buhre, W., Henry, R., & Born, J. (2025). COVID-BLUeS - a Prospective Study on the Value of AI in Lung Ultrasound Analysis. IEEE Journal of Biomedical and Health Informatics, 29(9), 6301–6310. https://doi.org/10.1109/JBHI.2025.3543686


## Splits

The dataset is split by patient into `train` and `test` splits:

- `train`: The frames for 43 patients in their original 350 x 350 px resolution.
- `test`: The frames for the remaining 20 patients in their original 350 x 350 px resolution.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.3`: The ground-truth CSV gains a `sequence_id` column so that COVID-19 classification is evaluated per patient. The images and splits are unchanged.
- `0.1`: The initial UltraBench v2 splits.

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                            |
| ------------ | ----------------- | -------------------------------------------------------------------------------------- |
| `filename`   | `Value("string")` | Original video filenames suffixed by the frame number (without the file extension)     |
| `video`      | `Value("string")` | Original video filename (without the file extension)                                   |
| `frame`      | `Value("int32")`  | Frame number                                                                           | 
| `patient`    | `Value("int32")`  | Patient ID                                                                             |
| `blue_point` | `Value("string")` | BLUE point (L1, L2, L3, R1, R2, or R3)                                                 |
| `covid19`    | `ClassLabel`      | COVID-19 PCR test result (negative = 0, positive = 1)                                  |
| `image`      | `Image`           | 3-channel ultrasound image                                                             |
| `scan_mask`  | `Image`           | Single-channel, binary ultrasound scan mask generated using morphological operations   |

## Tasks

The COVID-BLUES dataset supports one task:

- COVID-19 classification

## Evaluation

The `covid19` label is a PCR test result for the whole patient, not a property of any single frame or video, so the task is evaluated per **patient**. All frames of all videos belonging to a patient average into one prediction, leaving 20 predictions on the `test` split.

Generate one prediction per patient by averaging the predicted probabilities over every frame from that patient, then taking the argmax. The ground-truth CSV carries a `sequence_id` column (the patient ID) that identifies which frames belong together, so the predictions CSV stays per image:

```bash
python -m ultrabench.evaluate task=classification aggregate_by=sequence_id \
  ground_truth=test-ground-truth-classification.csv \
  predictions=predictions.csv
```

The `test` split holds only 20 patients, so the balanced accuracy moves in coarse steps. Average over several seeds before comparing models.

See [the evaluation documentation](../evaluation.md) for the CSV format.

## Usage

Load this dataset with `ultrabench.load_dataset`, which returns a Hugging Face [`DatasetDict`](https://huggingface.co/docs/datasets/index):

```python
import ultrabench

train_data = ultrabench.load_dataset(
    name="COVID-BLUES",
    split="train",
    revision="0.4",
)
test_data = ultrabench.load_dataset(
    name="COVID-BLUES",
    split="test.256",
    revision="0.4",
)
```

The first call downloads the original files and preprocesses them on your machine, which takes a few minutes. The result is cached in `$ULTRABENCH_CACHE`, or `~/.ultrabench` if that is not set, so later calls are immediate.
