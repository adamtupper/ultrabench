# Evaluation

The evaluation layer is framework independent. It reads two CSV files — the ground truth published with each dataset and the predictions produced by your model — and writes the performance metrics. Nothing about how you trained the model matters, so you can benchmark a model written in any framework.

## Running an evaluation

```bash
python -m ultrabench.evaluate task=<classification|segmentation> \
  ground_truth=<path to ground truth CSV> \
  predictions=<path to predictions CSV>
```

Results are written to the Hydra run directory:

| File | Task | Contents |
| ---- | ---- | -------- |
| `summary.csv` | Both | One row per metric, with `metric` and `value` columns. |
| `per_image_metrics.csv` | Segmentation | The Dice score of each (image, class) pair. |
| `per_sequence_predictions.csv` | Classification | The pooled prediction of each sequence, written only when `aggregate_by` is set. |

## Ground-truth CSVs

Each dataset publishes its ground truth alongside the images on the Hugging Face Hub. The files cover the `test` split only, since evaluation is test-only.

| Column | Task | Description |
| ------ | ---- | ----------- |
| `image_id` | Both | The image's `filename` value in the dataset. |
| `label` | Classification | The integer class index. |
| `sequence_id` | Classification | The sequence the image belongs to. Present only for tasks with sequence-level labels. |
| `class_id` | Segmentation | The foreground class index. Background (0) is omitted. |
| `encoded_pixels` | Segmentation | The run-length encoded binary mask for that class, or an empty string. |

## Prediction CSVs

Write one row per test image, using the same `image_id` values as the ground truth. Images missing from your predictions are logged as warnings and excluded from the metrics.

Classification:

| Column | Description |
| ------ | ----------- |
| `image_id` | The image identifier. |
| `label` | The predicted class index. |
| `prob_{k}` | The predicted probability of class `k`, one column per class. Required for sequence-level evaluation. |

```csv
image_id,label,prob_0,prob_1
41_0,0,0.82,0.18
41_1,1,0.34,0.66
```

Segmentation:

| Column | Description |
| ------ | ----------- |
| `image_id` | The image identifier. |
| `class_id` | The foreground class index. |
| `encoded_pixels` | The run-length encoded binary mask for that class. |

Run-length encoding is Kaggle style: 1-indexed, column-major, and space separated pairs of start position and run length. An all-zero mask encodes to an empty string. Use `ultrabench.core.rle.mask_to_rle` and `rle_to_mask` to convert.

## Sequence-level tasks

Some datasets label a whole sequence rather than an individual image. CAMUS grades the quality of a `(patient, view)` acquisition, COVID-BLUES records a PCR result per patient, and MASLD records a diagnosis per patient. Scoring these per image would weight each sequence by its frame count and overstate the size of the test set, so they are scored on one prediction per sequence.

| Task | Aggregation unit | Test sequences |
| ---- | ---------------- | -------------- |
| CAMUS image quality | `(patient, view)` | 100 |
| COVID-BLUES COVID-19 | `patient` | 20 |
| MASLD | `patient` | 11 |

Pool the predictions by averaging the predicted probabilities over every image in the sequence, then take the argmax. Pass `aggregate_by=sequence_id` to do this:

```bash
python -m ultrabench.evaluate task=classification aggregate_by=sequence_id \
  ground_truth=test-ground-truth-classification.csv \
  predictions=predictions.csv
```

Your predictions CSV stays per image. The evaluator reads the grouping from the ground truth's `sequence_id` column, so you do not need to know how the sequences are laid out. You do need the `prob_{k}` columns, since probabilities are averaged rather than votes counted.

The training layer reports the same metric inline. Set `sequence_by` in the task configuration (`src/ultrabench/training/config/task/*.yaml`) to the dataset columns that identify a sequence, and `test_metrics.csv` gains `seq_`-prefixed metrics next to the per-image ones.

## Metrics

| Task | Metrics |
| ---- | ------- |
| Classification | Balanced accuracy. |
| Segmentation | Macro Dice over the foreground classes, and the mean Dice of each class. |
