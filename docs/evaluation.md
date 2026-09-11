# Evaluation

The evaluation layer is framework independent. It reads two CSV files — the ground truth published with each dataset and the predictions produced by your model — and writes the performance metrics. Nothing about how you trained the model matters, so you can benchmark a model written in any framework.

## Running an evaluation

```bash
python -m ultrabench2.evaluate task=<classification|segmentation> \
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

Each dataset publishes its ground truth alongside the images on the Hugging Face Hub. Evaluation is test-only, so the files cover the two test splits: `test-ground-truth-<task>.csv` describes the original-resolution `test` split, and `test.256-ground-truth-<task>.csv` describes the `test.256` split. A dataset with subsets prefixes the subset name, as in `segmentation-test.256-ground-truth-segmentation.csv`.

Score your predictions against the split you predicted on. A segmentation CSV encodes its masks at the resolution of its own split, so scoring `test.256` predictions against the full-resolution ground truth gives a near-zero Dice score rather than an error.

| Column | Task | Description |
| ------ | ---- | ----------- |
| `image_id` | Both | The image's `filename` value in the dataset. |
| `label` | Classification | The integer class index. |
| `sequence_id` | Classification | The sequence the image belongs to. Present only for tasks with sequence-level labels. |
| `class_id` | Segmentation | The foreground class index. Background (0) is omitted. |
| `encoded_pixels` | Segmentation | The run-length encoded binary mask for that class, or an empty string. |

### Downloading the ground truth

Download the CSV of the split you predicted on, pinning the revision the task configuration pins:

```bash
hf download UltraBench/TN3K test.256-ground-truth-segmentation.csv \
  --repo-type dataset --revision 0.4 --local-dir ground_truth
```

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

Run-length encoding is Kaggle style: 1-indexed, column-major, and space separated pairs of start position and run length. An all-zero mask encodes to an empty string. Use `ultrabench2.core.rle.mask_to_rle` and `rle_to_mask` to convert.

## Sequence-level tasks

Some datasets label a whole sequence rather than an individual image. CAMUS grades the quality of a `(patient, view)` acquisition, COVID-BLUES records a PCR result per patient, and MASLD records a diagnosis per patient. Scoring these per image would weight each sequence by its frame count and overstate the size of the test set, so they are scored on one prediction per sequence.

| Task | Aggregation unit | Test sequences |
| ---- | ---------------- | -------------- |
| CAMUS image quality | `(patient, view)` | 100 |
| COVID-BLUES COVID-19 | `patient` | 20 |
| MASLD | `patient` | 11 |

Pool the predictions by averaging the predicted probabilities over every image in the sequence, then take the argmax. Pass `aggregate_by=sequence_id` to do this:

```bash
python -m ultrabench2.evaluate task=classification aggregate_by=sequence_id \
  ground_truth=test-ground-truth-classification.csv \
  predictions=predictions.csv
```

Your predictions CSV stays per image. The evaluator reads the grouping from the ground truth's `sequence_id` column, so you do not need to know how the sequences are laid out. You do need the `prob_{k}` columns, since probabilities are averaged rather than votes counted.

The training layer reports the same metric inline. Set `sequence_by` in the task configuration (`src/ultrabench2/training/config/task/*.yaml`) to the dataset columns that identify a sequence, and `test_metrics.csv` gains `seq_`-prefixed metrics next to the per-image ones.

## Scoring a whole submission

Scoring one task at a time gives a number without an error bar, and the leaderboard reports four category scores with one. The submission scorer does the whole job: it reads every run of every task, scores each of them, resamples the test sets, and writes the per-task and the per-category tables. It implements [the evaluation protocol](evaluation_protocol.md) and reproduces the numbers the benchmark publishes.

Lay the submission out as one directory per task, each holding one predictions CSV per training run:

```text
runs/
  aul_liver_segmentation/
    seed_0.csv
    ...
    seed_4.csv
  aul_mass_classification/
    ...
```

The CSVs are the ones described above, and their names are not interpreted. Score them with:

```bash
python -m ultrabench2.evaluate.protocol predictions=runs
```

Two CSVs are written to the Hydra run directory, and both are printed:

| File | Contents |
| ---- | -------- |
| `task_scores.csv` | One row per task: its metric, the cohort it is resampled over, the mean over the runs, and the error bar. |
| `category_scores.csv` | One row per category the submission covers completely. |

A submission covering all four categories also prints its leaderboard row, ready to paste into the table in the README.

### Configuration

| Setting | Default | Description |
| ------- | ------- | ----------- |
| `predictions` | required | The root of the submission. |
| `tasks` | every task found | The tasks to score. |
| `runs_per_task` | 5 | The number of runs each task must have. Set it to `null` to score a task with any number of runs while you iterate. |
| `n_resamples` | 2000 | The bootstrap replications drawn per source dataset. Lower it only for a quick check; the reported error bar needs all of them. |
| `confidence` | 1.96 | The normal quantile the interval spans. |
| `cache_dir` | `$ULTRABENCH_CACHE`, then `~/.ultrabench` | Where the per-dataset patient maps are cached. |
| `output_dir` | the Hydra run directory | Where the two CSVs are written. |

### The error bar

The reported interval is the mean plus or minus 1.96 SE, and the standard error combines the variability across your training runs with the sampling noise of the test set. Both terms are reported beside it. `sigma_run` shrinks as you train more runs. `sigma_boot` is a property of the benchmark's test cohorts, and no amount of training moves it.

The second term comes from 2000 bootstrap replications per source dataset, drawn from a seed frozen at release rather than from anything in your files, so your submission and the published results are resampled identically without either side exchanging any files. Where a dataset carries a per-image patient identifier the resample is taken over patients, since frames from one patient are correlated; the scorer reports which cohort it used in the `unit` column.

A category score is reported only when the submission covers every one of the category's tasks, since the score is the unweighted mean over all of them. Scoring part of the benchmark gives the same per-task numbers as scoring all of it: a replication resamples the test set of a source *dataset*, so a dataset's cohort is built from every benchmark task that uses it, whether or not you submitted those tasks. That costs the ground truth of the tasks you left out, never their predictions.

### What it downloads

The ground truth of every task it scores, and, for the datasets resampled over patients (BUS-BRA, CAMUS, COVID-BLUES, FetalPlanes, MASLD and STMUS), the test split itself. A split is read for its `patient` column alone and the result is cached as a CSV under `cache_dir`, so only the first run pays for it.

## Metrics

| Task | Metrics |
| ---- | ------- |
| Classification | Balanced accuracy. |
| Segmentation | Positive Dice over the foreground classes, and the mean Dice of each class. |

### The positive Dice convention

The positive Dice score excludes the background class, and follows two further conventions. Both match `torchmetrics.segmentation.DiceScore(include_background=False)`, which the training layer uses for the `positive_dice` column of `test_metrics.csv`, so the evaluator reproduces that column exactly.

- A class that is empty in both the ground truth and the prediction is undefined, and drops out of the score. Scoring it 1.0 instead rewards a model for predicting nothing on an image the class is absent from, which inflates the score on tasks where many images hold no instance of a class. An image whose every foreground class is undefined drops out of the score entirely.
- The score averages the defined foreground classes of an image first, and the images second. Averaging per class first gives a different number, since the undefined pairs leave the per-class groups unequal in size.

The undefined pairs are still written to `per_image_metrics.csv`, with an empty `dice` field.
