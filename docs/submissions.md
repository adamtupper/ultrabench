---
title: Submissions
nav_order: 5
---

# Submissions

Add a model to the [leaderboard]({% link index.md %}#leaderboard) by opening a pull request against `main`. A maintainer reviews every submission before merging it.

## Produce the scores

Produce the scores with the evaluation layer, following [the evaluation protocol]({% link evaluation_protocol.md %}):

1. Train the model on every task of the benchmark under the full fine-tuning paradigm, with five seeds per task. A category score covers all of the tasks in that category, so a missing task leaves the category unscored.
2. Write the test split predictions of each run to a prediction CSV, and lay the CSVs out as one directory per task. See [the evaluation documentation]({% link evaluation.md %}) for the format.
3. Score the submission:

   ```bash
   python -m ultrabench.evaluate.protocol predictions=<your directory>
   ```

   It writes the per-task and per-category scores with their standard errors, and prints the leaderboard row to add to the leaderboard data.

## Add your row

The leaderboard is built from one file, [`docs/_data/leaderboard.csv`](https://github.com/adamtupper/ultrabench/blob/main/docs/_data/leaderboard.csv). Your pull request adds one row to it. The site computes the ranking, the averages and the bold best scores, so the position of your row in the file does not matter.

The scorer prints the row with the scores filled in and placeholders for the rest:

```text
<model>,<url>,<release_date>,<params>,0.6450,0.0213,0.7571,0.0234,0.7821,0.0078,0.8219,0.0072
```

Replace the placeholders:

| Column | Description |
|--------|-------------|
| `model` | The model's display name, such as `URFM-L`. |
| `url` | The public repository for the model, so that readers reach the weights and the code that produced the scores. |
| `release_date` | The date the paper describing the model first appeared on arXiv, as `YYYY-MM-DD`. |
| `params` | The number of trainable parameters in the backbone, excluding the task head, as an integer. |

The remaining columns hold the mean and the standard error of each category score, to four decimal places, in this order: pathological classification, non-pathological classification, pathological segmentation, and anatomical segmentation.

## Describe the model

Describe how you trained the model in the pull request, and state which benchmark datasets, if any, appear in its pretraining data. A reviewer checks the row against the linked repository and asks for the per-task scores where a number needs verifying.
