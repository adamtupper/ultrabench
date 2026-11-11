---
title: Home
layout: home
nav_order: 1
leaderboard: true
description: >-
  UltraBench: a standardized benchmark of 21 convex ultrasound classification
  and segmentation tasks for evaluating ultrasound foundation models.
---

# UltraBench

UltraBench is a standardized benchmark of 21 convex ultrasound image classification and semantic segmentation tasks. The tasks are drawn from 12 publicly available datasets covering 10 anatomical regions. It provides the datasets, the backbones and task heads of the evaluated models, and an evaluation layer that scores the predictions of a model trained in any framework.

```bash
pip install ultrabench
```

[Get started]({% link getting_started.md %}){: .btn .btn-primary .mr-2 }
[Load a dataset]({% link dataset_format.md %}){: .btn .mr-2 }
[Submit a model]({% link submissions.md %}){: .btn }

## Leaderboard

The leaderboard ranks the evaluated ultrasound foundation models by the average of their four category scores. Each category score is the mean over the tasks in that category, measured with macro F1 for classification and positive Dice for segmentation, under the full fine-tuning paradigm. The best score in each category is bold. Click a column header to sort the table by it.

{% include leaderboard_table.html %}

Read the scores of a model pretrained on data that overlaps a category's tasks with care. See [the evaluation protocol]({% link evaluation_protocol.md %}) for the task categories, the metrics, and how the scores and their standard errors are computed.

### Performance over time

Each point is a model, placed by the date its paper first appeared on arXiv and by the average of its four category scores. The dashed line connects each model that beats every model released before it. Hover over a point to see its category scores.

<script>
  window.leaderboardData = {{ site.data.leaderboard | jsonify }};
</script>
<div class="chart-container">
  <canvas id="release-chart" role="img" aria-label="Scatter plot of the average category score of each model against its release date. The leaderboard table above holds the same data."></canvas>
</div>
<p class="chart-caption">Point area is proportional to the number of backbone parameters (<span id="release-chart-params">see the Params column</span>).</p>

## Datasets and tasks

The **Access** column says how each dataset reaches you. *Processed* datasets are published ready to use. *Raw* datasets may be redistributed unmodified but not preprocessed, so `load_dataset` downloads the original files and preprocesses them on your machine. *Local* datasets may not be redistributed at all, so you download them from the original source and pass the directory as `path`.

| Dataset | Access | Region | Classification Tasks | Segmentation Tasks |
|---------|--------|--------|----------------------|--------------------|
| [Annotated Ultrasound Liver]({% link dataset_cards/aul.md %}) ([article](https://doi.org/10.1093/bib/bbac569), [data](https://zenodo.org/records/7272660)) | Processed | Liver | Liver mass classification | Liver segmentation, Liver mass segmentation |
| [BUS-BRA]({% link dataset_cards/busbra.md %}) ([article](https://doi.org/10.1002/mp.16812), [data](https://zenodo.org/records/8231412)) | Processed | Breast | Histology classification, Pathology classification | Breast lesion segmentation |
| [CAMUS]({% link dataset_cards/camus.md %}) ([article](https://ieeexplore.ieee.org/document/8649738/), [data](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)) | Processed | Heart | Image quality classification | Cardiac region segmentation |
| [COVID-BLUES]({% link dataset_cards/covid_blues.md %}) ([article](https://doi.org/10.1109/TUFFC.2025.3535879), [data](https://github.com/NinaWie/COVID-BLUES)) | Raw | Lungs | COVID-19 classification | |
| [Fetal Planes]({% link dataset_cards/fetal_planes.md %}) ([article](https://www.nature.com/articles/s41598-020-67076-5), [data](https://zenodo.org/records/3904280)) | Processed | Fetus | Plane classification, Brain plane classification | |
| [Gallbladder Cancer Ultrasound]({% link dataset_cards/gbcu.md %}) ([article](https://ieeexplore.ieee.org/document/9879895), [data](https://gbc-iitd.github.io/data/gbcu)) | Local | Gallbladder | Gallbladder tumour classification | |
| [MASLD]({% link dataset_cards/masld.md %}) ([article](https://link.springer.com/article/10.1007/s11548-018-1843-2), [data](https://zenodo.org/records/1009146)) | Processed | Liver | Steatosis classification | |
| [MMOTU]({% link dataset_cards/mmotu.md %}) ([article](http://arxiv.org/abs/2207.06799), [data](https://github.com/cv516Buaa/MMOTU_DS2Net)) | Raw | Ovaries | Ovarian tumour classification | Ovarian tumour segmentation |
| [Open Kidney Ultrasound]({% link dataset_cards/open_kidney.md %}) ([article](https://link.springer.com/chapter/10.1007/978-3-031-44521-7_15), [data](https://rsingla.ca/kidneyUS/)) | Local | Kidney | | Kidney capsule segmentation, Kidney region segmentation |
| [PSFHS]({% link dataset_cards/psfhs.md %}) ([article](https://www.nature.com/articles/s41597-024-03266-4), [data](https://zenodo.org/records/10969427)) | Processed | Fetus | | Fetal head and pubic symphysis segmentation |
| [STMUS]({% link dataset_cards/stmus.md %}) ([data](https://data.mendeley.com/datasets/3jykz7wz8d)) | Processed | Muscle | | Muscle segmentation |
| [TN3K]({% link dataset_cards/tn3k.md %}) ([article](https://doi.org/10.1016/j.compbiomed.2022.106389), [data](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)) | Processed | Thyroid | Thyroid nodule classification | Thyroid nodule segmentation |
