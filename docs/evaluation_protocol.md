---
title: Evaluation Protocol
parent: Evaluation
nav_order: 1
math: true
---

# Evaluation Protocol

Performance on classification and segmentation tasks is measured using macro F1 and positive Dice scores, respectively. The positive Dice score excludes the background class since the object(s) of interest in the images typically represent only a small fraction of the image and including the background would inflate results.

The individual task scores are aggregated into four category scores: anatomical segmentation, pathological segmentation, pathological classification, and non-pathological classification. This distinguishes a model's ability to segment anatomical structures (e.g., organs) and pathological structures (e.g., nodules), as anatomical structures tend to be larger and more clearly defined than pathological ones. It also separates diagnostic tasks from image quality and categorization tasks. For tasks with patient- or sequence-level labels, performance is measured per-patient or per-sequence. The table below lists the tasks in each category.

| Category | Tasks |
|----------|-------|
| Pathological classification | AUL liver mass, BUS-BRA histology, BUS-BRA pathology, COVID-BLUES, GBCU, MASLD, MMOTU, TN3K |
| Non-pathological classification | CAMUS image quality, Fetal Planes brain plane, Fetal Planes plane |
| Pathological segmentation | AUL liver mass, BUS-BRA lesion, MMOTU ovarian tumour, TN3K thyroid nodule |
| Anatomical segmentation | AUL liver, CAMUS cardiac regions, Open Kidney capsule, Open Kidney regions, PSFHS, STMUS |

To standardize the calculation of scores, we provide evaluation scripts that accept predictions in CSV format. This framework-agnostic interface frees researchers to develop their model with their chosen tools and frameworks (e.g., PyTorch or TensorFlow).

Each submission must provide test set predictions for $$S = 5$$ different seeds per task. Let $$Y_{k,r}$$ be the performance metric of task $$k$$ under run $$r$$, and let a category consist of $$K$$ tasks scored over $$S$$ runs. The reported category score is the unweighted mean of the run-averaged task scores:

$$\bar{Y} = \frac{1}{K}\sum_{k=1}^{K}\bar{Y}_k,\qquad \bar{Y}_k = \frac{1}{S}\sum_{r=1}^{S} Y_{k,r}.$$

Uncertainty is quantified using error bars, $$\bar{Y} \pm \mathrm{SE}(\bar{Y})$$, which combine two sources of noise: the variability across runs and the sampling noise in the test sets. Both are computed at the level of the category score rather than per task, so that correlations between tasks that share a test set (e.g., BUS-BRA pathology and histology classification) are not ignored. $$\mathrm{SE}(\bar{Y})$$ is calculated as

$$\mathrm{SE}(\bar{Y}) = \sqrt{\frac{\sigma_{\mathrm{run}}^{2}}{S} + \sigma_{\mathrm{boot}}^{2}},$$

where $$\sigma_{\mathrm{run}}$$ is the standard deviation across runs of the per-run category score, and $$\sigma_{\mathrm{boot}}$$ is the standard deviation of the category score across 2000 bootstrap replications. Each replication draws resamples of the test set of each source dataset, shared by every task built on that dataset, and recomputes the category score with each task's metric averaged over the five runs. Where a dataset provides per-image patient identifiers, resampling is done at the patient level, since frames from one patient are correlated.

The per-task error bars use the same construction with $$K = 1$$.

Because we fix the train/test split, confidence intervals are conditional this split. They quantify sampling variability of the test cohorts and stochasticity of training, not variability that would arise from a different split, different hyperparameter search, or a different source population.