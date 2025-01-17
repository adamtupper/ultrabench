# UltraBench

UltraBench is a standardized benchmark of 14 different convex ultrasound image classification and semantic segmentation tasks (7 classification and 7 segmentation tasks) drawn from the 10 different publicly available datasets covering 11 regions of the body. This repository contains the processing scripts to transform raw datasets from their original sources into a standardized format for training and evaluating machine learning models.

This benchmark was published alongside the article [Revisiting Data Augmentation for Ultrasound Images](). If you use this benchmark in your research please cite this article and the articles for each of the included datasets. Instructions for how to do so are provided at the end.

<p align="center">
  <img src="figures/body_regions.svg" alt="Regions of the body covered by UltraBench"/>
</p>

## Motivation

A common limitation of studies on ultrasound image analysis using machine learning is a lack of evaluations across multiple domains. However, this is made difficult by a lack of ultrasound image analysis tasks in existing medical image analysis benchmarks, such as [MedMNIST](https://www.nature.com/articles/s41597-022-01721-8), [MedSegBench](https://www.nature.com/articles/s41597-024-04159-2) and the [Medical Segmentation Decathlon](https://www.nature.com/articles/s41467-022-30695-9). To address this issue, we created UltraBench. Our aim is to make it easier for researchers by standardizing the preprocessing of many publicly available ultrasound datasets and providing tools (e.g., dataset implementations for common deep learning frameworks) that (a) reduce the effort required to use them and (b) make it easier for researchers to compare results between papers.

## Installation

Install UltraBench using:

```bash
pip install .
```

## Getting Started

UltraBench is packaged as a commandline application. After installing, run
`ultrabench --help` to get started.

For more information, checkout the [Documentation](https://github.com/adamtupper/ultrabench/wiki) on the wiki!

## Current Datasets & Tasks

The following table contains the list of supported datasets and tasks. Click on the links to checkout the appropriate sections of the documentation or on the article/repository links to visit the original publications. We do not have the rights to republish the datasets, but links to the original articles and datasets are included here and in the [Documentation](https://github.com/adamtupper/ultrabench/wiki).

<!-- TODO: Add links to documentation -->

| Dataset                                                   | Region       | Classification Tasks             | Segmentation Tasks                          |
|-----------------------------------------------------------|--------------|----------------------------------|---------------------------------------------|
| Annotated Ultrasound Liver ([article](https://doi.org/10.1093/bib/bbac569), [data](https://zenodo.org/records/7272660))                      | Liver        | Liver mass classification        | Liver segmentation, Liver mass segmentation |
| Butterfly ([data](https://github.com/ButterflyNetwork/MITGrandHack2018))                                    | Multi-region | Region classification            |                                             |
| CAMUS ([article](https://ieeexplore.ieee.org/document/8649738/), [data](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html))                                           | Heart        | Image quality classification     | Cardiac region segmentation                 |
| Dataset of B-mode fatty liver ultrasound images ([article](https://link.springer.com/article/10.1007/s11548-018-1843-2), [data](https://zenodo.org/records/1009146)) | Liver        | NFLD classification              |                                             |
| Gallbladder Cancer Ultrasound (GBCU) ([article](https://ieeexplore.ieee.org/document/9879895), [data](https://gbc-iitd.github.io/data/gbcu))            | Gallbladder  | Gallbladder tumor classification |                                             |
| Multi-Modality Ovarian Tumor Ultrasound (MMOTU) ([article](http://arxiv.org/abs/2207.06799), [data](https://github.com/cv516Buaa/MMOTU_DS2Net)) | Ovaries      | Ovarian tumor classifcation      | Ovarian tumor segmentation                  |
| Open Kidney Ultrasound ([article](https://link.springer.com/chapter/10.1007/978-3-031-44521-7_15), [data](https://rsingla.ca/kidneyUS/))                          | Kidney       |                                  | Kidney capsule segmentation                 |
| Point-of-care Ultrasound (POCUS) ([article](http://dx.doi.org/10.3390/app11020672), [data](https://github.com/jannisborn/covid19_ultrasound/))                | Lung         | COVID-19 classification          |                                             |
| PSFHS ([article](https://www.nature.com/articles/s41597-024-03266-4), [data](https://zenodo.org/records/10969427))                                           | Fetus        |                                  | Fetal head and pubic symphysis segmentation |
| Stanford Thyroid ([data](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5))                             | Thyroid      |                                  | Thyroid nodule segmentation                 |

## Contributing

We would love for this benchmark to grow and flourish into a resource that anyone with an interest in machine learning for ultrasound analysis can pick up and use quickly and easily. Any help addressing bugs, contributing new datasets or tasks, or any other improvements are welcome and appreiciated! I only ask that you respect the community guidelines laid out in the `CODE_OF_CONDUCT.md`. For more information on how to contribute, checkout out the [Documentation](https://github.com/adamtupper/ultrabench/wiki).

To ensure that your code meets the style guidelines etc., make sure you install the optional development dependencies:

```bash
pip install '.[dev]'
nbstripout --install
pre-commit install
```

## Questions

If you have any questions, please open an issue or contact us via email.

## How to Cite UltraBench

If you use UltraBench in your research, please cite our article [Revisiting Data Augmentation for Ultrasound Images]() and the articles for each of the datasets. This ensures that the work of the authors of the original datasets is properly acknowledged, helps more people find the benchmark and encourages us to continue maintaining and improving it!

```
```