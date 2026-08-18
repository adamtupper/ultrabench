#!/bin/bash

# Download each dataset to PROJECT using the HF CLI.
#
# Usage: bash scripts/slurm/download_datasets.sh
#
# Expects the HF_TOKEN environment variable to be set to a valid Hugging
# Face access token with read access to the UltraBench org.

module load arrow/21.0.0 python/3.12
virtualenv --no-download ~/venvs/hf
source ~/venvs/hf/bin/activate
pip install --no-index huggingface_hub datasets

hf auth login --token $HF_TOKEN

cd $project/ultrabench2

python scripts/slurm/download_dataset.py --name UltraBench/AUL --revision 0.1 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/BUS-BRA --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/CAMUS --config classification --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/CAMUS --config segmentation --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/COVID-BLUES --revision 0.1 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/FetalPlanes --config fetal --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/FetalPlanes --config fetal_brain --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/GBCU --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/MASLD --revision 0.1 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/MMOTU-2D --revision 0.1 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/OpenKidney --revision 0.1 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/PSFHS --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/STMUS --revision 0.2 --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/TN3K --revision 0.2 --cache_dir $project/data/hf/datasets

# Deprecated datasets:
# python scripts/slurm/download_dataset.py --name UltraBench/Butterfly --revision 0.1 --cache_dir $project/data/hf/datasets
# python scripts/slurm/download_dataset.py --name UltraBench/POCUS --revision 0.1 --cache_dir $project/data/hf/datasets
# python scripts/slurm/download_dataset.py --name UltraBench/StanfordThyroid --revision 0.1 --cache_dir $project/data/hf/datasets


deactivate
rm -r ~/venvs/hf
module purge python/3.12
