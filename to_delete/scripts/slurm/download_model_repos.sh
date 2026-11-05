#!/bin/bash

# Download the repos for models that are loaded via PyTorch Hub (e.g., DINOv3).
#
# Usage: bash scripts/slurm/download_model_repos.sh

cd $project

git clone org-16943930@github.com:facebookresearch/dinov3.git models/ultrabench/dinov3-repo
