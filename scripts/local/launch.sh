#!/bin/bash

set -e

# Set GPU visibility for local runs.
export HF_HOME=/raid/aradhachandran/thyroid/code/UltraBench2o/hf_data
export CUDA_VISIBLE_DEVICES=3

# Model configuration.
MODEL="urfm"  # echocare, samus, ultrasam, urfm, usfm, usfmae
HEAD="linear_classifier"  # linear_classifier, segformer, setr_mla, upsampler
# CHECKPOINT_DIR="${REPO_ROOT}/baselines"

# Set directories.
OUTPUT_DIR="/raid/aradhachandran/thyroid/code/UltraBench2o/logs/hf_ptl_runs_fullft"

# Training mode
MODE="head_only"  # head_only, full_finetuning

# Dataset task sweep.
TASKS=(
	aul_mass_classification
    # butterfly_classification
    masld_classification
    gbcu_classification
    mmotu_classification
	aul_mass_segmentation
    aul_liver_segmentation
    mmotu_segmentation
    open_kidney_capsule_segmentation
    open_kidney_regions_segmentation
    psfhs_segmentation
)

# Seeds for reproducibility.
SEEDS=(42 43 44)

# Optional data settings.
IMAGE_SIZE=256

BATCH_SIZE=32
ACCUM_GRAD=1
MAX_STEPS=10000
VAL_CHECK_INTERVAL=100
PATIENCE=null
DATA_LOADER_WORKERS=8

declare -a combos
for seed in "${SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
        combos+=("$shot_pct $seed $task")
    done
done

for combo in "${combos[@]}"; do
	read -r SEED TASK <<< "${combo}"

	python main_launch.py -m \
		system.output_dir=${OUTPUT_DIR} \
		system.data_loader_workers=${DATA_LOADER_WORKERS} \
		model=${MODEL} \
		head=${HEAD} \
		task=${TASK} \
		mode=${MODE} \
		data.image_size=${IMAGE_SIZE} \
		optimization.seed=${SEED} \
		optimization.max_steps=${MAX_STEPS} \
  		optimization.val_check_interval=${VAL_CHECK_INTERVAL} \
		optimization.patience=${PATIENCE} \
		optimization.batch_size=${BATCH_SIZE} \
		optimization.accum_grad=${ACCUM_GRAD}
done

echo "All runs completed!"
