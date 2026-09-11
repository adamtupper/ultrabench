#!/bin/bash

# Local downstream evaluation sweep script using Hydra multi-run

# Model (backbone + head)
MODEL="urfm"
HEAD="linear_classifier"  # linear_classifier, segformer, setr_mla, upsampler

# Set directories
DATA_DIR="/scratch/data"
OUTPUT_DIR="/scratch/results/ultrabench-tmp"
CHECKPOINT_DIR="/project/models/ultrabench"
DATA_LOADER_WORKERS=4

# Training mode
MODE="head_only" # head_only or full_finetuning

TASK="aul_mass_classification"

# Seeds for reproducibility
SEEDS="42" # 42,43,44,45,46

# Shot percentages (training data fractions)
SHOT_PCTS="1.0"

# Data settings
IMAGE_SIZE=256

BATCH_SIZE=32
ACCUM_GRAD=1
MAX_STEPS=100
VAL_CHECK_INTERVAL=100
PATIENCE=null


# Run Hydra multi-run sweep
python src/ultrabench2/training/main_launch.py \
  model=${MODEL} \
  head=${HEAD} \
  task=${TASK} \
  data.shot_pct=${SHOT_PCTS} \
  mode=${MODE} \
  data.image_size=${IMAGE_SIZE} \
  optimization.seed=${SEEDS} \
  optimization.max_steps=${MAX_STEPS} \
  optimization.val_check_interval=${VAL_CHECK_INTERVAL} \
  optimization.patience=${PATIENCE} \
  optimization.batch_size=${BATCH_SIZE} \
  optimization.accum_grad=${ACCUM_GRAD} \
  system.data_dir=${DATA_DIR} \
  system.output_dir=${OUTPUT_DIR} \
  system.checkpoint_dir=${CHECKPOINT_DIR} \
  system.data_loader_workers=$DATA_LOADER_WORKERS \
  system.log_every_n_steps=50 \
  system.enable_progress_bar=true \
  --multirun

echo "All runs completed!"
