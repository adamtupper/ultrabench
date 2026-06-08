#!/bin/bash
#SBATCH --job-name=segmentation-usfmae
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=3:00:00
#SBATCH --array=0-53%1
#
# Run downstream segmentation evaluation for USF-MAE. Launches
# individual jobs for each task, evaluation method, and seed combo.
#
# Usage: sbatch scripts/slurm/run_segmentation_usfmae.sh

set -e

SEEDS=(42 43 44)
METHODS=(upsampler_probe full_finetune_seg)
TASKS=(
    aul_liver_segmentation
    aul_mass_segmentation
    camus_segmentation
    mmotu_segmentation
    open_kidney_capsule_segmentation
    open_kidney_regions_segmentation
    psfhs_segmentation
    stmus_segmentation
    tn3k_segmentation
)

declare -a combos
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        for task in "${TASKS[@]}"; do
            combos+=("$seed $method $task")
        done
    done
done

source $project/ultrabench2/scripts/slurm/setup_node.sh

cd $SLURM_TMPDIR

read -r SEED METHOD TASK <<< "${combos[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT_DIR="$project/models/ultrabench"
OUTPUT_DIR="$scratch/results/ultrabench"
MODEL="usfmae"
PREPROCESSING_FN="ultrasound_preprocessing"
IMAGE_SIZE=256
BATCH_SIZE=32
ACCUM_GRAD=1
EPOCHS=100
PATIENCE=null

python ultrabench2/src/ultrabench/training/main_launch.py \
    system.output_dir=${OUTPUT_DIR} \
    system.checkpoint_dir=${CHECKPOINT_DIR} \
    system.data_loader_workers=${SLURM_CPUS_PER_TASK} \
    system.enable_progress_bar=false \
    model=${MODEL} \
    task=${TASK} \
    "mode@cls_mode=none" \
    "mode@seg_mode=${METHOD}" \
    data.preprocessing_fn=${PREPROCESSING_FN} \
    data.image_size=${IMAGE_SIZE} \
    optimization.seed=${SEED} \
    optimization.num_epochs=${EPOCHS} \
    optimization.patience=${PATIENCE} \
    optimization.batch_size=${BATCH_SIZE} \
    optimization.accum_grad=${ACCUM_GRAD}
