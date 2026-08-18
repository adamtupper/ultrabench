#!/bin/bash
#SBATCH --job-name=classification-dinov3-convnext-base
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=3:00:00
#SBATCH --array=0-65%1
#
# Run downstream classification evaluation for DINOv3 ConvNeXt Base. Launches individual
# jobs for each task, evaluation method, and seed combo.
#
# Usage: sbatch scripts/slurm/run_classification_dinov3_convnext_base.sh

set -e

SEEDS=(42 43 44)
METHODS=(head_only full_finetuning)
TASKS=(
    aul_mass_classification
    busbra_histology_classification
    busbra_pathology_classification
    camus_classification
    covid_blues_classification
    fetal_planes_brain_plane_classification
    fetal_planes_plane_classification
    gbcu_classification
    masld_classification
    mmotu_classification
    tn3k_classification
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
MODEL="dinov3_convnext_base"
HEAD="linear_classifier"
BATCH_SIZE=32
ACCUM_GRAD=1

python ultrabench2/src/ultrabench/training/main_launch.py \
    system.output_dir=${OUTPUT_DIR} \
    system.checkpoint_dir=${CHECKPOINT_DIR} \
    system.save_last_checkpoint=false \
    system.data_loader_workers=${SLURM_CPUS_PER_TASK} \
    system.enable_progress_bar=false \
    model=${MODEL} \
    head=${HEAD} \
    task=${TASK} \
    mode=${METHOD} \
    optimization.seed=${SEED} \
    optimization.batch_size=${BATCH_SIZE} \
    optimization.accum_grad=${ACCUM_GRAD}
