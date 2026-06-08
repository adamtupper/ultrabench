#!/bin/bash
#SBATCH --job-name=classification-samus
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=42:00:00
#SBATCH --array=0-71%1
#
# Run downstream classification evaluation for SAMUS. Launches
# individual jobs for each task, evaluation method, and seed combo.
#
# Usage: sbatch scripts/slurm/run_classification_samus.sh

set -e

SEEDS=(42 43 44)
METHODS=(linear_probe full_finetune_cls)
TASKS=(
    aul_mass_classification
    busbra_histology_classification
    busbra_pathology_classification
    butterfly_classification
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
MODEL="samus"
PREPROCESSING_FN="ultrasound_preprocessing"
IMAGE_SIZE=256
BATCH_SIZE=16
ACCUM_GRAD=2
EPOCHS=100
PATIENCE=null

python ultrabench2/src/ultrabench/training/main_launch.py \
    system.output_dir=${OUTPUT_DIR} \
    system.checkpoint_dir=${CHECKPOINT_DIR} \
    system.data_loader_workers=${SLURM_CPUS_PER_TASK} \
    system.enable_progress_bar=false \
    model=${MODEL} \
    task=${TASK} \
    "mode@cls_mode=${METHOD}" \
    "mode@seg_mode=none" \
    data.preprocessing_fn=${PREPROCESSING_FN} \
    data.image_size=${IMAGE_SIZE} \
    optimization.seed=${SEED} \
    optimization.num_epochs=${EPOCHS} \
    optimization.patience=${PATIENCE} \
    optimization.batch_size=${BATCH_SIZE} \
    optimization.accum_grad=${ACCUM_GRAD}
