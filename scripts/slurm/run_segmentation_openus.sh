#!/bin/bash
#SBATCH --job-name=segmentation-openus
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=3:00:00
#SBATCH --array=0-59%1
#
# Run downstream segmentation evaluation for OpenUS. Launches individual
# jobs for each task, evaluation method, and seed combo.
#
# Usage: sbatch scripts/slurm/run_segmentation_openus.sh

set -e

SEEDS=(42 43 44)
METHODS=(head_only full_finetuning)
TASKS=(
    aul_liver_segmentation
    aul_mass_segmentation
    busbra_segmentation
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
MODEL="openus"
KERNEL_PATH="$project/hf-kernels/kernels--kernels-community--mamba-ssm/snapshots/a39ff24c08103278583f168091409653ada4c292/build/torch210-cxx11-cu126-x86_64-linux"
HEAD="segformer"
BATCH_SIZE=16
ACCUM_GRAD=2

python ultrabench2/src/ultrabench/training/main_launch.py \
    system.output_dir=${OUTPUT_DIR} \
    system.checkpoint_dir=${CHECKPOINT_DIR} \
    system.save_last_checkpoint=false \
    system.data_loader_workers=${SLURM_CPUS_PER_TASK} \
    system.enable_progress_bar=false \
    model=${MODEL} \
    model.selective_scan_kernel_path=${KERNEL_PATH} \
    head=${HEAD} \
    task=${TASK} \
    mode=${METHOD} \
    optimization.seed=${SEED} \
    optimization.batch_size=${BATCH_SIZE} \
    optimization.accum_grad=${ACCUM_GRAD}
