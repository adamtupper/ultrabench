#!/bin/bash
#SBATCH --job-name=segmentation-deblurmim-convmae
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=3
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=3:00:00
#SBATCH --array=0-3
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
#
# Run downstream segmentation evaluation for DeblurMIM ConvMAE. Each array task
# is a worker that takes task, evaluation method, and seed combos from a shared
# queue until the job is nearly out of time, then submits a successor job for
# whatever is left.
#
# Usage: sbatch scripts/slurm/run_segmentation_deblurmim_convmae.sh
#        scripts/slurm/queue_status.sh segmentation_deblurmim_convmae
#        scripts/slurm/queue_retry.sh segmentation_deblurmim_convmae

set -e

EXPERIMENT="segmentation_deblurmim_convmae"

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

declare -a COMBOS
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        for task in "${TASKS[@]}"; do
            COMBOS+=("$seed $method $task")
        done
    done
done

MODEL="deblurmim_convmae"
HEAD="upsampler"
BATCH_SIZE=32
ACCUM_GRAD=1

# Run a single combo. $1 is the run directory reserved for this attempt, the
# remaining arguments are the fields of the combo.
ub_run_one() {
    local run_dir="$1" seed="$2" method="$3" task="$4"

    python ultrabench/src/ultrabench/training/main_launch.py \
        system.output_dir=${OUTPUT_DIR} \
        system.checkpoint_dir=${CHECKPOINT_DIR} \
        hydra.run.dir="${run_dir}" \
        system.save_last_checkpoint=false \
        system.data_loader_workers=${SLURM_CPUS_PER_TASK} \
        system.enable_progress_bar=false \
        model=${MODEL} \
        head=${HEAD} \
        task=${task} \
        mode=${method} \
        optimization.seed=${seed} \
        optimization.batch_size=${BATCH_SIZE} \
        optimization.accum_grad=${ACCUM_GRAD}
}

source $project/ultrabench/scripts/slurm/run_queue.sh
ub_queue_init

source $project/ultrabench/scripts/slurm/setup_node.sh
ub_queue_run
