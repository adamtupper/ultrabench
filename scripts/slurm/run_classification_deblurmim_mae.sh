#!/bin/bash
#SBATCH --job-name=classification-deblurmim-mae
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=3
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=3:00:00
#SBATCH --array=0-3
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
#
# Run downstream classification evaluation for DeblurMIM MAE. Each array task
# is a worker that takes task, evaluation method, and seed combos from a shared
# queue until the job is nearly out of time, then submits a successor job for
# whatever is left.
#
# Usage: sbatch scripts/slurm/run_classification_deblurmim_mae.sh
#        scripts/slurm/queue_status.sh classification_deblurmim_mae
#        scripts/slurm/queue_retry.sh classification_deblurmim_mae

set -e

EXPERIMENT="classification_deblurmim_mae"

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

declare -a COMBOS
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        for task in "${TASKS[@]}"; do
            COMBOS+=("$seed $method $task")
        done
    done
done

MODEL="deblurmim_mae"
HEAD="linear_classifier"
BATCH_SIZE=32
ACCUM_GRAD=1

# Run a single combo. $1 is the run directory reserved for this attempt, the
# remaining arguments are the fields of the combo.
ub_run_one() {
    local run_dir="$1" seed="$2" method="$3" task="$4"

    python ultrabench2/src/ultrabench/training/main_launch.py \
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

source $project/ultrabench2/scripts/slurm/run_queue.sh
ub_queue_init

source $project/ultrabench2/scripts/slurm/setup_node.sh
ub_queue_run
