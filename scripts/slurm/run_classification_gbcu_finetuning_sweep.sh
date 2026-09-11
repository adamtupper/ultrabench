#!/bin/bash
#SBATCH --job-name=classification-gbcu-finetuning-sweep
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=3
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=3:00:00
#SBATCH --array=0-3
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm_logs/%x-%A_%a.out
#
# Sweep weight decay and augmentation strategy for full fine-tuning on GBCU
# classification with UltraFedFM, to validate a lower backbone_max_lr_ratio
# (capped max LR + linear warmup + per-layer decay). Each array task is a
# worker that takes combos from a shared queue until the job is nearly out of
# time, then submits a successor job for whatever is left.
#
# Usage: sbatch scripts/slurm/run_classification_gbcu_finetuning_sweep.sh
#        scripts/slurm/queue_status.sh classification_gbcu_finetuning_sweep
#        scripts/slurm/queue_retry.sh classification_gbcu_finetuning_sweep

set -e

EXPERIMENT="classification_gbcu_finetuning_sweep"

SEED=42
METHODS=(full_finetuning)
LRS=(1e-3)
WEIGHT_DECAYS=(1e-2 5e-2 1e-1)
AUGMENTATIONS=(null general)
BACKBONE_MAX_LR_RATIO=0.01

declare -a COMBOS
for method in "${METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
        for wd in "${WEIGHT_DECAYS[@]}"; do
            for aug in "${AUGMENTATIONS[@]}"; do
                COMBOS+=("$method $lr $wd $aug")
            done
        done
    done
done

OUTPUT_DIR="$scratch/results/ultrabench/testing"
MODEL="ultrafedfm"
HEAD="linear_classifier"
TASK="gbcu_classification"
BATCH_SIZE=32
ACCUM_GRAD=1

# This sweep names its runs after the hyperparameters rather than the time they
# started, so the timestamp the queue offers is unused.
ub_run_dir() {
    local method="$2" lr="$3" wd="$4" aug="$5"

    echo "${OUTPUT_DIR}/${MODEL}/${TASK}/seed_${SEED}/method_${method}_lr_${lr}_wd_${wd}_aug_${aug}_backbonemaxlrratio_${BACKBONE_MAX_LR_RATIO}"
}

# Run a single combo. $1 is the run directory reserved for this attempt, the
# remaining arguments are the fields of the combo.
ub_run_one() {
    local run_dir="$1" method="$2" lr="$3" wd="$4" aug="$5"

    python ultrabench2/src/ultrabench2/training/main_launch.py \
        system.output_dir=${OUTPUT_DIR} \
        system.checkpoint_dir=${CHECKPOINT_DIR} \
        hydra.run.dir="${run_dir}" \
        system.save_last_checkpoint=false \
        system.data_loader_workers=${SLURM_CPUS_PER_TASK} \
        system.enable_progress_bar=false \
        model=${MODEL} \
        head=${HEAD} \
        task=${TASK} \
        mode=${method} \
        mode.optimizer.lr=${lr} \
        mode.optimizer.weight_decay=${wd} \
        data.augmentation_strategy=${aug} \
        optimization.seed=${SEED} \
        optimization.patience=null \
        optimization.batch_size=${BATCH_SIZE} \
        optimization.accum_grad=${ACCUM_GRAD} \
        optimization.backbone_max_lr_ratio=${BACKBONE_MAX_LR_RATIO}
}

source $project/ultrabench2/scripts/slurm/run_queue.sh
ub_queue_init

source $project/ultrabench2/scripts/slurm/setup_node.sh
ub_queue_run
