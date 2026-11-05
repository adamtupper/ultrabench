#!/bin/bash

# Report the state of a run queue.
#
# Usage: scripts/slurm/queue_status.sh <experiment>
#        scripts/slurm/queue_status.sh --check-locking
#
# Example: scripts/slurm/queue_status.sh classification_ultrafedfm

set -e

OUTPUT_DIR="${OUTPUT_DIR:-$scratch/results/ultrabench}"

ub_check_locking() {
    local dir script

    dir="$OUTPUT_DIR/_queue/_locktest"
    mkdir -p "$dir"
    rm -f "$dir/report" "$dir/lock"
    script="$dir/locktest.sh"

    cat > "$script" <<'EOF'
#!/bin/bash
#SBATCH --job-name=ub-locktest
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --array=0-1
#SBATCH --output=/dev/null

DIR="$1"
exec {FD}>"$DIR/lock"
START=$(date +%s)
flock -w 120 "$FD"
WAITED=$(( $(date +%s) - START ))
echo "task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME waited=${WAITED}s" >> "$DIR/report"
sleep 20
flock -u "$FD"
EOF
    chmod +x "$script"

    echo "Submitting a two-task array that contends for one lock on $OUTPUT_DIR ..."
    sbatch --wait "$script" "$dir" >/dev/null
    echo
    cat "$dir/report"
    echo
    echo "Expected: the two tasks report different nodes, and one waited ~20s."
    echo "If both waited ~0s while on different nodes, flock is not shared across"
    echo "nodes on this filesystem - rerun the queue with UB_LOCK_IMPL=mkdir."
    echo "If both tasks landed on the same node the test is inconclusive; rerun it"
    echo "when the cluster is busier, or add --spread-job to the array above."
}

if [[ "${1:-}" == "--check-locking" ]]; then
    ub_check_locking
    exit 0
fi

EXPERIMENT="${1:-}"
if [[ -z "$EXPERIMENT" ]]; then
    echo "usage: $0 <experiment> | --check-locking" >&2
    exit 1
fi

LEDGER="$OUTPUT_DIR/_queue/$EXPERIMENT/ledger.csv"
if [[ ! -f "$LEDGER" ]]; then
    echo "no queue at $LEDGER" >&2
    exit 1
fi

echo "$EXPERIMENT  ($LEDGER)"
echo

awk -F, '
    NR > 1 { count[$2]++; total++ }
    END {
        split("pending running done failed abandoned", order, " ")
        for (i = 1; i <= 5; i++) printf "  %-10s %4d\n", order[i], count[order[i]] + 0
        printf "  %-10s %4d\n", "total", total
    }
' "$LEDGER"

awk -F, '
    NR > 1 && $2 == "done" && $12 != "" { n++; sum += $12; if ($12 > max) max = $12 }
    END {
        if (n) printf "\n  completed runs: mean %ds, longest %ds\n", sum / n, max
    }
' "$LEDGER"

awk -F, '
    NR > 1 && ($2 == "failed" || $2 == "abandoned") {
        if (!header++) printf "\n%-6s %-10s %-4s %s\n", "id", "status", "exit", "run"
        printf "%-6s %-10s %-4s %s\n", $1, $2, $13, $4
    }
' "$LEDGER"

awk -F, -v queue="$EXPERIMENT" '
    NR > 1 && ($2 == "failed" || $2 == "abandoned") { n++ }
    END { if (n) printf "\nRetry them with: scripts/slurm/queue_retry.sh %s\n", queue }
' "$LEDGER"
