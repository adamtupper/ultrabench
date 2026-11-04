#!/bin/bash

# Return failed runs to a queue and resubmit it.
#
# With no run ids, every failed and abandoned run is retried. Attempt counters
# are reset, so a retried run gets a full allowance again. The recorded run
# directory is left in place: whichever worker picks the run up next discards
# that output before starting, exactly as it does after an interruption.
#
# Usage: scripts/slurm/queue_retry.sh <experiment> [combo_id ...]
#        scripts/slurm/queue_retry.sh <experiment> --no-submit
#
# Example: scripts/slurm/queue_retry.sh classification_ultrafedfm 7 19

set -e

OUTPUT_DIR="${OUTPUT_DIR:-$scratch/results/ultrabench}"

EXPERIMENT="${1:-}"
if [[ -z "$EXPERIMENT" ]]; then
    echo "usage: $0 <experiment> [combo_id ...] [--no-submit]" >&2
    exit 1
fi
shift

SUBMIT=1
IDS=""
for arg in "$@"; do
    if [[ "$arg" == "--no-submit" ]]; then
        SUBMIT=0
    else
        IDS+=" $arg"
    fi
done

QUEUE_DIR="$OUTPUT_DIR/_queue/$EXPERIMENT"
LEDGER="$QUEUE_DIR/ledger.csv"
SCRIPT="${UB_SCRIPT:-$project/ultrabench/scripts/slurm/run_${EXPERIMENT}.sh}"

if [[ ! -f "$LEDGER" ]]; then
    echo "no queue at $LEDGER" >&2
    exit 1
fi

exec {FD}>"$QUEUE_DIR/lock"
flock -w 120 "$FD"

TMP="${LEDGER}.tmp.$$"
RESET=$(awk -F, -v OFS=, -v ids=" $IDS " -v tmp="$TMP" '
    NR == 1 { print > tmp; next }
    {
        want = (ids == "  ") ? ($2 == "failed" || $2 == "abandoned") \
                             : (index(ids, " " $1 " ") > 0)
        if (want && $2 != "running") {
            $2 = "pending"; $3 = 0; $11 = ""; $12 = ""; $13 = ""
            n++
        }
        print > tmp
    }
    END { print n + 0 }
' "$LEDGER")
mv -f "$TMP" "$LEDGER"

flock -u "$FD"
exec {FD}>&-

echo "reset $RESET run(s) in $EXPERIMENT"

if (( RESET == 0 )); then
    echo "nothing to resubmit"
    exit 0
fi

# A fresh chain: drop the guards so the new job can submit its own successor.
rm -f "$QUEUE_DIR"/successor.*

if (( SUBMIT )); then
    sbatch "$SCRIPT"
else
    echo "submit with: sbatch $SCRIPT"
fi
