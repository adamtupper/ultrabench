#!/bin/bash

# Shared run queue for the Slurm launch scripts.
#
# Each array task of a run script is a worker that repeatedly claims a run
# configuration from a shared CSV ledger, executes it, and records the outcome,
# until the job is nearly out of wall clock time. Whatever is left over is
# picked up by a successor job that the last worker out submits.
#
# A run script provides:
#
#   EXPERIMENT   name of the queue, e.g. "classification_ultrafedfm"
#   COMBOS       array of whitespace-separated run configurations
#   ub_run_one   callback running one configuration; $1 is the Hydra run
#                directory, the remaining arguments are the combo fields
#   ub_run_dir   callback echoing the Hydra run directory for an attempt;
#                optional, the default below suits "<seed> <method> <task>"
#
# and then calls ub_queue_init before sourcing setup_node.sh and ub_queue_run
# after it. Splitting the two lets a job with nothing left to do exit before
# paying for the virtual environment build.
#
# Usage: source scripts/slurm/run_queue.sh

# Tunables. Override by exporting before sourcing, or via --export on sbatch.
UB_MAX_ATTEMPTS="${UB_MAX_ATTEMPTS:-3}"          # tries before a run is abandoned
UB_STALE_AFTER="${UB_STALE_AFTER:-600}"          # seconds without a heartbeat
UB_MIN_START_MARGIN="${UB_MIN_START_MARGIN:-900}" # floor on the start margin
UB_MAX_HOPS="${UB_MAX_HOPS:-20}"                 # length limit on the job chain
UB_CHAIN_HOP="${UB_CHAIN_HOP:-0}"
UB_LOCK_TIMEOUT="${UB_LOCK_TIMEOUT:-120}"
UB_HEARTBEAT_INTERVAL="${UB_HEARTBEAT_INTERVAL:-60}"
UB_LOCK_IMPL="${UB_LOCK_IMPL:-flock}"            # "flock" or "mkdir"

# Defaults shared by every run script; the sweep scripts override OUTPUT_DIR.
OUTPUT_DIR="${OUTPUT_DIR:-$scratch/results/ultrabench}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$project/models/ultrabench}"
SHOT_PCT="${SHOT_PCT:-1.0}"

UB_LEDGER_HEADER="combo_id,status,attempts,fields,job_id,array_task_id,node,run_dir,started_at,heartbeat_at,ended_at,duration_s,exit_code"

# Hydra run directory for one attempt. Mirrors the hydra.run.dir template in
# src/ultrabench/training/config/default.yaml, but with a timestamp we generate
# ourselves so the directory is known before the run starts and can be recorded
# in the ledger. Scripts whose combos have different fields override this.
#
# SHOT_PCT above has to match data.shot_pct in that config, since the run
# scripts leave it at its default rather than passing it on the command line.
if ! declare -F ub_run_dir >/dev/null; then
    ub_run_dir() {
        local ts="$1" seed="$2" method="$3" task="$4"

        echo "${OUTPUT_DIR}/runs/${MODEL}/${task}/shot_${SHOT_PCT}/seed_${seed}/${ts}"
    }
fi

# --------------------------------------------------------------------- lock --

# A fresh descriptor per acquisition, so a background heartbeat that inherits
# the environment cannot accidentally share the parent's lock.
ub_lock_acquire() {
    if [[ "$UB_LOCK_IMPL" == "mkdir" ]]; then
        local waited=0
        until mkdir "$UB_QUEUE_DIR/lock.d" 2>/dev/null; do
            sleep 1
            waited=$((waited + 1))
            if (( waited > UB_LOCK_TIMEOUT )); then
                echo "[ub_queue] breaking a lock held for ${waited}s" >&2
                rm -rf "$UB_QUEUE_DIR/lock.d"
                waited=0
            fi
        done
        echo "${SLURM_JOB_ID:-$$}" > "$UB_QUEUE_DIR/lock.d/owner"
    else
        exec {UB_LOCK_FD}>"$UB_QUEUE_DIR/lock"
        flock -w "$UB_LOCK_TIMEOUT" "$UB_LOCK_FD" || {
            echo "[ub_queue] timed out waiting for the ledger lock" >&2
            return 1
        }
    fi
}

ub_lock_release() {
    if [[ "$UB_LOCK_IMPL" == "mkdir" ]]; then
        rm -rf "$UB_QUEUE_DIR/lock.d"
    else
        flock -u "$UB_LOCK_FD" 2>/dev/null || true
        exec {UB_LOCK_FD}>&-
        UB_LOCK_FD=""
    fi
}

# ------------------------------------------------------------------- ledger --

ub_ledger_create() {
    local i combo tmp="${UB_LEDGER}.tmp.$$"

    printf '%s\n' "$UB_LEDGER_HEADER" > "$tmp"
    for i in "${!COMBOS[@]}"; do
        combo="${COMBOS[$i]}"
        if [[ "$combo" == *,* ]]; then
            echo "[ub_queue] combo '$combo' contains a comma, which the ledger cannot represent" >&2
            rm -f "$tmp"
            exit 1
        fi
        printf '%s,pending,0,%s,,,,,,,,,\n' "$i" "$combo" >> "$tmp"
    done
    mv -f "$tmp" "$UB_LEDGER"
    echo "[ub_queue] created a ledger of ${#COMBOS[@]} run(s) at $UB_LEDGER"
}

ub_outstanding() {
    awk -F, 'NR > 1 && ($2 == "pending" || $2 == "running") { n++ } END { print n + 0 }' "$UB_LEDGER"
}

# Combos marked running whose worker has stopped touching its heartbeat file.
# Liveness lives in the file's mtime rather than a ledger column so that the
# heartbeat never has to take the lock.
ub_reapable_ids() {
    local now id claimed mtime

    now=$(date +%s)
    awk -F, -v OFS=' ' 'NR > 1 && $2 == "running" { print $1, $9 + 0 }' "$UB_LEDGER" \
        | while read -r id claimed; do
        # Fall back to the claim time if the heartbeat file has not appeared
        # yet, so a fresh claim is never mistaken for a dead one.
        mtime=$(stat -c %Y "$UB_QUEUE_DIR/heartbeats/$id" 2>/dev/null || echo "$claimed")
        if (( now - mtime > UB_STALE_AFTER )); then
            printf '%s ' "$id"
        fi
    done
}

# Reap dead claims, abandon exhausted runs, and claim the first pending run.
# Echoes "<combo_id>,<attempt>,<previous run dir>,<fields>" or nothing at all.
ub_claim_next() {
    local now reap tmp claim

    now=$(date +%s)
    reap=$(ub_reapable_ids)
    tmp="${UB_LEDGER}.tmp.$$"

    ub_lock_acquire
    claim=$(awk -F, -v OFS=, \
        -v now="$now" -v maxatt="$UB_MAX_ATTEMPTS" -v reap=" $reap " \
        -v jid="${SLURM_JOB_ID:-0}" -v aid="${SLURM_ARRAY_TASK_ID:-0}" \
        -v node="${SLURMD_NODENAME:-unknown}" -v tmp="$tmp" '
        NR == 1 { print > tmp; next }
        {
            n++
            id[n] = $1; st[n] = $2; at[n] = $3; fl[n] = $4; ji[n] = $5
            ai[n] = $6; nd[n] = $7; rd[n] = $8; sa[n] = $9; hb[n] = $10
            ea[n] = $11; du[n] = $12; ec[n] = $13
        }
        END {
            for (r = 1; r <= n; r++) {
                if (st[r] == "running" && index(reap, " " id[r] " ") > 0) st[r] = "pending"
                if (st[r] == "pending" && at[r] + 0 >= maxatt) st[r] = "abandoned"
            }
            pick = 0
            for (r = 1; r <= n && !pick; r++) if (st[r] == "pending") pick = r
            if (pick) {
                st[pick] = "running"
                at[pick] = at[pick] + 1
                ji[pick] = jid; ai[pick] = aid; nd[pick] = node
                sa[pick] = now; hb[pick] = now
                ea[pick] = ""; du[pick] = ""; ec[pick] = ""
                print id[pick], at[pick], rd[pick], fl[pick]
            }
            for (r = 1; r <= n; r++)
                print id[r], st[r], at[r], fl[r], ji[r], ai[r], nd[r], rd[r], \
                      sa[r], hb[r], ea[r], du[r], ec[r] > tmp
        }
    ' "$UB_LEDGER")
    mv -f "$tmp" "$UB_LEDGER"
    # Start the heartbeat inside the critical section: a claim that is visible
    # as running but has no heartbeat yet would look reapable to another worker.
    if [[ -n "$claim" ]]; then
        : > "$UB_QUEUE_DIR/heartbeats/${claim%%,*}"
    fi
    ub_lock_release

    printf '%s' "$claim"
}

# Reserve a run directory for an attempt and record it against the combo.
#
# hydra.run.dir has no mode component, so two combos that differ only by
# evaluation method land in the same directory whenever they start in the same
# second - which is exactly what happens when several workers claim their first
# run at once. Sharing a directory would make the restart in ub_queue_run wipe a
# sibling's finished output, so nudge the timestamp forward until it is unique.
ub_reserve_run_dir() {
    local cid="$1" epoch="$2"
    shift 2
    local ts dir taken tmp="${UB_LEDGER}.tmp.$$"

    ub_lock_acquire
    local previous=""
    while :; do
        ts=$(date -d "@$epoch" +%Y-%m-%d_%H-%M-%S)
        dir=$(ub_run_dir "$ts" "$@")
        # A run_dir scheme that ignores the timestamp is unique by construction,
        # so stop nudging once the path stops changing.
        if [[ "$dir" == "$previous" ]]; then
            break
        fi
        previous="$dir"
        taken=0
        if awk -F, -v d="$dir" -v cid="$cid" \
            'NR > 1 && $1 != cid && $8 == d { hit = 1 } END { exit !hit }' "$UB_LEDGER"; then
            taken=1
        elif [[ -d "$dir" ]]; then
            taken=1
        fi
        if (( ! taken )); then
            break
        fi
        epoch=$(( epoch + 1 ))
    done

    awk -F, -v OFS=, -v cid="$cid" -v dir="$dir" '
        NR == 1 { print; next }
        $1 == cid { $8 = dir }
        { print }
    ' "$UB_LEDGER" > "$tmp"
    mv -f "$tmp" "$UB_LEDGER"
    ub_lock_release

    printf '%s' "$dir"
}

ub_finish_combo() {
    local cid="$1" status="$2" code="$3" duration="$4" tmp="${UB_LEDGER}.tmp.$$"

    ub_lock_acquire
    awk -F, -v OFS=, -v cid="$cid" -v st="$status" -v code="$code" \
        -v dur="$duration" -v now="$(date +%s)" '
        NR == 1 { print; next }
        $1 == cid { $2 = st; $11 = now; $12 = dur; $13 = code }
        { print }
    ' "$UB_LEDGER" > "$tmp"
    mv -f "$tmp" "$UB_LEDGER"
    ub_lock_release
}

# Hand a claim back to the queue so the next job restarts it from scratch. The
# run directory is deliberately left recorded, so whoever claims it next wipes
# the partial output before starting again.
ub_release_combo() {
    local cid="$1" tmp="${UB_LEDGER}.tmp.$$"

    ub_lock_acquire
    awk -F, -v OFS=, -v cid="$cid" -v now="$(date +%s)" '
        NR == 1 { print; next }
        $1 == cid && $2 == "running" { $2 = "pending"; $11 = now }
        { print }
    ' "$UB_LEDGER" > "$tmp"
    mv -f "$tmp" "$UB_LEDGER"
    ub_lock_release
}

# ---------------------------------------------------------------- heartbeat --

ub_heartbeat_start() {
    local file="$UB_QUEUE_DIR/heartbeats/$1"

    touch "$file"
    # Detached from the job's stdout, so an orphaned sleep cannot hold it open.
    ( while :; do sleep "$UB_HEARTBEAT_INTERVAL"; touch "$file" || true; done ) \
        > /dev/null 2>&1 &
    UB_HEARTBEAT_PID=$!
}

ub_heartbeat_stop() {
    if [[ -n "${UB_HEARTBEAT_PID:-}" ]]; then
        # The sleep first: killing only the subshell would orphan it.
        pkill -P "$UB_HEARTBEAT_PID" 2>/dev/null || true
        kill "$UB_HEARTBEAT_PID" 2>/dev/null || true
        wait "$UB_HEARTBEAT_PID" 2>/dev/null || true
        UB_HEARTBEAT_PID=""
    fi
}

# ------------------------------------------------------------------ timing ---

# Seconds since the epoch at which this allocation expires. Derived rather than
# hardcoded, because --time varies between run scripts.
ub_deadline() {
    local end

    if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
        echo "$SLURM_JOB_END_TIME"
        return 0
    fi

    end=$(scontrol show job "${SLURM_JOB_ID:-0}" -o 2>/dev/null | tr ' ' '\n' \
        | sed -n 's/^EndTime=//p' | head -n 1)
    if [[ -n "$end" && "$end" != "Unknown" ]]; then
        date -d "$end" +%s
        return 0
    fi

    # Unknown: never stop voluntarily and rely on the SIGTERM trap instead.
    echo "[ub_queue] could not determine the job end time; relying on the TERM trap" >&2
    echo $(( $(date +%s) + 86400 ))
}

# Time to leave for one more run. The longest run this experiment has actually
# completed is a better estimate than any constant, so use it once we have one.
ub_start_margin() {
    local longest

    longest=$(awk -F, 'NR > 1 && $2 == "done" && $12 + 0 > m { m = $12 + 0 } END { print m + 0 }' "$UB_LEDGER")
    if (( longest > UB_MIN_START_MARGIN )); then
        echo "$longest"
    else
        echo "$UB_MIN_START_MARGIN"
    fi
}

# ------------------------------------------------------------------- chain ---

ub_submit_successor() {
    local outstanding guard next_hop

    outstanding=$(ub_outstanding)
    if (( outstanding == 0 )); then
        echo "[ub_queue] nothing outstanding; no successor needed"
        return 0
    fi

    next_hop=$(( UB_CHAIN_HOP + 1 ))
    if (( next_hop > UB_MAX_HOPS )); then
        echo "[ub_queue] chain hop limit ($UB_MAX_HOPS) reached with $outstanding run(s) outstanding" >&2
        return 0
    fi

    # One successor per array job, whichever worker gets here first.
    guard="$UB_QUEUE_DIR/successor.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}"
    ub_lock_acquire
    if [[ -e "$guard" ]]; then
        ub_lock_release
        return 0
    fi
    : > "$guard"
    ub_lock_release

    echo "[ub_queue] $outstanding run(s) outstanding; submitting hop $next_hop"
    sbatch --dependency="afterany:${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" \
        --chdir="$UB_SUBMIT_DIR" \
        --export="ALL,UB_CHAIN_HOP=$next_hop" \
        "$UB_SCRIPT" || {
        echo "[ub_queue] sbatch failed; resubmit manually with: sbatch $UB_SCRIPT" >&2
        rm -f "$guard"
    }
}

# Slurm signals the batch script shortly before the wall clock expires. Give the
# in-flight run back to the queue and get the successor queued.
ub_on_term() {
    trap '' TERM
    echo "[ub_queue] wall clock nearly reached; releasing in-flight work"

    UB_LOCK_TIMEOUT=10
    if [[ -n "${UB_CHILD_PID:-}" ]]; then
        kill -TERM "$UB_CHILD_PID" 2>/dev/null || true
    fi
    ub_heartbeat_stop
    if [[ -n "${UB_CURRENT_COMBO:-}" ]]; then
        ub_release_combo "$UB_CURRENT_COMBO" || true
    fi
    ub_submit_successor || true
    exit 0
}

# ------------------------------------------------------------ entry points ---

ub_queue_init() {
    local outstanding

    if [[ -z "${EXPERIMENT:-}" ]]; then
        echo "[ub_queue] EXPERIMENT must be set before calling ub_queue_init" >&2
        exit 1
    fi
    if ! declare -F ub_run_one >/dev/null; then
        echo "[ub_queue] ub_run_one must be defined before calling ub_queue_init" >&2
        exit 1
    fi
    if (( ${#COMBOS[@]} == 0 )); then
        echo "[ub_queue] COMBOS is empty" >&2
        exit 1
    fi

    UB_QUEUE_DIR="${UB_QUEUE_DIR:-$OUTPUT_DIR/_queue/$EXPERIMENT}"
    UB_LEDGER="$UB_QUEUE_DIR/ledger.csv"
    # Not $0: Slurm runs a spooled copy of the batch script, and resubmitting
    # that would also pin the chain to a stale snapshot of the code.
    UB_SCRIPT="${UB_SCRIPT:-$project/ultrabench/scripts/slurm/run_${EXPERIMENT}.sh}"
    UB_SUBMIT_DIR="${UB_SUBMIT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
    export UB_SCRIPT UB_SUBMIT_DIR

    mkdir -p "$UB_QUEUE_DIR/logs" "$UB_QUEUE_DIR/heartbeats"

    ub_lock_acquire
    if [[ ! -f "$UB_LEDGER" ]]; then
        ub_ledger_create
    fi
    ub_lock_release

    UB_DEADLINE=$(ub_deadline)

    outstanding=$(ub_outstanding)
    echo "[ub_queue] $EXPERIMENT: $outstanding run(s) outstanding, chain hop $UB_CHAIN_HOP"
    if (( outstanding == 0 )); then
        echo "[ub_queue] nothing to do; exiting before node setup"
        exit 0
    fi
}

ub_queue_run() {
    local remaining margin claim cid attempt prev_dir fields run_dir log rc started ended

    trap ub_on_term TERM

    while :; do
        remaining=$(( UB_DEADLINE - $(date +%s) ))
        margin=$(ub_start_margin)
        if (( remaining < margin )); then
            echo "[ub_queue] ${remaining}s left but a run may need ${margin}s; stopping"
            break
        fi

        claim=$(ub_claim_next)
        if [[ -z "$claim" ]]; then
            echo "[ub_queue] no runs left to claim"
            break
        fi
        IFS=, read -r cid attempt prev_dir fields <<< "$claim"

        # Training never resumes, so any output from an earlier attempt is dead
        # weight. Dropping it here is what makes an interrupted run restart.
        if [[ -n "$prev_dir" ]]; then
            echo "[ub_queue] run $cid: discarding output of a previous attempt at $prev_dir"
            rm -rf "$prev_dir" || true
        fi

        # Unquoted on purpose: the combo fields become separate arguments.
        run_dir=$(ub_reserve_run_dir "$cid" "$(date +%s)" $fields)

        log="$UB_QUEUE_DIR/logs/$cid/attempt${attempt}.log"
        mkdir -p "$(dirname "$log")"

        UB_CURRENT_COMBO="$cid"
        ub_heartbeat_start "$cid"

        echo "[ub_queue] run $cid attempt $attempt: $fields"
        started=$(date +%s)
        rc=0
        # Backgrounded so the TERM trap can fire while training is in progress;
        # bash defers traps until a foreground command returns.
        ub_run_one "$run_dir" $fields > "$log" 2>&1 &
        UB_CHILD_PID=$!
        wait "$UB_CHILD_PID" || rc=$?
        UB_CHILD_PID=""
        ended=$(date +%s)

        ub_heartbeat_stop
        UB_CURRENT_COMBO=""

        if (( rc == 0 )); then
            ub_finish_combo "$cid" done 0 $(( ended - started ))
            echo "[ub_queue] run $cid finished in $(( ended - started ))s"
        else
            ub_finish_combo "$cid" failed "$rc" $(( ended - started ))
            echo "[ub_queue] run $cid FAILED with exit $rc after $(( ended - started ))s, see $log" >&2
        fi
    done

    trap - TERM
    ub_submit_successor
}
