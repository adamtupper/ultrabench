# Running Experiments on DRAC Clusters

USAugment isn't currently available on the DRAC clusters, so should be downloaded on a login node before launching any jobs:

```shell
pip download --no-deps -d $project/wheels usaugment
```

## Launching a sweep

```shell
sbatch scripts/slurm/run_classification_ultrafedfm.sh
```

Each `run_*.sh` script enumerates its task, evaluation method, and seed combos into a shared
queue, then submits a small array of workers. A worker claims one run at a time and keeps going
until the job is nearly out of wall clock time, so a single allocation gets through as many runs
as it can fit instead of one. Whatever is left when the clock runs down is picked up by a
successor job that the last worker out submits with `--dependency=afterany`, and the chain
continues until the queue is empty. The final link finds nothing to do and exits in seconds.

Submit from the repository root: the `--output` path is relative to the submission directory, and
the successor jobs inherit it via `--chdir`.

Resubmitting a script that is already part-way through is safe. The queue is the only state, so
runs already recorded as finished are skipped.

## Checking progress

```shell
scripts/slurm/queue_status.sh classification_ultrafedfm
```

Reports how many runs are pending, running, finished, failed or abandoned, the mean and longest
completed run time, and a line per failure.

## Retrying failures

```shell
scripts/slurm/queue_retry.sh classification_ultrafedfm        # every failed and abandoned run
scripts/slurm/queue_retry.sh classification_ultrafedfm 7 19   # specific runs
scripts/slurm/queue_retry.sh classification_ultrafedfm --no-submit
```

Failed runs are left alone by the chain, so a configuration that fails immediately cannot burn
through the allocation. Retrying resets the attempt counter and resubmits the script.

## Layout

Each script has its own queue under the results directory, named after the script:

```
$scratch/results/ultrabench/_queue/classification_ultrafedfm/
    ledger.csv                  one row per run, with status, timings and run directory
    lock                        guards updates to the ledger
    heartbeats/<id>             touched while a run is in progress
    logs/<id>/attempt<n>.log    stdout and stderr of one attempt
    successor.<array_job_id>    marks that a job has already queued its successor
```

Job level output goes to `slurm_logs/<job-name>-<job-id>_<task-id>.out` in the submission
directory and holds one line per run. The training output itself is in the per-attempt logs
above, which is where to look when a run fails.

Training results are unaffected and still land in
`$scratch/results/ultrabench/runs/<model>/<task>/shot_<pct>/seed_<seed>/<timestamp>/<head>_<mode>/`.

## Interrupted runs

Training has no resume path, so a run still in progress when a job ends is not continued. The
next job in the chain deletes its partial output and starts it again from scratch. A run that
cannot finish inside one allocation would otherwise restart forever, so after `UB_MAX_ATTEMPTS`
tries it is marked `abandoned` and reported by `queue_status.sh`.

## Tuning

Workers per job is the `--array` line in each script; four is the default. Do not add a `%`
throttle, which would serialise the workers.

The queue reads these environment variables, all of which have sensible defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `UB_MAX_ATTEMPTS` | 3 | Tries before a run is abandoned |
| `UB_MIN_START_MARGIN` | 900 | Seconds to leave free before starting another run, until the queue has timings of its own |
| `UB_STALE_AFTER` | 600 | Seconds without a heartbeat before a claim is treated as dead |
| `UB_MAX_HOPS` | 20 | Length limit on the job chain |
| `UB_LOCK_IMPL` | `flock` | Set to `mkdir` if `flock` is not shared across nodes |

`flock` only coordinates workers on different nodes if the filesystem is mounted for it. Check
once per cluster with:

```shell
scripts/slurm/queue_status.sh --check-locking
```

If the two test tasks land on different nodes and neither waits for the other, rerun the sweeps
with `UB_LOCK_IMPL=mkdir`, which does not depend on the mount options.
