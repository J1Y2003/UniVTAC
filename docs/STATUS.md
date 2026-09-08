# Status

Current state, decisions in force, and the reference tables worth checking
before re-diagnosing something.

Last updated 2026-09-08.

## In flight

Four jobs, one per task, submitted by `slurm/submit_benchmark.sh`:
`insert_hole`, `insert_tube`, `pull_out_key` unconstrained, then `lift_bottle`
gated behind all three with `--dependency=afterany`.

## Verified

* **Datasets, all three reported tasks.** `insert_hole`, `insert_tube`,
  `pull_out_key`, variant `baseline_finetuned`, all 100 released episodes,
  config `clean`. 100 parquet each; **100 mp4 for `insert_hole` and
  `pull_out_key`, 200 for `insert_tube`** -- the per-task camera rule (two
  views on `insert_tube` and `lift_bottle`, third-person only elsewhere), split
  100-per-camera across `observation.images.head` and `.wrist`. Raw side: 100
  HDF5 per task under `data/isaac45/<task>/hdf5`, with the `data/<task>/clean`
  symlink bridge that `convert.sbatch` reads.
* **`lift_bottle`, both variants.** `state` 113-D (tactile) / 17-D (baseline),
  `action` 8-D, **310 rows per episode** (311 frames minus 1, from the
  `joint[:-1]`/`joint[1:]` shift). Converted datasets are ~31 MB.
* **GR00T trains.** The 3B model loads, the modality config registers under
  `NEW_EMBODIMENT`, both the 17-D and 113-D states are accepted, dataset stats
  generate, checkpoints are written and a `final` symlink created. Trainable
  1.62 B of 3.14 B params.
* **Throughput: 1.89 s/it** -- 1 GPU, `--global-batch-size 64`, cuDNN live. At
  `MAX_STEPS=10000` that is ~5.3 h of training per task.
* **Evaluation loads a finetuned checkpoint** and answers `get_action` for a
  full episode: `Gr00tPolicy`, the ZeroMQ client and the receding-horizon loop
  work end to end.
* **cuRobo** is healthy (`Plan True` on collected seeds).

## Open

* **Eval throughput.** Isaac Sim boot + 3B server load + 100 rollouts is now
  the dominant cost, not training. Budget it from a real
  `logs/benchmark-<task>-<jobid>/eval-*.log` before assuming the partition's
  2-day maximum is comfortable.
* **Checkpoint-write cost on NFS.** One step read 9.82 s against a 1.89 s
  steady state, most likely the step that wrote a checkpoint. At
  `SAVE_STEPS=1000` that happens 10 times per task.
* **Which camera set UniVTAC's published ACT numbers used.** Their paper
  specifies per-task cameras; confirming it against their released configs
  closes the last gap in the comparison. Ask the senior.

## Decisions in force

**Benchmark, not controlled ablation.** Hold the observation space (per-task
cameras) and the evaluation protocol (100 rollouts, their
`1_000_000 * (1 + seed)` convention) fixed; run GR00T N1.7 on **its own
defaults** for everything else -- `MAX_STEPS=10000`, batch 64,
`ACTION_HORIZON=40`, `EXECUTION_HORIZON=16`, GR00T's learning rate and weight
decay. Matching ACT's optimiser would answer a different question.

**Per-task, not multi-task.** One finetune per task, matching UniVTAC's ACT,
which trains one policy per task. Multi-dataset training would be cheaper and
arguably a better test of a generalist VLA, but it has GR00T solving a strictly
harder problem than its baseline. Revisit after the per-task numbers exist.

**Vision only.** Only `baseline_finetuned` (17-D state) is trained. The tactile
pipeline stays in the repo and stays working -- converter, 113-D modality
config and `tactile` variant are all exercised -- it is simply not being
trained. `VARIANTS="tactile baseline_finetuned"` runs the ablation again.

**Three reported tasks:** `insert_hole`, `insert_tube`, `pull_out_key`, the
contact-rich insertion and extraction tasks. `lift_bottle` trains as a gated
fourth: it is the only task a hyperparameter sweep may touch without fitting
the reported numbers.

**Training data: all 100 released episodes per task**, against the paper's 50.
GR00T therefore sees twice the demonstrations ACT did -- **a real advantage
that must be stated next to any number reported**. It does not affect the
internal tactile ablation, only the external ACT comparison.

**Do not tune against the 100 evaluation rollouts.** Selecting a checkpoint or
a hyperparameter on them fits the test set. Sweep on `lift_bottle`. See
[ABLATION.md](ABLATION.md#comparability-with-univtacs-act).

**Comparison against UniVTAC's own models.** Their configs line up with ours:
`train_config_vision.yml` maps to `baseline_finetuned`, `train_config.yml` to
`tactile`. Their released checkpoints (`data/download.sh --checkpoint`) fill
the ACT row without retraining. Confirm their episode count and seeds first --
50 and 100 rollouts are not comparable intervals.

## Failure -> cause

| Symptom | Cause |
|---|---|
| `CUDNN_STATUS_NOT_INITIALIZED` | cuDNN on disk does not match torch's pin (9.13.0 present, 9.10.2.21 required). Not the driver. `uv cache clean nvidia-cudnn-cu12`, reinstall the pin. `scripts/check_cudnn.py` runs in every GPU job and aborts on a mismatch; `preflight.py --deep` checks it from a login node. There is deliberately no way to run without cuDNN -- it costs ~86x, silently |
| ~170 s/step; GPU at 48% but ~110 W of a 400 W limit and ~0% memory-access time | the above, via Qwen3-VL's patch-embed `Conv3d` falling off the cuDNN path. See [SETUP.md](SETUP.md#slow-training-steps) |
| `module must have its parameters ... on device: cpu` | `--num-gpus 2` wraps the model in `nn.DataParallel`, which needs everything on `cuda:0`. **Use 1 GPU** |
| `401` on `nvidia/Cosmos-Reason2-2B` | no `HF_TOKEN` in the job. Preflight probes read access |
| `hf_transfer` `ValueError` | the flag is a hard error without the package. Probed before being set |
| `results=/var/spool/slurm/d/...` | `sbatch` copies the script to the spool; `REPO_ROOT` prefers `SLURM_SUBMIT_DIR` |
| a per-task convert loop converts only one task | all jobs share one `.venv-convert`, and `python -m venv` writes `bin/python` before pip installs anything, so the losers skip the build and run with no `pyarrow`. `convert.sbatch` serialises on a flock and gates on the imports |
| job pending for hours | ordinary queue wait on a shared account, not partition tier |
| `Batch job submission failed: Unspecified error` | a submit-filter rule; see CLAUDE.md, "Cluster rules" |

## Cluster facts

* **Driver versions differ by node.** `worker-node3` (A100 80GB PCIe) runs
  550.54.14; `worker-node109` (A100-SXM4-80GB) runs 550.163.01. Do not write
  "the cluster's driver".
* **A100 80GB, 400 W enforced power limit**, SM clock 1410 MHz, ~1.5 TB system
  RAM, 128 cores per node.
* **`srun --container` / `--container-id` exist**; there is no `module`
  command, and no `enroot`/`apptainer`/`singularity` binary. If the pinned
  cuDNN ever stops working, the containerised path is the sanctioned route.
* **No CPU cgroup isolation.** Inside a job, `cpuset.cpus.effective` is `0-127`
  with no `cpu.max`, so `--cpus-per-task` is advisory and neighbours' load is
  not fenced off. Size thread pools off `SLURM_CPUS_ON_NODE`, not `nproc`.
* **`ptrace_scope=1`.** `py-spy` cannot attach to a sibling process; profiling
  must launch the target as its own child.
* **The progress bar is buffered.** `tail -f` the log directly; piping it
  through anything else shows nothing.
* **The senior shares this account** and works on the same benchmark
  (`iclr2027_eval_univtac_*`). Coordinate rather than duplicating
  infrastructure or competing for GPU quota.

## Cost

~24 GB raw per task (~190 GB for all eight); converted datasets are negligible
(~31 MB); ~40 GB per retained checkpoint at `SAVE_TOTAL_LIMIT=3`.
