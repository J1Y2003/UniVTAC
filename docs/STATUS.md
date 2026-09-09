# Status

Current state, decisions in force, and the reference tables worth checking
before re-diagnosing something.

Last updated 2026-09-09.

## In flight

**Everything must be resubmitted through `bundle-sbatch`.** Plain `sbatch` is
no longer the supported path (server policy, 2026-09-09), and the whole
`slurm/` tree was converted for it. Jobs 166346-166349 predate both this and
the partition split, so they are wrong twice over: cancel them.

The shape is now **finetune first, evaluate later**, not two jobs chained:

```bash
bash slurm/submit_benchmark.sh          # one finetune per task, sjw_alinlab

# then, per checkpoint, on background:
bash slurm/eval_checkpoint.sh --task insert_hole --seed-offset 1     --checkpoint <output_dir>/checkpoint-10000
```

Evaluation cannot be queued in advance -- an eval job has to declare an existing
physical checkpoint directory, and `--parsable` belongs to the launcher so there
is no job id for a `--dependency`. Each eval writes one self-describing JSON to
`~/jaewon/workspace/eval_results/<task>-<variant>/`, building a library indexed
by task, checkpoint and seed block.

`background` preempts with `PreemptMode=REQUEUE`, so an eval job re-runs its
script from the top. It resumes from the JSONL rather than appending a second
pass from the first seed; results are deduplicated by seed either way.

**Not yet verified against the real launcher** (this conversion was written and
tested against a stub, since Claude does not touch the cluster):

* that the submitting shell's environment reaches the job. Every job script now
  refuses to run without `UNIVTAC_JOB_CONFIG=1` rather than silently taking
  default values, so the failure mode is loud.
* that `readlink -f` on a `final` symlink satisfies `--checkpoint`. The
  symlink-rejection rule is documented; resolving it could not be exercised on
  a Windows checkout, which cannot create symlinks.
* whether `bundle-sbatch` prints a job id this repo can parse. If it does not,
  submissions still work -- the id is only used for display.
* the resume-after-requeue arithmetic against a real preemption. It is tested
  against synthesised partial, duplicated and truncated result files, but not
  yet against Slurm actually requeueing a job.

## Verified

* **Datasets, all three reported tasks.** `insert_hole`, `insert_tube`,
  `pull_out_key`, variant `baseline_finetuned`, all 100 released episodes,
  config `clean`. 100 parquet each; **100 mp4 for `insert_hole` and
  `pull_out_key`, 200 for `insert_tube`** -- the per-task camera rule (two
  views on `insert_tube` and `lift_bottle`, third-person only elsewhere), split
  100-per-camera across `observation.images.head` and `.wrist`. Raw side: 100
  HDF5 per task under `data/isaac45/<task>/hdf5`, with the `data/<task>/clean`
  symlink bridge that `convert.sbatch` reads.
* **`lift_bottle` converts.** `state` 17-D, `action` 8-D, **310 rows per
  episode** (311 frames minus 1, from the `joint[:-1]`/`joint[1:]` shift).
  Converted datasets are ~31 MB.
* **GR00T trains.** The 3B model loads, the modality config registers under
  `NEW_EMBODIMENT`, the 17-D state is accepted, dataset stats generate,
  checkpoints are written and a `final` symlink created. Trainable 1.62 B of
  3.14 B params.
* **Throughput: 0.70 s/it** -- 1 GPU, `--global-batch-size 64`, cuDNN live.
  Measured on a real finetune: `insert_hole` did **10,000 steps in 117
  minutes**. At `MAX_STEPS=30000` that is **~5.9 h** of stepping per task,
  ~6.2 h with model load, dataset statistics and three checkpoint writes.
  **Size walltimes from 0.70.** The 1.89 s/it that appears in the cuDNN
  comments is a 20-step smoke-test average, two of whose steps wrote a
  checkpoint; it is the correct baseline for the cuDNN penalty ratio and the
  wrong one for a walltime.
* **Evaluation loads a finetuned checkpoint** and answers `get_action` for a
  full episode: `Gr00tPolicy`, the ZeroMQ client and the receding-horizon loop
  work end to end.
* **cuRobo** is healthy (`Plan True` on collected seeds).

## Open

* **Eval throughput.** Isaac Sim boot + 3B server load + 100 rollouts is now
  the dominant cost, not training. Budget it from a real
  `logs/benchmark-<task>-<jobid>/eval-*.log` before assuming the partition's
  2-day maximum is comfortable.
* **Checkpoint-write cost on NFS.** One step read 9.82 s against a
  sub-second steady state, most likely the step that wrote a checkpoint. Now
  largely moot: `SAVE_STEPS=10000` writes **three** checkpoints per task rather
  than ten, so even at 10 s each the total is ~30 s of a ~6 h run.
* **Per-task camera sets.** The paper and `policy/task_settings.json`
  disagree about which tasks are multi-view; we follow the paper. Confirming it
  against UniVTAC's released configs would settle it -- see
  [BENCHMARK.md](BENCHMARK.md#cameras-are-per-task).

## Decisions in force

**Benchmark, not controlled ablation.** Hold the observation space (per-task
cameras) and the evaluation protocol (100 rollouts, their
`1_000_000 * (1 + seed)` convention) fixed; run GR00T N1.7 on **its own
defaults** for everything else -- batch 64, `ACTION_HORIZON=40`,
`EXECUTION_HORIZON=16`, GR00T's learning rate and weight decay. Matching ACT's
optimiser would answer a different question.

**Step count: 30,000, and it is the one recipe number we chose.**
`MAX_STEPS=30000` is 3x GR00T's own 10000 default, so unlike everything above
it is **not** a GR00T default and has to be disclosed with the numbers. The
reason it had to be chosen at all: 10,000 steps over the full 100 released
episodes is only ~22 epochs, and with no validation split and no eval metric in
`launch_finetune.py` there is nothing that could tell us whether that is
enough. `SAVE_STEPS=10000` retains exactly `checkpoint-10000`,
`checkpoint-20000` and `checkpoint-30000` -- three points of success rate
against step count, which is the only instrument available. Which of the three
becomes the reported model is decided on a dev seed block; see
docs/BENCHMARK.md, "Choosing the step count".

**Per-task, not multi-task.** One finetune per task, matching UniVTAC's ACT,
which trains one policy per task. Multi-dataset training would be cheaper and
arguably a better test of a generalist VLA, but it has GR00T solving a strictly
harder problem than its baseline. Revisit after the per-task numbers exist.

**Evaluation on `background`, training on `sjw_alinlab`.** Lab policy: no
evaluation on the lab partition. A job holds one allocation on one partition,
so each task is two jobs. Unverified and worth checking before a long eval
lands there: whether `background` preempts, since `scripts/run_eval.py` has no
resume and would restart from the first seed.

**Observation: images, 17-D state, language instruction.** Only
`baseline_finetuned` is trained. The `tactile` variant and its 113-D state path
in `obs_adapter.py`, `spec.py`, `variants.py` and the converter are unused and
pending removal.

**Three reported tasks:** `insert_hole`, `insert_tube`, `pull_out_key`, the
contact-rich insertion and extraction tasks. `lift_bottle` trains as a gated
fourth: it is the only task a hyperparameter sweep may touch without fitting
the reported numbers.

**Training data: all 100 released episodes per task** (`0.hdf5`..`99.hdf5`),
which is ~28,932 samples and ~66 epochs at 30,000 steps. State the episode
count alongside any number, since published results on these tasks used 50.

**Do not tune against the reported evaluation rollouts.** Selecting a
checkpoint or a hyperparameter on seed offset 0 fits the test set. Use a
disjoint seed block, or `lift_bottle`. See
[BENCHMARK.md](BENCHMARK.md#choosing-the-step-count).

**Published numbers for other methods** live in
[BENCHMARK.md](BENCHMARK.md#published-numbers-on-these-tasks), for judging
whether a result is plausible.

## Failure -> cause

| Symptom | Cause |
|---|---|
| `CUDNN_STATUS_NOT_INITIALIZED` | cuDNN on disk does not match torch's pin (9.13.0 present, 9.10.2.21 required). Not the driver. `uv cache clean nvidia-cudnn-cu12`, reinstall the pin. `scripts/check_cudnn.py` runs in every GPU job and aborts on a mismatch; `preflight.py --deep` checks it from a login node. There is deliberately no way to run without cuDNN -- it costs ~86x, silently |
| ~170 s/step; GPU at 48% but ~110 W of a 400 W limit and ~0% memory-access time | the above, via Qwen3-VL's patch-embed `Conv3d` falling off the cuDNN path. See [SETUP.md](SETUP.md#3-cudnn-must-match-torchs-pin) |
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
(~31 MB); **~26 GB per retained checkpoint** (measured: 25,547,974,206 bytes on
the `insert_hole` checkpoints). Three retained per task is ~77 GB.

Nothing is copied out of a bundle. A checkpoint that ages out of managed storage
is retrained: at ~6.2 h that is cheaper than keeping a private duplicate of
every checkpoint on NFS.

Training: ~6.2 h per task at 30,000 steps, so ~25 GPU-hours for four tasks.
Evaluation is still unmeasured and is the open budget question.
