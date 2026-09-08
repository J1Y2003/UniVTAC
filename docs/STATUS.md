# Status — 2026-09-08 (cuDNN blocker resolved)

Where the work stands, what is verified, and what to do next. Update this when
the state changes; it is the handoff document between sessions and machines.

## Verified working

* **The three benchmark datasets are converted and verified** (2026-09-08).
  `insert_hole`, `insert_tube`, `pull_out_key`, variant `baseline_finetuned`,
  all 100 released episodes, config `clean`. 100 parquet each; **100 mp4 for
  `insert_hole` and `pull_out_key`, 200 for `insert_tube`** — that asymmetry
  is the per-task camera rule from the paper working (two views on
  `insert_tube` and `lift_bottle`, third-person only elsewhere), split
  100-per-camera across `observation.images.head` and `.wrist`. Raw side:
  100 HDF5 per task under `data/isaac45/<task>/hdf5` with the
  `data/<task>/clean` symlink bridge resolving.
* **Dataset conversion, at full scale.** Both variants converted from
  `isaac45/lift_bottle` (100 episodes, config `clean`): 100 parquet + 200 mp4
  each, `state` 113-D (tactile) / 17-D (baseline), `action` 8-D, **310 rows per
  episode** (311 frames − 1, confirming the `joint[:-1]`/`joint[1:]` shift).
  Datasets are tiny — **31 MB** each.
* **UniVTAC data pipeline.** `data/download.sh --task lift_bottle --version 45`,
  the `data/<task>/<config>` symlink bridge, and ACT's `process_data.sh` all run.
* **GR00T loads and trains.** The 3B model loads, the modality config registers
  under `NEW_EMBODIMENT`, the **113-D state is accepted**, dataset stats generate
  (28,932 samples, 29 shards), and training reaches the optimisation step.
  Trainable 1.62 B / 3.14 B params.
* **cuRobo** is healthy (`Plan True` on collected seeds).
* **Training runs at a usable rate: 1.89 s/it** (1 GPU, `--global-batch-size`
  64, cuDNN live), measured in smoke job 166002. That is ~90x faster than the
  170 s/it before the cuDNN fix, matching the 86x the vision-tower benchmark
  predicted. `MAX_STEPS=4000` is therefore **~2.1 h per variant**, ~4.2 h for
  both, against a 470 h projection before.
* **Checkpoints get written.** Smoke job 166002 produced `checkpoint-20` and a
  `final` symlink for *both* variants — the first checkpoints this project has
  ever written. Confirms the trainer accepts our modality config under
  `NEW_EMBODIMENT` and the **113-D state**, end to end.
* **Evaluation on a finetuned checkpoint works.** Same job:
  `eval tactile: exit 0 -- 1 scored, 0 success, SR=0.0%`. The 0% is meaningless
  at 20 training steps; what it proves is that `Gr00tPolicy` loads a *finetuned*
  checkpoint and answers `get_action` for a full episode — the last of the three
  interfaces `smoke_test.sh` was written to exercise.

## Not yet verified

* Whether 8-task scaling should be per-task or one multi-task finetune. See
  "Open decisions".
* Real-run eval throughput. The smoke test scored 1 episode; the real run wants
  50 per variant, and eval (Isaac Sim boot + 3B server load + episodes) is now
  the dominant cost rather than training. Budget it from `eval-*.log` before
  assuming the 24 h `TIMELIMIT` is comfortable.
* Checkpoint-write cost on NFS. One smoke-test step read **9.82 s/it** against a
  1.89 s/it steady state, most likely the step that wrote a checkpoint. At
  `SAVE_STEPS=1000` that happens 4x per variant, so measure it before trusting
  the schedule.

## Resolved: the 170 s/step blocker was a mismatched cuDNN

**Root cause.** The GR00T venv had cuDNN **9.13.0** on disk while
`torch==2.9.0+cu128` pins `nvidia-cudnn-cu12==9.10.2.21`. The mismatched
library reported `CUDNN_STATUS_NOT_INITIALIZED`, which was misread as "this
cluster's driver is too old", so `DISABLE_CUDNN=1` became the default — and
that is what cost 170 s/step. Reinstalling the pinned version fixed cuDNN with
no driver change and no admin involvement. Full write-up, including why every
cheap signal agreed and was wrong, in
[SETUP.md](SETUP.md#cudnn-check-the-library-not-the-metadata).

**Why disabling cuDNN was so expensive.** Qwen3VL reshapes every visual patch
into its own batch element, so the vision patch embed's
`Conv3d(3, 1024, (2,16,16), stride=(2,16,16))` runs over ~32,768 batch elements
per step (64 samples × 2 cameras × 256 patches at 256 px). Without cuDNN, ATen
walks that batch with a per-element im2col loop driven from the main Python
thread. Measured on an A100, vision-tower forward+backward at 32 images:
**46.16 s** with the shipped conv versus **0.53 s** with the algebraically
identical matmul — 86×, and 98.8% of the tower. Scaled to the real batch that
is ~185 s against a 170 s step, i.e. essentially the whole step.

**What the evidence ruled out**, so nobody re-chases it:

| Hypothesis | Verdict |
|---|---|
| random-access H.264 decode starving the GPU | **dead.** Both dataloader workers used 103 s of CPU each across 5 h — 0.55% of a core. `--dataloader-num-workers 10` would have changed nothing |
| NFS bandwidth | dead. 31 MB dataset |
| the driver being too old | dead. Same failure on 550.54.14 *and* 550.163.01, and in a fully pristine `env -i` environment |
| `LD_LIBRARY_PATH` / conda pollution picking up a stale cuDNN | dead. The loaded library was the venv's own, in both a polluted and a pristine environment |

**The signature to recognise next time.** Main Python thread pegged at 99.7% of
one core (18,541 s of user CPU in 18,589 s elapsed) with only 16 s of *system*
time; `pt_autograd_0` at 8 s; dataloader workers idle; GPU at 48% utilisation
but **121 W of 400 W** with SM clocks pinned at 1410 MHz. Kernels resident half
the time doing almost no arithmetic — that is a tiny-kernel launch flood driven
from Python, not a data-starvation problem. Read power draw, not utilisation.

**The slow run was optimising correctly.** wandb's training stream for job
164704 (`lift_bottle-tactile`, 140 steps at 170 s/step): loss 1.20 -> 0.55,
grad norm O(1) throughout, LR schedule advancing normally. So the pathology was
throughput alone -- the same arithmetic, executed slowly -- not a corrupted
graph producing garbage, which had not previously been ruled out explicitly.
Two details worth keeping: the run reached only **~26% of peak LR** (2.6e-5 of
1e-4, consistent with `warmup_ratio` 0.05 over 10,000 steps), so nothing about
the full trajectory can be read from it; and loss was **flat near 1.02 from
step 40 to 110, then fell 45% in 30 steps** alongside a 5x grad-norm spike --
the action head leaving its initial constant-output plateau, which under
`NEW_EMBODIMENT` comes after the randomly-initialised state/action projectors
have been learned. Expect a long flat opening on the real runs and do not read
it as a failure.

**Still unverified:** the 121 W / 48%-utilisation signature is corroborated
only by `nvidia-smi` snapshots taken during the run. wandb logged the same
counters continuously in job 164704's *system* stream, which nobody has read
back yet -- `python scripts/wandb_report.py 164704`, or the System tab on the
run page. If it disagrees, the mechanism described above needs correcting (the
fix does not: 1.89 s/it is measured).

**Guard.** `scripts/preflight.py --deep` now asks the loaded cuDNN its own
version via `cudnnGetVersion()` and fails on a mismatch, because pip metadata
reported the pinned version while the files on disk were a different release.

**Defaults changed as a result:** `DISABLE_CUDNN` now defaults to `0` in
`benchmark_task.sbatch` and `smoke_test.sh`, and `GPUS` defaults to `1` in
`submit_benchmark.sh` (it was 2, which triggers the `nn.DataParallel` failure
already in the table below).

## Failures already diagnosed (do not re-litigate)

| Symptom | Cause |
|---|---|
| `CUDNN_STATUS_NOT_INITIALIZED` | **cuDNN on disk did not match torch's pin** (9.13.0 present, 9.10.2.21 required). Not the driver. `uv cache clean nvidia-cudnn-cu12` then reinstall the pin; `preflight.py --deep` now catches it. Never "fix" it with `DISABLE_CUDNN=1` — that costs 86× |
| 170 s/step, GPU at 48% but only 121 W | the above, via the patch-embed `Conv3d` falling off the cuDNN path |
| `module must have its parameters ... on device: cpu` | `--num-gpus 2` wraps the model in `nn.DataParallel`, which needs everything on `cuda:0`. **Use 1 GPU.** |
| `401` on `nvidia/Cosmos-Reason2-2B` | no `HF_TOKEN` in the job. Preflight now probes read access |
| `hf_transfer` `ValueError` | the flag is a hard error without the package. Now probed before being set |
| `results=/var/spool/slurm/d/...` | `sbatch` copies the script to the spool; `REPO_ROOT` now prefers `SLURM_SUBMIT_DIR` |
| job pending 8 h | ordinary queue wait behind the senior's jobs on the shared account — not partition tier |
| `Batch job submission failed: Unspecified error` | a submit-filter rule; see CLAUDE.md |
| a per-task convert loop converts only ONE task | all jobs share one `.venv-convert`, and `python -m venv` writes `bin/python` before pip installs anything, so the losers skipped the build and ran with no `pyarrow`. `convert.sbatch` now serialises on a flock and gates on the imports, not on a file existing |

## Cluster facts learned while debugging

* **Driver versions differ by node.** `worker-node3` (A100 80GB PCIe) runs
  550.54.14; `worker-node109` (A100-SXM4-80GB) runs 550.163.01. Do not write
  "the cluster's driver".
* **`srun --container` / `--container-id` exist**, there is no `module` command,
  and no `enroot`/`apptainer`/`singularity` binary. If the pinned cuDNN ever
  stops working, the containerised path is the sanctioned route — worth asking
  the other GR00T users on this cluster which image they use.
* **No CPU cgroup isolation.** Inside a job, `cpuset.cpus.effective` is
  `0-127` with no `cpu.max`, so `--cpus-per-task` is advisory and neighbours'
  load is not fenced off. Node loadavg was 94 while two other users' jobs used
  ~47 cores. Aggravates anything single-threaded; not a cause on its own.
* **`ptrace_scope=1`.** `py-spy` cannot attach to a sibling process, so
  profiling has to launch the target as its own child — `srun --overlap` into a
  running job will not work for this.
* **The progress bar is buffered.** `tail -f` the log directly; piping it
  through anything else shows nothing.

## Open decisions

## Decided (2026-09-08)

**Per-task, not multi-task.** One GR00T N1.7 finetune per task, matching
UniVTAC's ACT, which trains one policy per task. Multi-dataset training would
be cheaper and arguably a better test of a generalist VLA, but it has GR00T
solving a strictly harder problem than the baseline it is being compared to,
and that asymmetry is not worth the ambiguity. Revisit only after the per-task
numbers exist.

**Vision only, for now.** The immediate question is how GR00T N1.7 performs on
UniVTAC *without* tactile input, so only `baseline_finetuned` (17-D state) gets
trained. The tactile pipeline stays in the repo and stays working — the
converter, the 113-D modality config and the `tactile` variant are all
exercised and verified — it just is not being trained. `VARIANTS` now defaults
to `baseline_finetuned` in `submit_benchmark.sh`; pass
`VARIANTS="tactile baseline_finetuned"` to run the ablation again.

**`lift_bottle` trains fourth, gated** (2026-09-08). It is submitted by
`submit_benchmark.sh` as `EXTRA_TASKS`, with `--dependency=afterany` on all
three priority jobs, so it queues behind them and can never take a GPU a
reported task is waiting for. `afterany` rather than `afterok` on purpose: it
is an independent finetune on its own dataset, so a priority task crashing is
no reason to abandon it, and `afterok` would strand it in
`DependencyNeverSatisfied` needing a manual `scancel`. Having the model is
useful — it is a fourth data point and the substrate for any sweep — it just
is not a reported task.

**Renamed, 2026-09-08:** `overnight_ablation.sbatch` →
`benchmark_task.sbatch`, `submit_overnight.sh` → `submit_benchmark.sh`, and the
per-run stage log directory `logs/overnight-<jobid>` →
`logs/benchmark-<task>-<jobid>`. The old names described *when* the job was
meant to run and *two variants* that are no longer both trained; the new ones
describe what it does. The stage directory under `CKPT_ROOT/.stages` is
unchanged, so in-flight resumability and the recipe pins survive the rename.

**Three tasks first:** `insert_hole`, `insert_tube`, `pull_out_key`. These are
the contact-rich insertion/extraction tasks where the benchmark is most
interesting, and three per-task finetunes at ~2.1 h each is a day's work rather
than a week's. `TASKS` in `submit_benchmark.sh` defaults to exactly these.

**Settled: benchmark, not controlled ablation.** Hold the observation space
(per-task cameras) and the evaluation protocol (100 rollouts, their seeds)
fixed; run GR00T N1.7 on **its own defaults** for everything else —
`MAX_STEPS=10000`, batch 64, `ACTION_HORIZON=40`, `EXECUTION_HORIZON=16`,
GR00T's learning rate and weight decay. Matching ACT's optimizer would answer
the wrong question.

**Training data: all 100 released episodes per task** (operator's decision,
2026-09-08), against the paper's 50. So GR00T sees twice the demonstrations ACT
did — a real advantage that has to be stated next to the number. It does not
affect the internal tactile ablation, only the external ACT comparison.

The one thing to protect against is tuning against the 100 evaluation rollouts;
if a sweep is wanted, run it on `lift_bottle`, which is not a reported task.
See [ABLATION.md](ABLATION.md#comparability-with-univtacs-act).

**Comparison against UniVTAC's own models.** Their configs line up with ours:
`train_config_vision.yml` ↔ `baseline_finetuned`, `train_config.yml` ↔
`tactile`. Their released checkpoints (`data/download.sh --checkpoint`) fill the
ACT row without retraining. Before claiming a head-to-head, confirm how many
evaluation episodes and which seeds their published numbers use — ours follow
their `1_000_000 * (1 + seed)` convention, but 50 vs 100 seeds are not
comparable intervals.

**The senior is working on the same benchmark** (`iclr2027_eval_univtac_*`,
apparently a Cosmos-3 tactile model, ICLR 2027). Worth coordinating rather than
duplicating infrastructure or competing for the account's GPU quota.

## Scaling cost, once one task works

~24 GB raw per task (~190 GB for 8), converted datasets are negligible (31 MB),
~40 GB per retained checkpoint. Validate `lift_bottle` end to end before
committing to the rest.
