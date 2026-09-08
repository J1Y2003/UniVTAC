# Status — 2026-09-08

Where the work stands, what is verified, and what to do next. Update this when
the state changes; it is the handoff document between sessions and machines.

## Verified working

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

## Not yet verified

* **No training step has completed at a usable rate** (see blocker below).
* **No checkpoint has ever been written.**
* **Evaluation on a finetuned checkpoint has never run.** The eval path was
  exercised zero-shot only, and even that never completed an episode end to end.
* Whether 8-task scaling should be per-task or one multi-task finetune. See
  "Open decisions".

## Current blocker: 167 s/step

One GPU, steady state at step 3+ — not warmup. That is 465 h for the default
10,000 steps, and with `SAVE_STEPS=2500` the first checkpoint would land at
~116 h, so the job cannot produce anything within any walltime.

Evidence gathered so far:

| Fact | Value |
|---|---|
| GPU utilisation | ~50% |
| CPUs allocated | 12 (`AllocTRES=cpu=12`) |
| `--global-batch-size` | 64 (default), `--gradient-accumulation-steps` 1 |
| dataloader workers | **2** (log shows only `Worker 0`/`Worker 1`) |
| dataset size | 31 MB — so **not** an NFS bandwidth problem |
| `DISABLE_CUDNN` | 1 (driver 550.54.14 is older than torch's cu128 build) |

Leading hypothesis: **random-access H.264 decode starving the GPU.** Each step
needs ~128 frame seeks (64 samples × 2 cameras), and seeking into H.264 decodes
forward from the previous keyframe. Two workers serialising that matches 50%
utilisation. Second hypothesis: the missing cuDNN conv path.

**Next action** — the measurement that decides it, ~10 min on a debug node:
re-run `launch_finetune.py` unchanged except `--dataloader-num-workers 10`, and
read `s/it` at step 3–5. Do not pipe it through `tail`; the progress bar is
buffered and you will see nothing.

* drops to ~20–35 s/it → decode-bound. Then `MAX_STEPS=2000` (4.4 epochs on
  28,932 samples; 10,000 was 22 epochs) gives roughly 15 h in one job.
* stays near 167 s/it → compute-bound, i.e. cuDNN. No code fix exists; needs
  driver r570+ or `cuda-compat-12-8` from an admin. **167 s/step is the number
  to bring them.**

If decode-bound and 10 workers is not enough, the fix is ours: the converter
writes H.264 with a default keyframe interval, so re-encoding with `-g 1`
(all-intra) makes seeks nearly free. At 31 MB the size cost is irrelevant.

## Failures already diagnosed (do not re-litigate)

| Symptom | Cause |
|---|---|
| `CUDNN_STATUS_NOT_INITIALIZED` | driver older than torch's CUDA build; cuBLAS survives via minor-version compat, cuDNN does not. Workaround `DISABLE_CUDNN=1`; real fix needs an admin |
| `module must have its parameters ... on device: cpu` | `--num-gpus 2` wraps the model in `nn.DataParallel`, which needs everything on `cuda:0`. **Use 1 GPU.** |
| `401` on `nvidia/Cosmos-Reason2-2B` | no `HF_TOKEN` in the job. Preflight now probes read access |
| `hf_transfer` `ValueError` | the flag is a hard error without the package. Now probed before being set |
| `results=/var/spool/slurm/d/...` | `sbatch` copies the script to the spool; `REPO_ROOT` now prefers `SLURM_SUBMIT_DIR` |
| job pending 8 h | ordinary queue wait behind the senior's jobs on the shared account — not partition tier |
| `Batch job submission failed: Unspecified error` | a submit-filter rule; see CLAUDE.md |

## Open decisions

**Per-task or multi-task?** UniVTAC's ACT trains one policy per task, so
matching it literally means 8 tasks × 2 variants = 16 finetunes. GR00T supports
multi-dataset training, making one multi-task finetune per variant (2 total)
both cheaper and arguably the more honest test of a generalist VLA — but it has
GR00T solving a harder problem than the per-task baselines, which must be stated
plainly rather than buried. Settle this before committing GPU weeks.

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
