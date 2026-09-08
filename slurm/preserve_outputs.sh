#!/bin/bash
# Copy the checkpoints we still need out of the retention-managed unified folder.
#
#   bash slurm/preserve_outputs.sh --check    # what exists, and how close to expiry
#   bash slurm/preserve_outputs.sh            # copy the final checkpoints to KEEP_ROOT
#
# Why this exists. The training-outputs policy ("모델 학습 Outputs 통합 저장 및
# Retention 정책") requires outputs to live under the cluster's unified folder
# (Kakao: /rlwrld-unified-checkpoints/{user}/...), and applies retention there:
# a folder untouched for 4 days (§5.1 body; the §5.1 table says 7 -- the document
# contradicts itself, so treat 4 as the deadline) is moved to Object Storage and
# removed from NFS, then deleted 90 days after that.
#
# Our study runs for weeks across 8 tasks. A lift_bottle checkpoint we want to
# re-evaluate later will simply not be on NFS any more, and <output_dir>/final
# will be a dangling symlink. §7 is explicit about the remedy: store outputs in
# the unified folder first, then move anything needing permanent retention into
# your own user folder.
#
# So this copies only the *final* checkpoint per variant -- the one evaluation
# uses -- and leaves the intermediates to expire. That is deliberate: a full 3B
# checkpoint with optimizer state is ~40 GB, and keeping every one of them would
# recreate the NFS sprawl the policy is trying to prevent.

set -uo pipefail

WHOAMI="${USER:-$(id -un)}"

# Retention-managed. Outputs must be written here first (policy §3, §7).
CKPT_ROOT="${CKPT_ROOT:-${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/${WHOAMI}/checkpoints/univtac-groot}}"
# Your own space, outside the retention sweep. Not the unified folder.
KEEP_ROOT="${KEEP_ROOT:-${HOME}/jaewon/workspace/groot-keep}"

TASK="${TASK:-lift_bottle}"
VARIANTS="${VARIANTS:-tactile baseline_finetuned}"
# Days after which the unified folder's contents are archived off NFS.
RETENTION_DAYS="${RETENTION_DAYS:-4}"

MODE="${1:-copy}"

# Resolve the checkpoint evaluation actually used: the `final` symlink written by
# overnight_ablation.sbatch, else the highest-numbered checkpoint-N.
resolve_final() {
  local out_dir="$1" latest="" d n
  if [[ -e "${out_dir}/final" ]]; then
    readlink -f "${out_dir}/final"
    return
  fi
  for d in "${out_dir}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    n="${d##*checkpoint-}"
    [[ "${n}" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${latest}" || "${n}" -gt "${latest}" ]]; then latest="${d}"; fi
  done
  printf '%s' "${latest}"
}

echo "unified (retention-managed): ${CKPT_ROOT}"
echo "keep (yours, no sweep):      ${KEEP_ROOT}"
echo

STATUS=0
for variant in ${VARIANTS}; do
  out_dir="${CKPT_ROOT}/${TASK}-${variant}"
  if [[ ! -d "${out_dir}" ]]; then
    echo "${TASK}/${variant}: nothing at ${out_dir}"
    continue
  fi

  final=$(resolve_final "${out_dir}")
  if [[ -z "${final}" || ! -d "${final}" ]]; then
    echo "${TASK}/${variant}: no checkpoint yet"
    continue
  fi

  # Age from mtime. The policy prefers atime, but atime is frequently disabled
  # (relatime/noatime) on network filesystems, and mtime is the conservative
  # choice: it only ever makes a folder look OLDER than atime would, so a
  # warning here can be early but never late.
  age_days=$(( ( $(date +%s) - $(stat -c %Y "${final}" 2>/dev/null || echo 0) ) / 86400 ))
  size=$(du -sh "${final}" 2>/dev/null | cut -f1)
  left=$(( RETENTION_DAYS - age_days ))

  printf '%s/%s\n' "${TASK}" "${variant}"
  printf '  final    %s\n' "${final}"
  printf '  size     %s\n' "${size:-?}"
  if [[ "${left}" -le 0 ]]; then
    printf '  age      %s days -- PAST the %s-day window; may already be archived\n' \
      "${age_days}" "${RETENTION_DAYS}"
  else
    printf '  age      %s days (%s day(s) before archival)\n' "${age_days}" "${left}"
  fi

  dest="${KEEP_ROOT}/${TASK}-${variant}"
  if [[ "${MODE}" == "--check" ]]; then
    if [[ -d "${dest}" ]]; then
      printf '  kept     yes -> %s\n' "${dest}"
    else
      printf '  kept     NO  (run without --check to copy)\n'
    fi
    echo
    continue
  fi

  if [[ -d "${dest}" ]]; then
    echo "  kept     already present at ${dest}; skipping"
    echo
    continue
  fi

  mkdir -p "${dest}"
  # Copy, do not move: the policy wants outputs written to the unified folder,
  # and moving them out would also break the eval stage's `final` symlink while
  # a run is still in progress.
  if cp -a "${final}/." "${dest}/"; then
    echo "  kept     copied -> ${dest}"
    # Record which step this was; the directory name is lost by the copy.
    printf 'source=%s\ncopied=%s\ntask=%s\nvariant=%s\n' \
      "${final}" "$(date -Iseconds)" "${TASK}" "${variant}" > "${dest}/PROVENANCE"
  else
    echo "  kept     COPY FAILED -- check space with: df -h ${KEEP_ROOT}" >&2
    STATUS=1
  fi
  echo
done

if [[ "${MODE}" != "--check" ]]; then
  echo "Note: only the FINAL checkpoint per variant is copied. Intermediates are"
  echo "left to expire on purpose -- each is ~40 GB with optimizer state, and"
  echo "hoarding them all would recreate the NFS sprawl the policy prevents."
  echo
  echo "Re-evaluate a preserved checkpoint with:"
  echo "  GROOT_MODEL=${KEEP_ROOT}/${TASK}-<variant> VARIANT=<variant> TASK=${TASK} \\"
  echo "    bash slurm/eval_ablation.sbatch"
fi
exit ${STATUS}
