#!/bin/bash
# Shared helper for submitting through `bundle-sbatch`, sourced by the
# submitters and the job scripts here. Not executable on its own.
#
# bundle-sbatch wraps file-based sbatch: one output bundle per submission, a
# snapshot of the script and git state, injected output/checkpoint environment,
# then one Slurm request. Its command line has three regions:
#
#   bundle-sbatch <wrapper options> -- <slurm options> -- <script> [args]
#
# The two option regions take OPPOSITE forms, which is the easy mistake:
# wrapper options are separate tokens (`--job-kind train`), Slurm options are
# attached long form (`--partition=background`).
#
# The launcher owns --output, --error, --open-mode, --export, --export-file,
# --get-user-env, --parsable, --quiet, --wait, --test-only, --wrap, --clusters
# and arrays (which go through its own `--array SPEC`). It also owns the five
# environment variables below and refuses to run if it finds one already set,
# so every submission here scrubs them.
BUNDLE_OWNED_ENV=(
  OUTPUT_DIR
  JOB_OUTPUT_BUNDLE_DIR
  CODE_OUTPUT_DIR
  MODEL_OUTPUT_DIR
  CHECKPOINT_DIR
)

# bundle_scrub_args -- `env -u` flags for the variables the launcher owns.
bundle_scrub_args() {
  local var
  BUNDLE_SCRUB=()
  for var in "${BUNDLE_OWNED_ENV[@]}"; do
    BUNDLE_SCRUB+=(-u "${var}")
  done
}

# bundle_warn_inherited -- say so once, so a stale env.sh gets fixed rather than
# silently worked around on every future run.
bundle_warn_inherited() {
  local var found=()
  for var in "${BUNDLE_OWNED_ENV[@]}"; do
    [[ -n "${!var:-}" ]] && found+=("${var}")
  done
  ((${#found[@]})) || return 0
  echo "note: ${found[*]} set in this shell; scrubbing for the launcher," >&2
  echo "      which injects them itself. Remove them from env.sh." >&2
}

# bundle_build <script> [args...] -- validate and assemble BUNDLE_CMD, the full
# argv, from BUNDLE_JOB_KIND, BUNDLE_CHECKPOINT, BUNDLE_GIT_ROOT,
# BUNDLE_SLURM_ARGS and the optional BUNDLE_ARRAY. Returns 2 on a contract
# violation, so a bad call fails here rather than after the launcher has
# recorded a submission.
bundle_build() {
  local script="$1"; shift
  local kind="${BUNDLE_JOB_KIND:-}" ckpt="${BUNDLE_CHECKPOINT:-}"
  local git_root="${BUNDLE_GIT_ROOT:-}"

  case "${kind}" in
    train|eval)
      if [[ -z "${ckpt}" ]]; then
        echo "bundle_build: --job-kind ${kind} requires BUNDLE_CHECKPOINT" >&2
        return 2
      fi
      ;;
    data_process|other)
      if [[ -n "${ckpt}" ]]; then
        echo "bundle_build: --job-kind ${kind} forbids a checkpoint (got '${ckpt}')" >&2
        return 2
      fi
      ;;
    *)
      echo "bundle_build: BUNDLE_JOB_KIND must be train|eval|data_process|other" >&2
      return 2
      ;;
  esac

  # `from_scratch` is the only non-path checkpoint value, and only for training.
  if [[ -n "${ckpt}" && "${ckpt}" != "from_scratch" ]]; then
    if [[ -L "${ckpt}" ]]; then
      echo "bundle_build: checkpoint '${ckpt}' is a symlink, which the launcher" >&2
      echo "  rejects. Pass the physical directory it points at:" >&2
      echo "    readlink -f '${ckpt}'" >&2
      return 2
    fi
    if [[ ! -d "${ckpt}" ]]; then
      echo "bundle_build: checkpoint '${ckpt}' is not an existing directory" >&2
      return 2
    fi
  fi
  if [[ "${kind}" == "eval" && "${ckpt}" == "from_scratch" ]]; then
    echo "bundle_build: --job-kind eval needs a real checkpoint, not from_scratch" >&2
    return 2
  fi

  if [[ ! -d "${git_root}/.git" ]]; then
    echo "bundle_build: BUNDLE_GIT_ROOT='${git_root}' is not a repository root." >&2
    echo "  It must be the exact root, not a subdirectory." >&2
    return 2
  fi
  if [[ ! -f "${script}" || -L "${script}" ]]; then
    echo "bundle_build: script '${script}' must be a regular, non-symlink file" >&2
    return 2
  fi

  # Nothing the launcher owns may appear in the Slurm region.
  local arg
  for arg in "${BUNDLE_SLURM_ARGS[@]}"; do
    case "${arg}" in
      --output*|--error*|--open-mode*|--export*|--get-user-env*|\
      --parsable|--quiet|--wait|--test-only|--wrap*|--clusters*|--array*)
        echo "bundle_build: '${arg}' is owned by the launcher, not by us" >&2
        return 2
        ;;
      --*) ;;
      *)
        # A bare token here is a detached value, e.g. (--partition background),
        # which the Slurm region does not accept.
        echo "bundle_build: '${arg}' is a detached value or short option." >&2
        echo "  The Slurm region takes attached long form only: --name=value" >&2
        return 2
        ;;
    esac
  done

  BUNDLE_CMD=(bundle-sbatch --job-kind "${kind}")
  [[ -n "${ckpt}" ]] && BUNDLE_CMD+=(--checkpoint "${ckpt}")
  BUNDLE_CMD+=(--code-git-root "${git_root}")
  [[ -n "${BUNDLE_ARRAY:-}" ]] && BUNDLE_CMD+=(--array "${BUNDLE_ARRAY}")
  BUNDLE_CMD+=(--)
  BUNDLE_CMD+=("${BUNDLE_SLURM_ARGS[@]}")
  BUNDLE_CMD+=(-- "${script}" "$@")
  return 0
}

# bundle_submit <script> [args...] -- build and run one submission. Sets
# BUNDLE_JOBID when an id could be parsed out, empty otherwise, since
# --parsable is the launcher's and the id is whatever it chooses to print.
#
# NEVER retries: once a submission has started, a fresh invocation for the same
# request is forbidden even when the outcome is unclear. Read the diagnostic
# and the retained bundle instead.
bundle_submit() {
  BUNDLE_JOBID=""
  declare -p BUNDLE_ENV >/dev/null 2>&1 || BUNDLE_ENV=()
  bundle_build "$@" || return 2

  # BUNDLE_ENV carries the job's configuration to the launcher, which is how it
  # reaches the job now that --export belongs to the launcher. It is applied to
  # the launcher process alone, so one submitter can send different values per
  # submission without leaking them into its own environment.
  local out status
  bundle_warn_inherited
  bundle_scrub_args
  out=$(env "${BUNDLE_SCRUB[@]}" "${BUNDLE_ENV[@]}" "${BUNDLE_CMD[@]}" 2>&1)
  status=$?
  printf '%s\n' "${out}"
  [[ ${status} -ne 0 ]] && return ${status}

  # "Submitted batch job 12345", a bare id, or nothing at all.
  BUNDLE_JOBID=$(printf '%s\n' "${out}" \
    | grep -oE '(Submitted batch job |job[ _-]?id[ :=]+)?[0-9]{3,}' \
    | grep -oE '[0-9]{3,}' | tail -1)
  return 0
}

# --------------------------------------------------------------------------- #
# bundle_print <script> [script args...] -- render the command, submit nothing.
# --------------------------------------------------------------------------- #
bundle_print() {
  declare -p BUNDLE_ENV >/dev/null 2>&1 || BUNDLE_ENV=()
  bundle_build "$@" || return 2
  # printf reuses its format for every argument, so `env` is printed once and
  # the pairs after it -- not `env X env Y`.
  bundle_scrub_args
  printf 'env '
  printf '%s ' "${BUNDLE_SCRUB[@]}"
  [[ ${#BUNDLE_ENV[@]} -gt 0 ]] && printf '%q ' "${BUNDLE_ENV[@]}"
  printf '%q ' "${BUNDLE_CMD[@]}"
  printf '\n'
}

# --------------------------------------------------------------------------- #
# bundle_require -- fail early if the wrapper is not on PATH, with the reason
# this repo now depends on it.
# --------------------------------------------------------------------------- #
bundle_require() {
  command -v bundle-sbatch >/dev/null 2>&1 && return 0
  echo "bundle-sbatch is not on PATH." >&2
  echo "  Every job here is submitted through it -- plain sbatch is no longer" >&2
  echo "  the supported path. See https://github.com/RLWRLD/bundle-sbatch" >&2
  return 2
}

# univtac_job_guard -- the two checks every job script runs: the wckey and the
# configuration sentinel. Both no-op outside Slurm, so a local dry run is
# unaffected.
univtac_job_guard() {
  # An exported SBATCH_WCKEY outranks what the submitter passed on the command
  # line, and a job once went out under the wrong project exactly that way.
  # SLURM_WCKEY is set only when a wckey was applied, so unset means valid.
  local want="project-short-name:sub_4dpdata"
  if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_WCKEY:-${want}}" != "${want}" ]]; then
    echo "error: this job's wckey is '${SLURM_WCKEY}', not '${want}'." >&2
    echo "       Almost certainly a stale SBATCH_WCKEY in env.sh. Fix with:" >&2
    echo "         export SBATCH_WCKEY=${want}" >&2
    echo "       and do NOT export SLURM_WCKEY. scripts/preflight.py checks this." >&2
    exit 2
  fi

  # bundle-sbatch owns --export, so a job's configuration reaches it only by
  # environment inheritance. If that ever stops holding, every variable would
  # fall back to its default and the job would do the WRONG WORK while looking
  # healthy. The submitters set this sentinel; a real job without it stops.
  if [[ -n "${SLURM_JOB_ID:-}" && -z "${UNIVTAC_JOB_CONFIG:-}" ]]; then
    echo "error: UNIVTAC_JOB_CONFIG is unset, so this job's environment did not" >&2
    echo "       reach it and its configuration would silently take defaults." >&2
    echo "       Submit through the scripts in slurm/, which set it." >&2
    exit 2
  fi
}
