#!/bin/bash
# Shared helper for submitting through `bundle-sbatch`, sourced by the
# submitters in this directory. Not executable on its own.
#
# bundle-sbatch is a managed wrapper around file-based sbatch: it creates one
# output bundle per submission, snapshots the script and git state, injects the
# output/checkpoint/bundle environment, and then submits ONE Slurm request.
# Its command line has three regions separated by `--`:
#
#   bundle-sbatch <wrapper options> -- <slurm options> -- <script> [args]
#
# The two regions follow OPPOSITE conventions, which is the easiest thing to
# get wrong:
#
#   wrapper options   separate tokens      --job-kind train
#   slurm options     attached long form   --partition=background
#
# `-x value` and `--name value` are both invalid in the Slurm region.
#
# What the launcher owns, and what we therefore must not pass anywhere:
#
#   --output --error --open-mode          log paths
#   --export --export-file --get-user-env the job environment
#   --parsable --quiet --wait --test-only result reporting
#   --wrap --clusters --array=...         unsupported submission modes
#
# An array goes through the wrapper's own `--array SPEC` (separate tokens, once,
# before the first `--`). We submit no arrays today.

# --------------------------------------------------------------------------- #
# bundle_build <script> [script args...]
#
# Reads:
#   BUNDLE_JOB_KIND    train | eval | data_process | other
#   BUNDLE_CHECKPOINT  required for train (from_scratch or a PATH) and for eval
#                      (a PATH); must be EMPTY for data_process and other
#   BUNDLE_GIT_ROOT    exact repository root, not a subdirectory
#   BUNDLE_SLURM_ARGS  array of attached-long-form Slurm options
#   BUNDLE_ARRAY       optional array SPEC
#
# Writes BUNDLE_CMD, the full argv. Returns 2 on a contract violation, so a bad
# call fails here rather than after the launcher has recorded a submission.
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# bundle_submit <script> [script args...]
#
# Builds and runs one submission. Sets BUNDLE_JOBID when a job id could be read
# back, empty otherwise -- `--parsable` belongs to the launcher, so the id is
# whatever it chooses to print and must be treated as a bonus, not a contract.
#
# NEVER retries. The launcher's rule is explicit: once a submission has started,
# a fresh invocation for the same request is forbidden even when the outcome is
# unclear. On a non-zero exit, read the diagnostic and the retained bundle.
# --------------------------------------------------------------------------- #
bundle_submit() {
  BUNDLE_JOBID=""
  declare -p BUNDLE_ENV >/dev/null 2>&1 || BUNDLE_ENV=()
  bundle_build "$@" || return 2

  # BUNDLE_ENV carries the job's configuration to the launcher, which is how it
  # reaches the job now that --export belongs to the launcher. It is applied to
  # the launcher process alone, so one submitter can send different values per
  # submission without leaking them into its own environment.
  local out status
  if [[ ${#BUNDLE_ENV[@]} -gt 0 ]]; then
    out=$(env "${BUNDLE_ENV[@]}" "${BUNDLE_CMD[@]}" 2>&1)
  else
    out=$("${BUNDLE_CMD[@]}" 2>&1)
  fi
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
  if [[ ${#BUNDLE_ENV[@]} -gt 0 ]]; then
    printf 'env '
    printf '%q ' "${BUNDLE_ENV[@]}"
  fi
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
