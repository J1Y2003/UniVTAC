"""Episode accounting and result files for the ablation.

Design constraints from the guideline: the cluster's login nodes have tight
memory limits and heavy I/O is discouraged. So results are written as
append-only JSONL — one line per episode, flushed as it completes — plus a
single summary JSON at the end. Nothing is accumulated in memory beyond scalar
per-episode records, and no frames are retained.

Writing each episode as it finishes also makes a job that hits its SLURM
walltime still useful: ``summarize_jsonl`` reconstructs the summary from a
partial file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable, Iterator, Mapping


@dataclass
class EpisodeResult:
    """Outcome of a single evaluation episode."""

    seed: int
    success: bool
    reward: float
    steps: int
    """Actions dispatched (``take_action_cnt``)."""
    truncated: bool
    """Hit the step cap rather than terminating."""
    early_stop: bool
    """The task's own ``check_early_stop`` fired."""
    instruction: str = ""
    wall_seconds: float = 0.0
    inferences: int = 0
    inference_seconds: float = 0.0
    error: str | None = None
    """Set when the episode raised; such episodes are excluded from the rate."""
    skipped: str | None = None
    """Set when the seed was unusable rather than failed -- e.g. the task's own
    scripted pre-move could not be planned, so the policy never had a fair
    attempt. Excluded from the success rate, like UniVTAC's expert-check
    rejects, and reported separately."""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ResultWriter:
    """Append-only JSONL writer with an aggregated summary.

    Args:
        path: JSONL destination. Parent directories are created.
        metadata: run-level context (variant, task, checkpoint, horizons) recorded
            in the summary so a results directory is self-describing.

    Example:
        >>> import tempfile, os
        >>> d = tempfile.mkdtemp()
        >>> w = ResultWriter(os.path.join(d, "r.jsonl"), {"variant": "baseline"})
        >>> w.add(EpisodeResult(seed=1, success=True, reward=1.0, steps=10,
        ...                     truncated=False, early_stop=False))
        >>> w.summary()["success_rate"]
        1.0
        >>> _ = w.close()
    """

    def __init__(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self.results: list[EpisodeResult] = []
        self._started = time.time()
        self._handle = self.path.open("a", encoding="utf-8")

    # -- writing -----------------------------------------------------------
    def add(self, result: EpisodeResult) -> None:
        """Record one episode and flush it to disk immediately."""
        self.results.append(result)
        self._handle.write(result.as_json() + "\n")
        self._handle.flush()

    # -- aggregation -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Aggregate scored episodes into the run summary.

        Episodes that raised (``error`` set) are counted separately and left out
        of the success rate, matching UniVTAC's own evaluator, which decrements
        ``test_num`` on an exception rather than scoring it as a failure.
        """
        return summarize(self.results, metadata=self.metadata, wall_seconds=time.time() - self._started)

    def close(self) -> dict[str, Any]:
        """Flush, write ``<stem>.summary.json`` next to the JSONL, and return it."""
        summary = self.summary()
        try:
            self._handle.close()
        except Exception:
            pass
        summary_path = self.path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def summarize(
    results: Iterable[EpisodeResult],
    *,
    metadata: Mapping[str, Any] | None = None,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Build the summary dict for a sequence of episode results."""
    results = list(results)
    skipped = [r for r in results if r.skipped is not None]
    errored = [r for r in results if r.error is not None and r.skipped is None]
    scored = [r for r in results if r.error is None and r.skipped is None]
    successes = [r for r in scored if r.success]

    n = len(scored)
    rate = len(successes) / n if n else 0.0
    summary: dict[str, Any] = {
        **dict(metadata or {}),
        "episodes_scored": n,
        "episodes_errored": len(errored),
        "episodes_skipped": len(skipped),
        "successes": len(successes),
        "success_rate": round(rate, 6),
        "success_rate_pct": round(rate * 100.0, 2),
        # Sparse-success tasks: mean reward equals the success rate by construction.
        "mean_reward": round(sum(r.reward for r in scored) / n, 6) if n else 0.0,
        "mean_steps": round(sum(r.steps for r in scored) / n, 2) if n else 0.0,
        "mean_steps_on_success": (
            round(sum(r.steps for r in successes) / len(successes), 2) if successes else None
        ),
        "truncated": sum(1 for r in scored if r.truncated),
        "early_stopped": sum(1 for r in scored if r.early_stop),
        "total_inferences": sum(r.inferences for r in scored),
        "mean_inference_seconds": (
            round(sum(r.inference_seconds for r in scored) / max(1, sum(r.inferences for r in scored)), 4)
            if any(r.inferences for r in scored)
            else None
        ),
    }
    if n:
        # Wilson score interval: with 20-50 episodes per task the normal
        # approximation is too loose to compare variants honestly.
        lo, hi = wilson_interval(len(successes), n)
        summary["success_rate_ci95"] = [round(lo, 6), round(hi, 6)]
    if wall_seconds is not None:
        summary["wall_seconds"] = round(wall_seconds, 2)
    if skipped:
        reasons: dict[str, int] = {}
        for r in skipped:
            reasons[str(r.skipped)] = reasons.get(str(r.skipped), 0) + 1
        summary["skip_reasons"] = reasons
    if errored:
        summary["errors"] = [
            {"seed": r.seed, "error": r.error[:400] if r.error else None} for r in errored
        ][:20]
    return summary


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial success rate.

    Preferred over the Wald interval because episode counts here are small and
    rates often sit near 0 or 1, where Wald produces bounds outside ``[0, 1]``.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def read_jsonl(path: str | Path) -> Iterator[EpisodeResult]:
    """Stream episode results back out of a JSONL file, line by line."""
    known = set(EpisodeResult.__dataclass_fields__)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            extra = {k: v for k, v in raw.items() if k not in known}
            kept = {k: v for k, v in raw.items() if k in known}
            kept.setdefault("extra", {}).update(extra)
            yield EpisodeResult(**kept)


def summarize_jsonl(path: str | Path, **metadata: Any) -> dict[str, Any]:
    """Re-aggregate a (possibly partial) results file, e.g. after a walltime kill."""
    return summarize(read_jsonl(path), metadata=metadata)


def compare(baseline: Mapping[str, Any], tactile: Mapping[str, Any]) -> dict[str, Any]:
    """Side-by-side of two run summaries, for the ablation table.

    Reports the absolute difference in success rate and flags whether the two
    Wilson intervals overlap. Non-overlapping intervals are a conservative
    signal, not a hypothesis test: with per-task episode counts in the tens,
    prefer reporting both intervals over claiming significance.
    """
    b_rate = float(baseline.get("success_rate", 0.0))
    t_rate = float(tactile.get("success_rate", 0.0))
    b_ci = baseline.get("success_rate_ci95")
    t_ci = tactile.get("success_rate_ci95")

    overlap: bool | None = None
    if b_ci and t_ci:
        overlap = not (b_ci[1] < t_ci[0] or t_ci[1] < b_ci[0])

    return {
        "baseline_success_rate": round(b_rate, 6),
        "tactile_success_rate": round(t_rate, 6),
        "delta": round(t_rate - b_rate, 6),
        "delta_pct_points": round((t_rate - b_rate) * 100.0, 2),
        "baseline_ci95": b_ci,
        "tactile_ci95": t_ci,
        "ci95_overlap": overlap,
        "episodes": {
            "baseline": baseline.get("episodes_scored"),
            "tactile": tactile.get("episodes_scored"),
        },
    }
