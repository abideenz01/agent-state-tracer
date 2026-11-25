"""Find where two runs of the same input split, and name what changed first.

Used in two places with the same logic:
  * in-process, when a run finishes and the previous run of the same input took
    a different path (the result is stamped on the root span);
  * offline, by the analyzer over exported traces.

Rule: the fork is the first step whose target state differs. The cause is the
set of evidence fields that differ at the EARLIEST step (at or before the
fork) where any evidence differs. Diffing only the fork step blames
downstream symptoms (for example retrieved ids that changed because a tool
result changed two steps earlier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .conventions import CAUSE_SAMPLING, CAUSE_UNRECORDED
from .fingerprint import SAFE_RAW_FIELDS


@dataclass(frozen=True)
class StepRecord:
    """What one state transition looked like, without payloads."""

    to_state: str
    evidence: Mapping[str, object] = field(default_factory=dict)
    decided_by: str = ""


@dataclass(frozen=True)
class ForkResult:
    fork_step: int  # index of the first differing state
    from_state: str  # state both runs were in before the split
    to_a: str  # where run A went ("<end>" if it stopped)
    to_b: str
    cause: tuple[str, ...]  # sorted field names, or (sampling,) / (unrecorded,)
    cause_step: int | None  # earliest step with an evidence difference
    fork_step_fields: tuple[str, ...]  # fields that differ at the fork step only (naive view)

    @property
    def cause_label(self) -> str:
        return ",".join(self.cause)


END = "<end>"


def _diff_fields(a: StepRecord, b: StepRecord) -> tuple[str, ...]:
    """Evidence fields that differ between two aligned steps.

    Model config fields (model, prompt_version, temperature ...) are only
    recorded on model decisions. When one side was decided by a model and the
    other by a guard or rule, those fields are missing for a boring reason, so
    they are left out of the comparison.
    """
    ea, eb = a.evidence, b.evidence
    keys = set(ea) | set(eb)
    if a.decided_by and b.decided_by and a.decided_by != b.decided_by:
        keys -= SAFE_RAW_FIELDS
    return tuple(sorted(k for k in keys if ea.get(k, None) != eb.get(k, None)))


def _temperature(steps: Sequence[StepRecord], upto: int) -> float:
    temp = 0.0
    for s in steps[: upto + 1]:
        t = s.evidence.get("temperature")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            temp = max(temp, float(t))
    return temp


def find_fork(a: Sequence[StepRecord], b: Sequence[StepRecord]) -> ForkResult | None:
    """Return None when both runs visited the same states in the same order."""
    n = min(len(a), len(b))
    k = None
    for i in range(n):
        if a[i].to_state != b[i].to_state:
            k = i
            break
    if k is None:
        if len(a) == len(b):
            return None
        k = n  # one run stopped earlier
    from_state = a[k - 1].to_state if k > 0 else "START"
    to_a = a[k].to_state if k < len(a) else END
    to_b = b[k].to_state if k < len(b) else END

    cause: tuple[str, ...] = ()
    cause_step = None
    last = min(k, n - 1)
    for i in range(0, last + 1):
        d = _diff_fields(a[i], b[i])
        if d:
            cause, cause_step = d, i
            break
    fork_fields = _diff_fields(a[k], b[k]) if k < n else ()
    if not cause:
        hot = max(_temperature(a, last), _temperature(b, last))
        cause = (CAUSE_SAMPLING,) if hot > 0 else (CAUSE_UNRECORDED,)
    return ForkResult(k, from_state, to_a, to_b, cause, cause_step, fork_fields)
