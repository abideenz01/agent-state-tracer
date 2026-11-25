"""Sampling policies, replayed offline over a full export, plus a Collector config.

Head sampling decides when a run starts, before anyone knows whether it will
fail, take a rare path or split from its last run. Tail sampling decides after
the run ends. The tracer already writes a ``agent_state.sample_hint`` on the
root span, so the Collector only needs plain attribute and status rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import conventions as c
from .loader import SpanRec

_TWO64 = 2**64


def trace_ratio_keep(trace_id: str, rate: float) -> bool:
    """Same idea as the SDK's TraceIdRatioBased sampler: compare the low 64 bits of the id."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    return (int(trace_id, 16) & (_TWO64 - 1)) < int(rate * _TWO64)


@dataclass(frozen=True)
class HeadPolicy:
    rate: float

    def keep(self, trace_id: str, spans: list[SpanRec]) -> bool:
        return trace_ratio_keep(trace_id, self.rate)


@dataclass(frozen=True)
class TailPolicy:
    rate: float = 0.05
    keep_hints: tuple[str, ...] = c.KEEP_HINTS
    keep_error_status: bool = True

    def keep(self, trace_id: str, spans: list[SpanRec]) -> bool:
        if self.keep_error_status and any(s.status == "ERROR" for s in spans):
            return True
        if any(s.attributes.get(c.SAMPLE_HINT) in self.keep_hints for s in spans):
            return True
        # Use a different slice of the id than HeadPolicy so the two baselines are independent.
        return (int(trace_id, 16) >> 64) < int(self.rate * _TWO64)


def apply_policy(spans: Iterable[SpanRec], policy) -> set[str]:
    by_trace: dict[str, list[SpanRec]] = {}
    for s in spans:
        by_trace.setdefault(s.trace_id, []).append(s)
    return {tid for tid, group in by_trace.items() if policy.keep(tid, group)}


def collector_config(rate: float = 0.05, decision_wait: str = "30s") -> str:
    """OpenTelemetry Collector (contrib) tail_sampling snippet that mirrors TailPolicy.

    Policies are OR-ed: a trace is kept if any policy says sample. All spans of
    a trace must reach the same Collector instance (use a load-balancing
    exporter keyed on trace id in front of a Collector pool).
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    pct = round(rate * 100, 4)
    hints = ", ".join(f'"{h}"' for h in c.KEEP_HINTS)
    return f"""processors:
  tail_sampling:
    decision_wait: {decision_wait}
    num_traces: 50000
    policies:
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: keep-agent-hints
        type: string_attribute
        string_attribute:
          key: {c.SAMPLE_HINT}
          values: [{hints}]
      - name: baseline-sample
        type: probabilistic
        probabilistic:
          sampling_percentage: {pct}
"""
