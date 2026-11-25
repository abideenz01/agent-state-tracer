"""Instrument an agent's state machine with OpenTelemetry spans.

Span layout for one run::

    invoke_agent <agent>                 root, summary of the whole run
      agent_state PLAN                   one span per state, covers time spent in it
        execute_tool orders_api          tool calls made while in that state
      agent_state ANSWER

Each state span records the transition INTO that state: from, to, step index,
reason code, who decided (model, rule, tool, guard) and payload-free evidence
fingerprints. Transitions are spans, not span events, because the OpenTelemetry
Span Event API is being deprecated in favour of log-based events.

When the run ends, the root span gets the path, a path signature, an outcome
and a sampling hint. If the previous run of the same input took a different
path, the root span also carries the fork step, the reference path and the
evidence fields that changed first. That makes a kept divergent trace
explainable on its own, even if the reference trace was sampled away.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from . import conventions as c
from .fingerprint import encode_evidence, fingerprint, path_signature
from .fork import StepRecord, find_fork

ALLOWED_DECIDERS = frozenset({"model", "rule", "tool", "guard", "human"})
_STATE_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _check_state(name: str) -> str:
    if not name or len(name) > 40 or not set(name) <= _STATE_OK:
        raise ValueError(f"state names must be UPPER_SNAKE_CASE, max 40 chars, got {name!r}")
    return name


def _check_code(name: str, what: str) -> str:
    if not name or len(name) > 64 or not all(ch.isalnum() or ch in "_:.-" for ch in name):
        raise ValueError(f"{what} must be a short code (letters, digits, _ : . -), got {name!r}")
    return name


@dataclass
class RunDigest:
    """Tiny, payload-free summary of one run kept in memory for comparison."""

    trace_id: str
    steps: tuple[StepRecord, ...]

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(s.to_state for s in self.steps)


class DigestStore:
    """Bounded LRU of the latest run digest per run key, plus path counts.

    Per process. With many replicas, runs of the same input may land on
    different processes and a divergence can be missed; back this with a
    shared store if that matters for your workload.
    """

    def __init__(self, max_keys: int = 10_000, max_paths: int = 50_000):
        if max_keys < 1 or max_paths < 1:
            raise ValueError("max_keys and max_paths must be positive")
        self._digests: OrderedDict[str, RunDigest] = OrderedDict()
        self._paths: OrderedDict[str, int] = OrderedDict()
        self._max_keys = max_keys
        self._max_paths = max_paths
        self._lock = threading.Lock()

    def swap(self, run_key: str, digest: RunDigest) -> RunDigest | None:
        with self._lock:
            prev = self._digests.pop(run_key, None)
            self._digests[run_key] = digest
            while len(self._digests) > self._max_keys:
                self._digests.popitem(last=False)
            return prev

    def bump_path(self, sig: str) -> int:
        """Increment and return the count seen BEFORE this run."""
        with self._lock:
            before = self._paths.pop(sig, 0)
            self._paths[sig] = before + 1
            while len(self._paths) > self._max_paths:
                self._paths.popitem(last=False)
            return before


class ToolCall:
    def __init__(self, run: "AgentRun", span: Span, name: str):
        self._run, self._span, self._name = run, span, name

    def record_result(self, result: Any) -> None:
        """Fingerprint the result and add it to the evidence of the next transition."""
        self._run._pending[f"tool:{self._name}"] = result

    def __enter__(self) -> "ToolCall":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._span.set_status(Status(StatusCode.ERROR, exc_type.__name__))
            self._run._pending[f"tool:{self._name}"] = {"error": exc_type.__name__}
        self._span.end(end_time=self._run._tracer._clock())
        return False


class AgentRun:
    """One agent run. Use through ``StateTracer.run`` as a context manager."""

    def __init__(self, tracer: "StateTracer", run_key: str, input_value: Any, conversation_id: str | None):
        self._tracer = tracer
        self.run_key = _check_code(run_key, "run_key")
        self._pending: dict[str, Any] = {}
        self._steps: list[StepRecord] = []
        self._state_span: Span | None = None
        self._outcome: str | None = None
        self._done = False
        self.outcome: str | None = None
        self.path: tuple[str, ...] = ()
        self.hint: str | None = None
        self.fork = None
        attrs = {
            c.GEN_AI_OPERATION_NAME: c.OP_INVOKE_AGENT,
            c.GEN_AI_AGENT_NAME: tracer.agent_name,
            c.RUN_KEY: run_key,
        }
        if conversation_id:
            attrs[c.GEN_AI_CONVERSATION_ID] = conversation_id
        self._input_fp = fingerprint(input_value)
        attrs[c.INPUT_FP] = self._input_fp
        self._root = tracer._otel.start_span(
            f"{c.OP_INVOKE_AGENT} {tracer.agent_name}",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
            start_time=tracer._clock(),
        )
        self._root_ctx = trace.set_span_in_context(self._root)
        self._pending["input"] = input_value

    # ------------------------------------------------------------------ API
    @property
    def state(self) -> str:
        return self._steps[-1].to_state if self._steps else c.START_STATE

    @property
    def trace_id(self) -> str:
        return format(self._root.get_span_context().trace_id, "032x")

    def observe(self, field_name: str, value: Any) -> None:
        """Add evidence that will be attached to the next transition."""
        self._pending[field_name] = value

    def transition(
        self,
        to_state: str,
        *,
        reason: str,
        decided_by: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if self._done:
            raise RuntimeError("run already finished")
        _check_state(to_state)
        _check_code(reason, "reason")
        if decided_by not in ALLOWED_DECIDERS:
            raise ValueError(f"decided_by must be one of {sorted(ALLOWED_DECIDERS)}")
        merged = dict(self._pending)
        if evidence:
            merged.update(evidence)
        self._pending.clear()
        encoded = encode_evidence(merged)
        now = self._tracer._clock()
        if self._state_span is not None:
            self._state_span.end(end_time=now)
        step = len(self._steps)
        attrs: dict[str, Any] = {
            c.STEP: step,
            c.FROM: self.state,
            c.TO: to_state,
            c.REASON: reason,
            c.DECIDED_BY: decided_by,
        }
        for k, v in encoded.items():
            attrs[c.EVIDENCE_PREFIX + k] = v
        if decided_by == "model":
            if "model" in encoded:
                attrs[c.GEN_AI_REQUEST_MODEL] = str(encoded["model"])
            if isinstance(encoded.get("temperature"), (int, float)):
                attrs[c.GEN_AI_REQUEST_TEMPERATURE] = float(encoded["temperature"])
        self._state_span = self._tracer._otel.start_span(
            c.STATE_SPAN_PREFIX + to_state,
            context=self._root_ctx,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
            start_time=now,
        )
        self._steps.append(StepRecord(to_state, encoded, decided_by))

    def tool(self, name: str, call_id: str | None = None) -> ToolCall:
        _check_code(name, "tool name")
        parent = self._state_span or self._root
        attrs = {c.GEN_AI_OPERATION_NAME: c.OP_EXECUTE_TOOL, c.GEN_AI_TOOL_NAME: name}
        if call_id:
            attrs[c.GEN_AI_TOOL_CALL_ID] = call_id
        span = self._tracer._otel.start_span(
            f"{c.OP_EXECUTE_TOOL} {name}",
            context=trace.set_span_in_context(parent),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
            start_time=self._tracer._clock(),
        )
        return ToolCall(self, span, name)

    def fail(self, reason: str, error_type: str | None = None) -> None:
        """Move to FAIL and mark the run as an error. Only the error type is recorded, never the message."""
        self.transition("FAIL", reason=reason, decided_by="guard")
        desc = error_type or reason
        if self._state_span is not None:
            self._state_span.set_status(Status(StatusCode.ERROR, desc))
        self._root.set_status(Status(StatusCode.ERROR, desc))
        self._outcome = c.OUTCOME_ERROR

    def abort(self, reason: str) -> None:
        """Stop early with a labelled partial outcome (for example a deadline or budget stop)."""
        self.transition("STOPPED", reason=reason, decided_by="guard")
        self._outcome = c.OUTCOME_PARTIAL

    # ------------------------------------------------------------ internals
    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        now = self._tracer._clock()
        if self._state_span is not None:
            self._state_span.end(end_time=now)
        if not self._steps:
            self._outcome = c.OUTCOME_PARTIAL
        outcome = self._outcome or c.OUTCOME_OK
        path = tuple(s.to_state for s in self._steps)
        sig = path_signature(path)
        root = self._root
        root.set_attribute(c.PATH, ">".join(path))
        root.set_attribute(c.PATH_SIG, sig)
        root.set_attribute(c.STEPS, len(path))
        root.set_attribute(c.TERMINAL, path[-1] if path else c.START_STATE)
        root.set_attribute(c.OUTCOME, outcome)

        store = self._tracer.store
        seen_before = store.bump_path(sig)
        root.set_attribute(c.PATH_SEEN, seen_before)
        prev = store.swap(self.run_key, RunDigest(self.trace_id, tuple(self._steps)))
        fork = None
        if prev is not None and prev.path != path:
            fork = find_fork(prev.steps, self._steps)
        if fork is not None:
            root.set_attribute(c.REF_PATH, ">".join(prev.path))
            root.set_attribute(c.REF_TRACE_ID, prev.trace_id)
            root.set_attribute(c.FORK_STEP, fork.fork_step)
            root.set_attribute(c.FORK_FROM, fork.from_state)
            root.set_attribute(c.FORK_TO_REF, fork.to_a)
            root.set_attribute(c.FORK_TO, fork.to_b)
            root.set_attribute(c.FORK_CAUSE, fork.cause_label)
            if fork.cause_step is not None:
                root.set_attribute(c.FORK_CAUSE_STEP, fork.cause_step)

        if outcome != c.OUTCOME_OK:
            hint = c.HINT_ERROR
        elif fork is not None and fork.cause_label not in self._tracer.quiet_fork_causes:
            hint = c.HINT_DIVERGENT
        elif seen_before < self._tracer.rare_min_count:
            hint = c.HINT_RARE
        else:
            hint = c.HINT_COMMON
        root.set_attribute(c.SAMPLE_HINT, hint)
        root.end(end_time=now)
        self.outcome, self.path, self.hint, self.fork = outcome, path, hint, fork

    def __enter__(self) -> "AgentRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and not self._done and self._outcome is None:
            self.fail("unhandled_exception", exc_type.__name__)
        self._finish()
        return False


class StateTracer:
    """Factory for traced agent runs.

    Parameters
    ----------
    agent_name: value for ``gen_ai.agent.name``.
    tracer_provider: an OpenTelemetry TracerProvider; defaults to the global one.
    store: shared DigestStore (one per process is the normal setup).
    rare_min_count: every distinct path keeps its first N runs as exemplars.
    quiet_fork_causes: fork causes that do NOT earn a "divergent" hint. Default
        is model sampling: with temperature above zero, splits are expected
        and would flood the sampler. Splits caused by a change are kept.
    clock: callable returning nanoseconds; injectable for tests and replays.
    """

    def __init__(
        self,
        agent_name: str,
        tracer_provider: Any = None,
        store: DigestStore | None = None,
        rare_min_count: int = 3,
        clock: Callable[[], int] | None = None,
        quiet_fork_causes: tuple[str, ...] = (c.CAUSE_SAMPLING,),
    ):
        self.agent_name = _check_code(agent_name, "agent_name")
        if rare_min_count < 0:
            raise ValueError("rare_min_count must be >= 0")
        provider = tracer_provider or trace.get_tracer_provider()
        self._otel = provider.get_tracer("agent_state_tracer", "0.1.0")
        self.store = store or DigestStore()
        self.rare_min_count = rare_min_count
        self.quiet_fork_causes = tuple(quiet_fork_causes)
        self._clock = clock or time.time_ns

    def run(self, run_key: str, input_value: Any, conversation_id: str | None = None) -> AgentRun:
        """``run_key`` must be the same for every run of the same logical input (for example a hash of the normalised request)."""
        return AgentRun(self, run_key, input_value, conversation_id)
