"""Rebuild agent runs from exported spans, check they are complete, and explain path splits.

Three questions this answers over a trace export:
  1. Which traces are incomplete (dropped spans, missing root, broken chain)?
     Path statistics over incomplete traces are wrong, so they are reported
     and excluded rather than silently counted.
  2. For each input (run key), how many different paths did the agent take?
  3. Where each different path split from the previous run of the same input,
     and which recorded evidence changed first.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from . import conventions as c
from .fork import ForkResult, StepRecord, find_fork
from .loader import SpanRec

ISSUE_NO_ROOT = "no_root"
ISSUE_STEP_GAP = "step_gap"
ISSUE_BROKEN_CHAIN = "broken_chain"
ISSUE_PATH_MISMATCH = "path_mismatch"
ISSUE_ORPHAN = "orphan_span"
ISSUE_DUP_STEP = "duplicate_step"
ISSUE_NO_SUMMARY = "no_summary"


@dataclass
class RunView:
    trace_id: str
    root: SpanRec | None
    states: list[SpanRec]
    tools: list[SpanRec]
    issues: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.issues

    @property
    def run_key(self) -> str | None:
        return self.root.attributes.get(c.RUN_KEY) if self.root else None

    @property
    def start_ns(self) -> int:
        if self.root:
            return self.root.start_ns
        return min((s.start_ns for s in self.states), default=0)

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(s.attributes.get(c.TO, "?") for s in self.states)

    @property
    def outcome(self) -> str:
        return str(self.root.attributes.get(c.OUTCOME, "unknown")) if self.root else "unknown"

    def step_records(self) -> list[StepRecord]:
        out = []
        n = len(c.EVIDENCE_PREFIX)
        for s in self.states:
            ev = {k[n:]: v for k, v in s.attributes.items() if k.startswith(c.EVIDENCE_PREFIX)}
            out.append(StepRecord(str(s.attributes.get(c.TO, "?")), ev, str(s.attributes.get(c.DECIDED_BY, ""))))
        return out


def build_runs(spans: Iterable[SpanRec]) -> dict[str, RunView]:
    by_trace: dict[str, list[SpanRec]] = defaultdict(list)
    for s in spans:
        by_trace[s.trace_id].append(s)
    runs: dict[str, RunView] = {}
    for tid, group in by_trace.items():
        ids = {s.span_id for s in group}
        roots = [
            s for s in group
            if s.attributes.get(c.GEN_AI_OPERATION_NAME) == c.OP_INVOKE_AGENT
            and (s.parent_span_id is None or s.parent_span_id not in ids)
        ]
        root = min(roots, key=lambda s: s.start_ns) if roots else None
        states = sorted((s for s in group if c.STEP in s.attributes), key=lambda s: (s.attributes[c.STEP], s.start_ns))
        tools = [s for s in group if s.attributes.get(c.GEN_AI_OPERATION_NAME) == c.OP_EXECUTE_TOOL]
        if root is None and not states:
            continue  # not an agent trace
        rv = RunView(tid, root, states, tools)
        rv.issues = check_integrity(rv, ids)
        runs[tid] = rv
    return runs


def check_integrity(rv: RunView, ids: set[str]) -> list[str]:
    issues: list[str] = []
    if rv.root is None:
        issues.append(ISSUE_NO_ROOT)
    steps = [s.attributes.get(c.STEP) for s in rv.states]
    if len(set(steps)) != len(steps):
        issues.append(ISSUE_DUP_STEP)
    if sorted(set(steps)) != list(range(len(set(steps)))):
        issues.append(ISSUE_STEP_GAP)
    prev_to = c.START_STATE
    for s in rv.states:
        if s.attributes.get(c.FROM) != prev_to:
            issues.append(ISSUE_BROKEN_CHAIN)
            break
        prev_to = s.attributes.get(c.TO)
    if rv.root is not None:
        for s in rv.states + rv.tools:
            if s.parent_span_id is None or s.parent_span_id not in ids:
                issues.append(ISSUE_ORPHAN)
                break
        summary = rv.root.attributes.get(c.PATH)
        if summary is None:
            issues.append(ISSUE_NO_SUMMARY)
        elif summary != ">".join(rv.path):
            issues.append(ISSUE_PATH_MISMATCH)
    return sorted(set(issues))


@dataclass
class ForkFinding:
    trace_id: str
    run_key: str
    ref_trace_id: str
    fork: ForkResult


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def find_forks(runs: dict[str, RunView]) -> list[ForkFinding]:
    """Compare each complete run with the previous complete run of the same input."""
    by_key: dict[str, list[RunView]] = defaultdict(list)
    for rv in runs.values():
        if rv.complete and rv.run_key:
            by_key[rv.run_key].append(rv)
    findings = []
    for key, group in by_key.items():
        group.sort(key=lambda r: r.start_ns)
        for prev, cur in zip(group, group[1:]):
            if prev.path == cur.path:
                continue
            fork = find_fork(prev.step_records(), cur.step_records())
            if fork is not None:
                findings.append(ForkFinding(cur.trace_id, key, prev.trace_id, fork))
    return findings


def analyze(spans: Iterable[SpanRec], top: int = 5) -> dict:
    runs = build_runs(spans)
    complete = [r for r in runs.values() if r.complete]
    issue_counts = Counter(i for r in runs.values() for i in r.issues)
    outcomes = Counter(r.outcome for r in complete)
    path_counts = Counter(">".join(r.path) for r in complete)
    by_key: dict[str, Counter] = defaultdict(Counter)
    for r in complete:
        if r.run_key:
            by_key[r.run_key][">".join(r.path)] += 1
    multi = {k: v for k, v in by_key.items() if sum(v.values()) > 1}
    unstable = {k: v for k, v in multi.items() if len(v) > 1}
    forks = find_forks(runs)
    causes = Counter(f.fork.cause_label for f in forks)
    edges = Counter(f"{f.fork.from_state}: {f.fork.to_a} vs {f.fork.to_b}" for f in forks)
    worst = sorted(unstable.items(), key=lambda kv: (-_entropy(kv[1]), kv[0]))[:top]
    return {
        "traces": len(runs),
        "complete_traces": len(complete),
        "incomplete_traces": len(runs) - len(complete),
        "integrity_issues": dict(sorted(issue_counts.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "distinct_paths": len(path_counts),
        "top_paths": [{"path": p, "runs": n} for p, n in path_counts.most_common(top)],
        "inputs_with_repeats": len(multi),
        "inputs_with_more_than_one_path": len(unstable),
        "mean_path_entropy_bits": round(sum(_entropy(v) for v in multi.values()) / len(multi), 3) if multi else 0.0,
        "forks": len(forks),
        "fork_causes": dict(causes.most_common()),
        "top_fork_points": [{"split": e, "runs": n} for e, n in edges.most_common(top)],
        "least_stable_inputs": [
            {"run_key": k, "paths": len(v), "runs": sum(v.values()), "entropy_bits": round(_entropy(v), 3)}
            for k, v in worst
        ],
    }


def format_report(rep: dict) -> str:
    lines = [
        f"traces              {rep['traces']}  (complete {rep['complete_traces']}, incomplete {rep['incomplete_traces']})",
    ]
    if rep["integrity_issues"]:
        lines.append("integrity issues    " + ", ".join(f"{k}={v}" for k, v in rep["integrity_issues"].items()))
    lines += [
        "outcomes            " + ", ".join(f"{k}={v}" for k, v in rep["outcomes"].items()),
        f"distinct paths      {rep['distinct_paths']}",
        f"inputs repeated     {rep['inputs_with_repeats']}  (more than one path: {rep['inputs_with_more_than_one_path']})",
        f"mean path entropy   {rep['mean_path_entropy_bits']} bits",
        f"forks found         {rep['forks']}",
    ]
    for cause, n in rep["fork_causes"].items():
        lines.append(f"  first change      {cause:<32} {n}")
    if rep["top_fork_points"]:
        lines.append("top split points")
        for t in rep["top_fork_points"]:
            lines.append(f"  {t['split']:<44} {t['runs']}")
    return "\n".join(lines)
