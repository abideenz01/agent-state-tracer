"""Synthetic evaluation: fork attribution, sampling retention, completeness checks.

All numbers come from the synthetic simulator in ``simulate.py``. They show
how the mechanism behaves, not how your agent will behave. Re-run with your
own exported traces through ``analyze`` to get real numbers.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from . import conventions as c
from .analyze import build_runs, find_forks
from .fork import StepRecord, find_fork
from .loader import parse_records
from .sampler import TailPolicy, apply_policy
from .simulate import INJECTED_CAUSE, SCENARIO_CAUSE, simulate


def _truth_cause(prev, cur) -> str | None:
    if prev.injected and cur.injected:
        return None  # ambiguous, excluded
    if cur.injected:
        return INJECTED_CAUSE[cur.injected]
    if prev.injected:
        return INJECTED_CAUSE[prev.injected]
    straddles = prev.round < cur.drift_round <= cur.round
    if cur.scenario == "sampling" or (cur.scenario == "hidden_hot" and not straddles):
        return "sampling"
    if straddles and cur.scenario in SCENARIO_CAUSE:
        return SCENARIO_CAUSE[cur.scenario]
    return "unexpected"


def _naive_label(fork) -> str:
    if fork.fork_step_fields:
        return ",".join(fork.fork_step_fields)
    return fork.cause_label if fork.cause_step is None else "none_at_fork_step"


def _pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 1) if b else 0.0


def evaluate(keys: int = 400, replays: int = 5, seed: int = 7, tail_rate: float = 0.05) -> dict:
    sim = simulate(keys=keys, replays=replays, seed=seed)
    spans = parse_records(sim.records)
    runs = build_runs(spans)
    truth = {t.trace_id: t for t in sim.truths}

    # ---- 1. completeness checks
    injected_loss = sim.dropped_state | sim.dropped_root
    flagged = {tid for tid, rv in runs.items() if not rv.complete}
    tp = len(flagged & injected_loss)
    integrity = {
        "injected_incomplete": len(injected_loss),
        "flagged_incomplete": len(flagged),
        "caught": tp,
        "false_alarms": len(flagged - injected_loss),
    }

    # ---- 2. fork attribution (offline, complete traces, previous complete run)
    forks = find_forks(runs)
    per_class = defaultdict(lambda: [0, 0, 0])  # total, earliest-correct, fork-step-correct
    unexpected = excluded = 0
    for f in forks:
        cur, prev = truth[f.trace_id], truth[f.ref_trace_id]
        t = _truth_cause(prev, cur)
        if t is None:
            excluded += 1
            continue
        if t == "unexpected":
            unexpected += 1
            continue
        row = per_class[t]
        row[0] += 1
        row[1] += int(f.fork.cause_label == t)
        row[2] += int(_naive_label(f.fork) == t)
    total = sum(r[0] for r in per_class.values())
    attribution = {
        "forks_scored": total,
        "earliest_change_correct": sum(r[1] for r in per_class.values()),
        "fork_step_only_correct": sum(r[2] for r in per_class.values()),
        "excluded_ambiguous": excluded,
        "unexpected_forks": unexpected,
        "by_cause": {k: {"forks": v[0], "earliest_change_correct": v[1], "fork_step_only_correct": v[2]}
                     for k, v in sorted(per_class.items())},
    }

    # ---- 3. in-process stamp (what a kept divergent trace carries on its own)
    by_key = defaultdict(list)
    for t in sim.truths:
        by_key[t.run_key].append(t)
    divergent = []  # (prev, cur)
    for group in by_key.values():
        group.sort(key=lambda t: t.round)
        for prev, cur in zip(group, group[1:]):
            if prev.path != cur.path:
                divergent.append((prev, cur))
    stamped_ok = stamped_total = 0
    for prev, cur in divergent:
        t = _truth_cause(prev, cur)
        if t in (None, "unexpected"):
            continue
        stamped_total += 1
        stamped_ok += int(cur.stamped_cause == t)

    # ---- 4. sampling at a matched budget
    path_freq = Counter(t.path for t in sim.truths)
    rare_paths = {p for p, n in path_freq.items() if n < 0.01 * len(sim.truths)}
    tail_keep = apply_policy(spans, TailPolicy(rate=tail_rate))
    # Pick the head-sampling rate that keeps exactly as many traces as tail sampling did.
    # Integer comparison avoids float rounding at the boundary.
    low64 = sorted(int(tid, 16) & (2**64 - 1) for tid in truth)
    cut = low64[len(tail_keep) - 1] + 1 if tail_keep else 0
    head_rate = cut / 2**64
    head_keep = {tid for tid in truth if (int(tid, 16) & (2**64 - 1)) < cut}
    span_bytes = defaultdict(int)
    span_count = Counter()
    for rec in sim.records:
        span_bytes[rec["trace_id"]] += len(json.dumps(rec, sort_keys=True))
        span_count[rec["trace_id"]] += 1

    def has_stamp(tid: str) -> bool:
        rv = runs.get(tid)
        return bool(rv and rv.root is not None and c.FORK_CAUSE in rv.root.attributes)

    def retention(keep: set[str], self_contained: bool) -> dict:
        failed = [t for t in sim.truths if t.injected]
        rare = [t for t in sim.truths if t.path in rare_paths]
        rare_kept_paths = {t.path for t in rare if t.trace_id in keep}
        explained = Counter()
        totals = Counter()
        for prev, cur in divergent:
            kind = "sampling" if _truth_cause(prev, cur) == "sampling" else "change"
            totals[kind] += 1
            both = cur.trace_id in keep and prev.trace_id in keep
            if both or (self_contained and cur.trace_id in keep and has_stamp(cur.trace_id)):
                explained[kind] += 1
        return {
            "kept_traces": len(keep),
            "kept_spans": sum(span_count[t] for t in keep),
            "kept_json_bytes": sum(span_bytes[t] for t in keep),
            "failed_or_stopped_kept": f"{sum(t.trace_id in keep for t in failed)}/{len(failed)}",
            "rare_path_runs_kept": f"{sum(t.trace_id in keep for t in rare)}/{len(rare)}",
            "rare_paths_with_an_example": f"{len(rare_kept_paths)}/{len(rare_paths)}",
            "change_caused_splits_explainable": f"{explained['change']}/{totals['change']}",
            "sampling_splits_explainable": f"{explained['sampling']}/{totals['sampling']}",
        }

    # ---- 5. cost of the extra attributes
    total_bytes = sum(span_bytes.values())
    stripped = 0
    for rec in sim.records:
        slim = dict(rec)
        slim["attributes"] = {k: v for k, v in rec["attributes"].items() if not k.startswith(c.PREFIX)}
        stripped += len(json.dumps(slim, sort_keys=True))

    tail_bytes = sum(span_bytes[t] for t in tail_keep)
    return {
        "label": "synthetic",
        "config": {"inputs": keys, "replays": replays, "seed": seed, "tail_baseline_rate": tail_rate},
        "runs": len(sim.truths),
        "spans_exported": len(sim.records),
        "integrity": integrity,
        "fork_attribution": attribution,
        "in_process_stamp": {"divergent_runs": stamped_total, "cause_correct": stamped_ok},
        "sampling": {
            "matched_head_rate": round(head_rate, 4),
            "head": retention(head_keep, self_contained=False),
            "tail_with_hints": retention(tail_keep, self_contained=True),
        },
        "attribute_overhead": {
            "json_bytes_with_state_attributes": total_bytes,
            "json_bytes_without": stripped,
            "overhead_pct": round(100.0 * (total_bytes - stripped) / stripped, 1),
            "tail_kept_bytes_vs_keep_all_plain_pct": round(100.0 * tail_bytes / stripped, 1),
        },
    }


def to_markdown(r: dict) -> str:
    a, s, i = r["fork_attribution"], r["sampling"], r["integrity"]
    h, t = s["head"], s["tail_with_hints"]
    lines = [
        f"# Synthetic evaluation ({r['config']['inputs']} inputs x {r['config']['replays']} replays = {r['runs']} runs, seed {r['config']['seed']})",
        "",
        "All numbers are synthetic and come from the bundled simulator.",
        "",
        "## Where runs split and why",
        "",
        f"Forks scored against known causes: {a['forks_scored']}",
        "",
        "| Method | Correct cause |",
        "|---|---|",
        f"| Earliest evidence change (this tool) | {a['earliest_change_correct']}/{a['forks_scored']} ({_pct(a['earliest_change_correct'], a['forks_scored'])}%) |",
        f"| Diff at the fork step only | {a['fork_step_only_correct']}/{a['forks_scored']} ({_pct(a['fork_step_only_correct'], a['forks_scored'])}%) |",
        "",
        "| True cause | Forks | Earliest change | Fork step only |",
        "|---|---|---|---|",
    ]
    for k, v in a["by_cause"].items():
        lines.append(f"| {k} | {v['forks']} | {v['earliest_change_correct']} | {v['fork_step_only_correct']} |")
    st = r["in_process_stamp"]
    lines += [
        "",
        f"In-process stamp on the root span names the right cause for {st['cause_correct']}/{st['divergent_runs']} divergent runs.",
        "",
        f"## Sampling at the same budget ({h['kept_traces']} of {r['runs']} traces kept)",
        "",
        "| | Head sampling | Tail sampling with hints |",
        "|---|---|---|",
        f"| Traces kept | {h['kept_traces']} | {t['kept_traces']} |",
        f"| Spans kept | {h['kept_spans']} | {t['kept_spans']} |",
        f"| JSON bytes kept | {h['kept_json_bytes']} | {t['kept_json_bytes']} |",
        f"| Failed or stopped runs kept | {h['failed_or_stopped_kept']} | {t['failed_or_stopped_kept']} |",
        f"| Runs on rare paths kept | {h['rare_path_runs_kept']} | {t['rare_path_runs_kept']} |",
        f"| Rare paths with at least one example | {h['rare_paths_with_an_example']} | {t['rare_paths_with_an_example']} |",
        f"| Splits caused by a change you can explain | {h['change_caused_splits_explainable']} | {t['change_caused_splits_explainable']} |",
        f"| Splits caused by model sampling you can explain | {h['sampling_splits_explainable']} | {t['sampling_splits_explainable']} |",
        "",
        "## Completeness checks",
        "",
        f"Injected incomplete traces {i['injected_incomplete']}, flagged {i['flagged_incomplete']}, caught {i['caught']}, false alarms {i['false_alarms']}.",
        "",
        "## Size of the extra attributes",
        "",
        f"agent_state attributes add {r['attribute_overhead']['overhead_pct']}% to exported JSON bytes "
        f"({r['attribute_overhead']['json_bytes_without']} to {r['attribute_overhead']['json_bytes_with_state_attributes']}). "
        f"Tail sampling keeps {r['attribute_overhead']['tail_kept_bytes_vs_keep_all_plain_pct']}% of the bytes you would store "
        "by keeping every trace without them.",
        "",
    ]
    return "\n".join(lines)
