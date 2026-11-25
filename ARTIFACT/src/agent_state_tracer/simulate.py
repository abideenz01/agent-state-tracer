"""Synthetic support agent used for the demo and the evaluation.

Everything here is synthetic. The agent is a small state machine
(PLAN, CALL_TOOL, RETRIEVE, REFLECT, then ANSWER / ESCALATE / ASK_USER) whose
choices are a hash of what it has seen, so a run is repeatable unless
something changes. Each input is replayed several rounds and is assigned one
scenario that may change something between rounds:

  stable           nothing changes (some runs get an injected tool timeout or deadline stop)
  sampling         temperature 0.7, model randomness decides some branches
  tool_drift       the orders tool starts returning different data
  retrieval_drift  the search index is refreshed
  model_upgrade    the model version changes
  prompt_change    the prompt version changes
  deploy           model and prompt change together
  input_edit       the input behind the same run key is edited
  hidden           a flag the agent reads but never records flips
  hidden_hot       same hidden flip, but at temperature 0.7, so the tool cannot
                   tell it apart from sampling (a known blind spot)

The real StateTracer and OpenTelemetry SDK produce the spans. Telemetry loss
(a dropped state span or a dropped root span) is injected afterwards to test
the completeness checks.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.id_generator import IdGenerator

from .exporters import span_to_dict
from .fingerprint import fingerprint
from .tracer import DigestStore, StateTracer

SCENARIOS = (
    ("stable", 0.45),
    ("sampling", 0.09),
    ("tool_drift", 0.09),
    ("retrieval_drift", 0.08),
    ("model_upgrade", 0.07),
    ("prompt_change", 0.06),
    ("deploy", 0.04),
    ("input_edit", 0.04),
    ("hidden", 0.05),
    ("hidden_hot", 0.03),
)
SCENARIO_CAUSE = {
    "sampling": "sampling",
    "tool_drift": "tool:orders_api",
    "retrieval_drift": "tool:search_kb",
    "model_upgrade": "model",
    "prompt_change": "prompt_version",
    "deploy": "model,prompt_version",
    "input_edit": "input",
    "hidden": "unrecorded",
    "hidden_hot": "unrecorded",
}
INJECTED_CAUSE = {"tool_timeout": "tool:orders_api", "deadline": "deadline_hit"}


class SeededIds(IdGenerator):
    def __init__(self, rng: random.Random):
        self._rng = rng

    def generate_span_id(self) -> int:
        return self._rng.getrandbits(64) or 1

    def generate_trace_id(self) -> int:
        return self._rng.getrandbits(128) or 1


class FakeClock:
    def __init__(self, start_ns: int = 1_767_225_600_000_000_000):
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += int(ms * 1_000_000)


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@dataclass
class RunTruth:
    trace_id: str
    run_key: str
    round: int
    scenario: str
    drift_round: int
    injected: str | None = None
    path: tuple[str, ...] = ()
    hint: str | None = None
    stamped_cause: str | None = None


@dataclass
class SimResult:
    records: list[dict]
    truths: list[RunTruth]
    dropped_state: set[str] = field(default_factory=set)
    dropped_root: set[str] = field(default_factory=set)


def _pick_scenario(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for name, w in SCENARIOS:
        acc += w
        if x < acc:
            return name
    return SCENARIOS[0][0]


def _choose(options, weights, key, temperature, rng):
    x = _u(*key)
    if temperature > 0 and rng.random() < 0.5 * temperature:
        x = rng.random()
    acc = 0.0
    for opt, w in zip(options, weights):
        acc += w
        if x < acc:
            return opt
    return options[-1]


def _run_agent(st: StateTracer, clock: FakeClock, run_key: str, cfg: dict, rng: random.Random, inject: str | None):
    text = cfg["input"]
    ev = {"model": cfg["model"], "prompt_version": cfg["prompt"], "temperature": cfg["temperature"]}
    base = (text, cfg["model"], cfg["prompt"], cfg["hidden"])
    happened = None
    with st.run(run_key, {"ticket": text}) as run:
        clock.advance_ms(rng.uniform(5, 20))
        run.transition("PLAN", reason="start", decided_by="rule")
        clock.advance_ms(rng.uniform(300, 900))
        nxt = _choose(["CALL_TOOL", "RETRIEVE", "ANSWER"], [0.5, 0.4, 0.1], ("plan",) + base, cfg["temperature"], rng)
        orders, retrieves = None, 0
        while True:
            if nxt == "CALL_TOOL":
                run.transition("CALL_TOOL", reason="needs_account_data", decided_by="model", evidence=ev)
                try:
                    with run.tool("orders_api") as call:
                        clock.advance_ms(rng.uniform(80, 400))
                        if inject == "tool_timeout":
                            raise TimeoutError
                        orders = f"orders:{fingerprint(text)}:{cfg['tool_ver']}"
                        call.record_result(orders)
                except TimeoutError:
                    happened = "tool_timeout"
                    run.fail("tool_error", "TimeoutError")
                    break
                clock.advance_ms(rng.uniform(300, 900))
                nxt = _choose(["RETRIEVE", "ANSWER", "ESCALATE"], [0.55, 0.38, 0.07], ("tool", orders) + base,
                              cfg["temperature"], rng)
            elif nxt == "RETRIEVE":
                run.transition("RETRIEVE", reason="needs_context", decided_by="model", evidence=ev)
                with run.tool("search_kb") as call:
                    clock.advance_ms(rng.uniform(40, 200))
                    ids = sorted({int(_u("doc", text, orders, cfg["index_ver"], retrieves, j) * 5000) for j in range(4)})
                    call.record_result(ids)
                retrieves += 1
                clock.advance_ms(rng.uniform(300, 900))
                nxt = _choose(["REFLECT", "ANSWER", "ASK_USER"], [0.33, 0.58, 0.09], ("ret", ids) + base,
                              cfg["temperature"], rng)
            elif nxt == "REFLECT":
                run.transition("REFLECT", reason="check_coverage", decided_by="model", evidence=ev)
                clock.advance_ms(rng.uniform(300, 900))
                if retrieves < 2:
                    opts, w = ["RETRIEVE", "ANSWER", "ESCALATE"], [0.45, 0.45, 0.10]
                else:
                    opts, w = ["ANSWER", "ESCALATE"], [0.85, 0.15]
                nxt = _choose(opts, w, ("refl", retrieves) + base, cfg["temperature"], rng)
            else:
                if inject == "deadline":
                    happened = "deadline"
                    run.observe("deadline_hit", True)
                    run.abort("deadline")
                    break
                reason = {"ANSWER": "enough_evidence", "ESCALATE": "needs_human", "ASK_USER": "missing_detail"}[nxt]
                run.transition(nxt, reason=reason, decided_by="model", evidence=ev)
                clock.advance_ms(rng.uniform(200, 600))
                break
    return run, happened


def simulate(
    keys: int = 400,
    replays: int = 5,
    seed: int = 7,
    error_rate: float = 0.03,
    deadline_rate: float = 0.01,
    drop_state_rate: float = 0.01,
    drop_root_rate: float = 0.005,
    rare_min_count: int = 3,
) -> SimResult:
    if keys < 1 or replays < 2:
        raise ValueError("need at least 1 key and 2 replays")
    rng = random.Random(seed)
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "support-agent"}), id_generator=SeededIds(random.Random(seed + 1)))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    clock = FakeClock()
    st = StateTracer("support-agent", tracer_provider=provider, store=DigestStore(), rare_min_count=rare_min_count, clock=clock)

    plans = []
    for i in range(keys):
        sc = _pick_scenario(rng)
        plans.append({
            "key": f"ticket-{i:04d}",
            "scenario": sc,
            "drift": rng.randint(1, replays - 1),
            "text": f"synthetic ticket {i} about order {rng.randint(10000, 99999)}",
        })
    truths: list[RunTruth] = []
    for r in range(replays):
        order = list(range(keys))
        rng.shuffle(order)
        for i in order:
            p = plans[i]
            sc, after = p["scenario"], r >= p["drift"]
            cfg = {
                "input": p["text"] + (" (edited)" if sc == "input_edit" and after else ""),
                "model": "model-b" if sc in ("model_upgrade", "deploy") and after else "model-a",
                "prompt": "v8" if sc in ("prompt_change", "deploy") and after else "v7",
                "temperature": 0.7 if sc in ("sampling", "hidden_hot") else 0.0,
                "tool_ver": 2 if sc == "tool_drift" and after else 1,
                "index_ver": 2 if sc == "retrieval_drift" and after else 1,
                "hidden": int(sc in ("hidden", "hidden_hot") and after),
            }
            inject = None
            if sc == "stable":
                x = rng.random()
                if x < error_rate:
                    inject = "tool_timeout"
                elif x < error_rate + deadline_rate:
                    inject = "deadline"
            run_rng = random.Random(f"{seed}:{p['key']}:{r}")
            run, happened = _run_agent(st, clock, p["key"], cfg, run_rng, inject)
            truths.append(RunTruth(run.trace_id, p["key"], r, sc, p["drift"], happened, run.path, run.hint,
                                   run.fork.cause_label if run.fork else None))
            clock.advance_ms(rng.uniform(50, 500))
    provider.shutdown()
    records = [span_to_dict(s) for s in exporter.get_finished_spans()]

    # Inject telemetry loss after export.
    by_trace: dict[str, list[dict]] = {}
    for rec in records:
        by_trace.setdefault(rec["trace_id"], []).append(rec)
    loss_rng = random.Random(seed + 2)
    drop_ids: set[str] = set()
    dropped_state, dropped_root = set(), set()
    for tid in sorted(by_trace):
        group = by_trace[tid]
        x = loss_rng.random()
        states = [g for g in group if "agent_state.step" in g["attributes"]]
        if x < drop_root_rate:
            root = next(g for g in group if g["parent_span_id"] is None)
            drop_ids.add(root["span_id"])
            dropped_root.add(tid)
        elif x < drop_root_rate + drop_state_rate and len(states) >= 2:
            victim = loss_rng.choice(states[:-1])
            drop_ids.add(victim["span_id"])
            dropped_state.add(tid)
    records = [rec for rec in records if rec["span_id"] not in drop_ids]
    return SimResult(records, truths, dropped_state, dropped_root)


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
