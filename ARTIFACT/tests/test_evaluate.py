"""Regression guard on the synthetic evaluation numbers quoted in the README."""

import json
import os

from agent_state_tracer.evaluate import evaluate
from agent_state_tracer.simulate import simulate

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_simulation_is_deterministic():
    a, b = simulate(keys=20, replays=3, seed=3), simulate(keys=20, replays=3, seed=3)
    assert a.records == b.records


def test_eval_matches_committed_results():
    with open(os.path.join(EX, "eval_results.json")) as fh:
        committed = json.load(fh)
    assert evaluate() == committed


def test_eval_invariants():
    r = evaluate(keys=120, replays=4, seed=5)
    s = r["sampling"]
    assert s["head"]["kept_traces"] == s["tail_with_hints"]["kept_traces"]
    assert r["integrity"]["false_alarms"] == 0
    assert r["integrity"]["caught"] == r["integrity"]["injected_incomplete"]
    assert r["fork_attribution"]["unexpected_forks"] == 0
    a = r["fork_attribution"]
    assert a["earliest_change_correct"] >= a["fork_step_only_correct"]
    failed_tail = s["tail_with_hints"]["failed_or_stopped_kept"].split("/")
    assert failed_tail[0] == failed_tail[1]
