import json
import os

import pytest

from agent_state_tracer import conventions as c
from agent_state_tracer.analyze import analyze, build_runs, find_forks
from agent_state_tracer.cli import main
from agent_state_tracer.exporters import JsonlSpanExporter, span_to_dict
from agent_state_tracer.loader import LoadError, load_spans, parse_records
from agent_state_tracer.sampler import HeadPolicy, TailPolicy, apply_policy, collector_config, trace_ratio_keep

HERE = os.path.dirname(__file__)
EX = os.path.join(HERE, "..", "examples")
M = {"model": "m1", "prompt_version": "v1", "temperature": 0.0}


def make_runs(tracer, exporter, n=3, change_at=2):
    for i in range(n):
        with tracer.run("k1", "same") as run:
            run.transition("PLAN", reason="start", decided_by="rule")
            run.transition("CALL_TOOL", reason="go", decided_by="model", evidence=M)
            with run.tool("orders_api") as call:
                call.record_result("v2" if i >= change_at else "v1")
            run.transition("ESCALATE" if i >= change_at else "ANSWER", reason="done", decided_by="model", evidence=M)
    return parse_records(span_to_dict(s) for s in exporter.get_finished_spans())


def test_complete_runs_and_fork(tracer, otel):
    _, exp = otel
    spans = make_runs(tracer, exp)
    runs = build_runs(spans)
    assert len(runs) == 3 and all(r.complete for r in runs.values())
    forks = find_forks(runs)
    assert len(forks) == 1 and forks[0].fork.cause_label == "tool:orders_api"
    rep = analyze(spans)
    assert rep["inputs_with_more_than_one_path"] == 1 and rep["fork_causes"] == {"tool:orders_api": 1}


def test_dropped_middle_state_span_is_flagged(tracer, otel):
    _, exp = otel
    spans = make_runs(tracer, exp, n=1)
    spans = [s for s in spans if s.name != "agent_state CALL_TOOL"]
    rv = next(iter(build_runs(spans).values()))
    assert {"step_gap", "broken_chain", "path_mismatch", "orphan_span"} <= set(rv.issues)


def test_dropped_root_is_flagged_and_excluded(tracer, otel):
    _, exp = otel
    spans = make_runs(tracer, exp, n=1)
    spans = [s for s in spans if not s.name.startswith("invoke_agent")]
    rep = analyze(spans)
    assert rep["incomplete_traces"] == 1 and rep["integrity_issues"]["no_root"] == 1
    assert rep["complete_traces"] == 0


def test_last_state_dropped_is_caught_by_path_summary(tracer, otel):
    _, exp = otel
    spans = make_runs(tracer, exp, n=1)
    spans = [s for s in spans if s.name != "agent_state ANSWER"]
    rv = next(iter(build_runs(spans).values()))
    assert "path_mismatch" in rv.issues


def test_non_agent_traces_are_ignored():
    rec = {"trace_id": "a" * 32, "span_id": "b" * 16, "name": "GET /health", "attributes": {}}
    assert build_runs(parse_records([rec])) == {}


def test_jsonl_exporter_roundtrip(tmp_path, tracer, otel):
    provider, exp = otel
    make_runs(tracer, exp, n=1)
    path = tmp_path / "s.jsonl"
    e = JsonlSpanExporter(path=str(path))
    e.export(exp.get_finished_spans())
    e.shutdown()
    assert len(load_spans(str(path))) == 5
    with pytest.raises(ValueError):
        JsonlSpanExporter()


def test_otlp_json_sample_loads():
    spans = load_spans(os.path.join(EX, "otlp_sample.json"))
    assert spans and all(len(s.trace_id) == 32 for s in spans)
    assert any(isinstance(s.attributes.get(c.STEP), int) for s in spans)
    rep = analyze(spans)
    assert rep["incomplete_traces"] == 0 and rep["forks"] >= 1


def test_loader_errors(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"trace_id": "x"}\n')
    with pytest.raises(LoadError):
        load_spans(str(bad))
    bad.write_text("{not json\n")
    with pytest.raises(LoadError):
        load_spans(str(bad))
    b64 = {"resourceSpans": [{"scopeSpans": [{"spans": [{"traceId": "AAAAAAAAAAAAAAAAAAAAAA==", "spanId": "x"}]}]}]}
    bad.write_text(json.dumps(b64))
    with pytest.raises(LoadError):
        load_spans(str(bad))
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    assert load_spans(str(empty)) == []


def test_sample_report_regression():
    spans = load_spans(os.path.join(EX, "sample_spans.jsonl"))
    with open(os.path.join(EX, "expected_report.json")) as fh:
        expected = json.load(fh)
    assert analyze(spans) == expected


def test_head_and_tail_policies(tracer, otel):
    _, exp = otel
    spans = make_runs(tracer, exp, n=6, change_at=5)
    tail = apply_policy(spans, TailPolicy(rate=0.0))
    hints = {s.trace_id: s.attributes.get(c.SAMPLE_HINT) for s in spans if c.SAMPLE_HINT in s.attributes}
    assert tail == {t for t, h in hints.items() if h in c.KEEP_HINTS}
    assert apply_policy(spans, HeadPolicy(rate=1.0)) == set(hints)
    assert apply_policy(spans, HeadPolicy(rate=0.0)) == set()
    with pytest.raises(ValueError):
        trace_ratio_keep("a" * 32, 1.5)


def test_collector_config_mentions_hint_key():
    cfg = collector_config(0.1)
    assert "tail_sampling" in cfg and c.SAMPLE_HINT in cfg and "sampling_percentage: 10.0" in cfg
    with pytest.raises(ValueError):
        collector_config(2)


def test_cli_analyze_exit_codes(tmp_path, capsys):
    assert main(["analyze", os.path.join(EX, "otlp_sample.json")]) == 0
    assert main(["analyze", os.path.join(EX, "sample_spans.jsonl"), "--json", str(tmp_path / "r.json")]) == 3
    assert json.loads((tmp_path / "r.json").read_text())["traces"] == 120
    assert main(["analyze", str(tmp_path / "missing.jsonl")]) == 2


def test_cli_collector_config(capsys):
    assert main(["collector-config", "--rate", "0.02"]) == 0
    assert "sampling_percentage: 2.0" in capsys.readouterr().out
