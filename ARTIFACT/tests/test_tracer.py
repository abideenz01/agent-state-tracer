import pytest
from opentelemetry.trace import StatusCode

from agent_state_tracer import conventions as c

M = {"model": "m1", "prompt_version": "v1", "temperature": 0.0}


def _by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def simple_run(tracer, key="k1", tool_result="r1", final="ANSWER", text="hello"):
    with tracer.run(key, {"text": text}) as run:
        run.transition("PLAN", reason="start", decided_by="rule")
        run.transition("CALL_TOOL", reason="needs_data", decided_by="model", evidence=M)
        with run.tool("orders_api", call_id="c1") as call:
            call.record_result(tool_result)
        run.transition(final, reason="done", decided_by="model", evidence=M)
    return run


def test_span_layout_and_semconv(tracer, otel):
    _, exp = otel
    run = simple_run(tracer)
    spans = exp.get_finished_spans()
    names = sorted(s.name for s in spans)
    assert names == ["agent_state ANSWER", "agent_state CALL_TOOL", "agent_state PLAN",
                     "execute_tool orders_api", "invoke_agent test-agent"]
    by = _by_name(exp)
    root = by["invoke_agent test-agent"]
    assert root.attributes[c.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert root.attributes[c.GEN_AI_AGENT_NAME] == "test-agent"
    assert root.attributes[c.PATH] == "PLAN>CALL_TOOL>ANSWER"
    assert root.attributes[c.OUTCOME] == "ok" and root.attributes[c.TERMINAL] == "ANSWER"
    tool = by["execute_tool orders_api"]
    assert tool.parent.span_id == by["agent_state CALL_TOOL"].context.span_id
    assert tool.attributes[c.GEN_AI_TOOL_CALL_ID] == "c1"
    for name in ("agent_state PLAN", "agent_state CALL_TOOL", "agent_state ANSWER"):
        assert by[name].parent.span_id == root.context.span_id
    ans = by["agent_state ANSWER"]
    assert ans.attributes[c.FROM] == "CALL_TOOL" and ans.attributes[c.STEP] == 2
    assert ans.attributes[c.GEN_AI_REQUEST_MODEL] == "m1"
    assert c.EVIDENCE_PREFIX + "tool:orders_api" in ans.attributes
    assert run.hint == "rare_path"


def test_no_raw_payload_is_exported(tracer, otel):
    _, exp = otel
    simple_run(tracer, tool_result={"card": "4111-1111"}, text="my secret address")
    blob = repr([dict(s.attributes) for s in exp.get_finished_spans()])
    assert "4111" not in blob and "secret" not in blob


def test_divergence_is_stamped_on_root(tracer, otel):
    _, exp = otel
    first = simple_run(tracer, tool_result="r1", final="ANSWER")
    second = simple_run(tracer, tool_result="r2", final="ESCALATE")
    assert second.hint == "divergent"
    root = [s for s in exp.get_finished_spans() if s.name.startswith("invoke_agent")][-1]
    assert root.attributes[c.FORK_CAUSE] == "tool:orders_api"
    assert root.attributes[c.FORK_STEP] == 2
    assert root.attributes[c.REF_PATH] == "PLAN>CALL_TOOL>ANSWER"
    assert root.attributes[c.REF_TRACE_ID] == first.trace_id


def test_rare_then_common(tracer):
    hints = [simple_run(tracer, key=f"k{i}").hint for i in range(5)]
    assert hints == ["rare_path", "rare_path", "rare_path", "common", "common"]


def test_sampling_forks_are_quiet_by_default(tracer):
    hot = dict(M, temperature=0.9)
    for i, final in enumerate(["ANSWER", "ANSWER", "ANSWER", "ANSWER", "ESCALATE"]):
        with tracer.run("hot", "same") as run:
            run.transition("PLAN", reason="start", decided_by="rule")
            run.transition(final, reason="done", decided_by="model", evidence=hot)
    assert run.fork.cause_label == "sampling"
    assert run.hint == "rare_path"  # first ESCALATE, not "divergent"


def test_fail_sets_error_status_and_hint(tracer, otel):
    _, exp = otel
    with tracer.run("k", "x") as run:
        run.transition("PLAN", reason="start", decided_by="rule")
        run.fail("tool_error", "TimeoutError")
    root = _by_name(exp)["invoke_agent test-agent"]
    assert root.status.status_code == StatusCode.ERROR
    assert root.status.description == "TimeoutError"
    assert run.outcome == "error" and run.hint == "error" and run.path[-1] == "FAIL"


def test_unhandled_exception_is_recorded_and_reraised(tracer, otel):
    _, exp = otel
    with pytest.raises(KeyError):
        with tracer.run("k", "x") as run:
            run.transition("PLAN", reason="start", decided_by="rule")
            raise KeyError("boom-secret")
    root = _by_name(exp)["invoke_agent test-agent"]
    assert root.attributes[c.OUTCOME] == "error"
    assert "boom-secret" not in repr(root.status.description)


def test_tool_exception_marks_tool_span(tracer, otel):
    _, exp = otel
    with tracer.run("k", "x") as run:
        run.transition("PLAN", reason="start", decided_by="rule")
        with pytest.raises(TimeoutError):
            with run.tool("orders_api"):
                raise TimeoutError
        run.fail("tool_error", "TimeoutError")
    tool = _by_name(exp)["execute_tool orders_api"]
    assert tool.status.status_code == StatusCode.ERROR
    assert c.EVIDENCE_PREFIX + "tool:orders_api" in _by_name(exp)["agent_state FAIL"].attributes


def test_abort_is_partial(tracer):
    with tracer.run("k", "x") as run:
        run.transition("PLAN", reason="start", decided_by="rule")
        run.observe("deadline_hit", True)
        run.abort("deadline")
    assert run.outcome == "partial" and run.path == ("PLAN", "STOPPED") and run.hint == "error"


def test_empty_run_is_partial(tracer):
    with tracer.run("k", "x") as run:
        pass
    assert run.outcome == "partial"


@pytest.mark.parametrize("kwargs", [
    dict(to_state="lower", reason="r", decided_by="rule"),
    dict(to_state="OK", reason="has space", decided_by="rule"),
    dict(to_state="OK", reason="r", decided_by="wizard"),
])
def test_transition_validation(tracer, kwargs):
    with tracer.run("k", "x") as run:
        with pytest.raises(ValueError):
            run.transition(kwargs.pop("to_state"), **kwargs)


def test_transition_after_finish_rejected(tracer):
    with tracer.run("k", "x") as run:
        run.transition("PLAN", reason="start", decided_by="rule")
    with pytest.raises(RuntimeError):
        run.transition("ANSWER", reason="late", decided_by="rule")


def test_digest_store_is_bounded():
    from agent_state_tracer.tracer import DigestStore, RunDigest
    st = DigestStore(max_keys=2, max_paths=2)
    for k in "abc":
        st.swap(k, RunDigest("t", ()))
        st.bump_path(k)
    assert st.swap("a", RunDigest("t", ())) is None  # evicted
    assert st.bump_path("a") == 0
    with pytest.raises(ValueError):
        DigestStore(max_keys=0)
