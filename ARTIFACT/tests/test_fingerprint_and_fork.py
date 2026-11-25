import pytest

from agent_state_tracer.fingerprint import encode_evidence, fingerprint, path_signature
from agent_state_tracer.fork import StepRecord, find_fork

M = {"model": "m1", "prompt_version": "v1", "temperature": 0.0}


def test_fingerprint_ignores_key_order_and_is_short():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert len(fingerprint("x")) == 16
    assert fingerprint([1, 2]) != fingerprint([2, 1])


def test_encode_keeps_safe_config_raw_and_hashes_everything_else():
    enc = encode_evidence({"model": "m1", "temperature": 0.2, "tool:orders": {"card": "4111"}, "input": "secret"})
    assert enc["model"] == "m1" and enc["temperature"] == 0.2
    assert enc["tool:orders"] != {"card": "4111"} and len(enc["tool:orders"]) == 16
    assert "secret" not in str(enc)


def test_encode_hashes_long_safe_strings():
    enc = encode_evidence({"model": "x" * 500})
    assert len(enc["model"]) == 16


@pytest.mark.parametrize("bad", ["Upper", "", "has space", "x" * 65, 5])
def test_encode_rejects_bad_field_names(bad):
    with pytest.raises(ValueError):
        encode_evidence({bad: 1})


def test_path_signature_depends_on_order():
    assert path_signature(["A", "B"]) != path_signature(["B", "A"])


def S(to, by="model", **ev):
    return StepRecord(to, ev, by)


def test_same_path_has_no_fork():
    a = [S("PLAN", "rule"), S("ANSWER", **M)]
    assert find_fork(a, list(a)) is None


def test_earliest_change_beats_fork_step_symptom():
    a = [S("PLAN", "rule", input="i"), S("CALL_TOOL", **M), S("RETRIEVE", **{"tool:orders": "o1"}, **M),
         S("ANSWER", **{"tool:kb": "k1"}, **M)]
    b = [S("PLAN", "rule", input="i"), S("CALL_TOOL", **M), S("RETRIEVE", **{"tool:orders": "o2"}, **M),
         S("ESCALATE", **{"tool:kb": "k2"}, **M)]
    f = find_fork(a, b)
    assert f.fork_step == 3 and f.from_state == "RETRIEVE"
    assert f.to_a == "ANSWER" and f.to_b == "ESCALATE"
    assert f.cause == ("tool:orders",) and f.cause_step == 2
    assert f.fork_step_fields == ("tool:kb",)  # what a naive diff would blame


def test_no_recorded_change_hot_model_is_sampling():
    hot = dict(M, temperature=0.7)
    a = [S("PLAN", "rule"), S("ANSWER", **hot)]
    b = [S("PLAN", "rule"), S("ESCALATE", **hot)]
    assert find_fork(a, b).cause == ("sampling",)


def test_no_recorded_change_cold_model_is_unrecorded():
    a = [S("PLAN", "rule"), S("ANSWER", **M)]
    b = [S("PLAN", "rule"), S("ESCALATE", **M)]
    assert find_fork(a, b).cause == ("unrecorded",)


def test_guard_step_ignores_missing_model_config():
    a = [S("PLAN", "rule"), S("CALL_TOOL", **M), S("ANSWER", **{"tool:o": "ok"}, **M)]
    b = [S("PLAN", "rule"), S("CALL_TOOL", **M), S("FAIL", "guard", **{"tool:o": "err"})]
    assert find_fork(a, b).cause == ("tool:o",)


def test_multiple_fields_changed_together():
    a = [S("PLAN", "rule"), S("ANSWER", model="m1", prompt_version="v1", temperature=0.0)]
    b = [S("PLAN", "rule"), S("RETRIEVE", model="m2", prompt_version="v2", temperature=0.0)]
    assert find_fork(a, b).cause_label == "model,prompt_version"


def test_one_run_stops_early():
    a = [S("PLAN", "rule"), S("ANSWER", **M)]
    b = [S("PLAN", "rule")]
    f = find_fork(a, b)
    assert f.fork_step == 1 and f.to_b == "<end>"
