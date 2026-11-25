![Architecture](assets/architecture.gif)

# agent-state-tracer

Trace an agent's state machine with OpenTelemetry, keep the traces that matter, and explain why two runs of the same input took different paths.

## The problem

Agents are not deterministic. The same ticket can go `PLAN > CALL_TOOL > ANSWER` on Monday and `PLAN > RETRIEVE > ESCALATE` on Tuesday. When that happens, an engineer needs three answers fast, and ordinary tracing gives none of them.

1. **Where did the run split?** A normal trace is a tree of LLM and tool spans. It shows what happened in one run, but it has no idea what state the agent was in, so you cannot line up two runs and see the step where they went different ways.
2. **What changed?** The split might come from a tool returning new data, a model or prompt upgrade, an edited input, model randomness, or something you never recorded. Diffing the step where the paths split often blames a symptom. If a tool result changed at step 2, the retrieved documents at step 3 change too, and a naive diff blames retrieval.
3. **Is the evidence even there?** Head sampling decides to keep or drop a trace before the run starts. It cannot know the run will fail, take a rare path or split from yesterday's run. At a 10 to 15 percent rate you usually keep one side of a split and not the other, so you cannot compare them. On top of that, a dropped span quietly changes the path you see, and path statistics built on incomplete traces are wrong without anyone noticing.

## How this repo solves it

| Problem | What the repo does |
|---|---|
| No idea where runs split | Every agent state is its own span (`agent_state PLAN`, `agent_state CALL_TOOL` ...) recording the step index, the state it came from, a reason code, who decided (model, rule, tool, guard) and evidence fingerprints. The root `invoke_agent` span gets the full path and a path signature. |
| No idea what changed | The fork finder lines up two runs of the same input, finds the first step where they go different ways, then walks back to the **earliest** step where any recorded evidence differs. That field is the cause. If nothing recorded changed, it says `sampling` (temperature above zero) or `unrecorded` (temperature zero, so something you do not record changed). |
| Payloads in traces | Prompts, tool results and documents never leave the process. Each becomes a 16-character SHA-256 fingerprint, which is enough to tell whether two runs saw the same thing. Only low-cardinality config (model, prompt version, temperature, seed) is recorded as-is. |
| Head sampling loses the interesting runs | When a run ends, the tracer compares it with the previous run of the same input (a small in-memory digest) and writes a `agent_state.sample_hint` on the root span: `error`, `divergent`, `rare_path` or `common`. A divergent run also carries its own comparison (reference path, fork step, first change), so it stays explainable even if the other run was dropped. The Collector's `tail_sampling` processor keeps hinted traces with plain attribute rules. |
| Silent telemetry loss | The analyzer checks every trace before using it (missing root, step gaps, broken from/to chain, path summary mismatch, orphan spans) and reports incomplete traces instead of counting them. `analyze` exits with code 3 when any trace is incomplete, so it can gate a CI job. |

Splits caused by model sampling are expected when temperature is above zero, so by default they do not earn a `divergent` hint (`quiet_fork_causes`). Splits caused by a change do.

## Quick start

```bash
cd ARTIFACT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python -m pytest -q                                          # 47 tests
python examples/instrument_your_agent.py                     # 3 runs, the third one splits
python -m agent_state_tracer analyze examples/out_spans.jsonl
python -m agent_state_tracer analyze examples/sample_spans.jsonl   # synthetic, has 2 incomplete traces, exits 3
python -m agent_state_tracer evaluate                        # synthetic evaluation below
python -m agent_state_tracer collector-config --rate 0.05    # Collector tail_sampling snippet
```

Python 3.10 or newer. Runtime dependencies are `opentelemetry-api` and `opentelemetry-sdk` only. The analyzer, sampler replay and evaluation need nothing beyond the standard library at analysis time.

## Instrumenting an agent

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from agent_state_tracer import StateTracer

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(your_exporter))   # OTLP in production
tracer = StateTracer("refund-agent", tracer_provider=provider)

MODEL = {"model": "model-a", "prompt_version": "v3", "temperature": 0.0}

with tracer.run(run_key=f"ticket-{ticket_id}", input_value=ticket) as run:
    run.transition("PLAN", reason="start", decided_by="rule")
    run.transition("CALL_TOOL", reason="needs_order_status", decided_by="model", evidence=MODEL)
    with run.tool("orders_api", call_id=call_id) as call:
        order = orders_api(ticket["order"])
        call.record_result(order)          # fingerprint only
    run.transition("ANSWER", reason="enough_evidence", decided_by="model", evidence=MODEL)
```

* `run_key` must be the same for every run of the same logical input, for example a hash of the normalised request.
* Evidence recorded between two transitions (tool results, `run.observe(...)`, the `evidence=` argument) is attached to the next transition.
* `run.fail(reason, error_type)` moves to `FAIL` and sets span status to error. Only the exception type is recorded, never the message.
* `run.abort(reason)` moves to `STOPPED` with a labelled `partial` outcome, for example when a deadline or budget guard stops the run.
* An exception that escapes the `with` block is recorded as `FAIL` and re-raised.

### Span layout

```
invoke_agent refund-agent              gen_ai.operation.name=invoke_agent, gen_ai.agent.name
  agent_state PLAN                     agent_state.step=0, from=START, to=PLAN, reason, decided_by, ev.input
  agent_state CALL_TOOL                step=1, ev.model, ev.prompt_version, ev.temperature, gen_ai.request.model
    execute_tool orders_api            gen_ai.operation.name=execute_tool, gen_ai.tool.name, gen_ai.tool.call.id
  agent_state ANSWER                   step=2, ev.tool:orders_api=<fingerprint>
root, set at the end:  agent_state.path, path_sig, steps, terminal, outcome, sample_hint, path_seen
root, when it split:   agent_state.ref_path, ref_trace_id, fork_step, fork_from, fork_to, fork_to_ref, fork_cause, fork_cause_step
```

Standard attribute names follow the OpenTelemetry GenAI semantic conventions (`invoke_agent`, `execute_tool`, `gen_ai.agent.name`, `gen_ai.tool.call.id`). Those conventions are at Development status upstream and may be renamed; every key lives in `src/agent_state_tracer/conventions.py` so a rename is a one-file change. State changes are spans, not span events, because OpenTelemetry is deprecating the Span Event API in favour of log-based events.

## Analyzing an export

`analyze` reads either one span per line (from `JsonlSpanExporter`) or OTLP JSON, including the Collector file exporter output (`examples/otlp_sample.json`).

```
$ python -m agent_state_tracer analyze examples/sample_spans.jsonl
traces              120  (complete 118, incomplete 2)
integrity issues    broken_chain=1, no_root=1, path_mismatch=1, step_gap=1
outcomes            ok=118
distinct paths      15
inputs repeated     30  (more than one path: 10)
mean path entropy   0.299 bits
forks found         14
  first change      sampling                         8
  first change      tool:search_kb                   2
  first change      model,prompt_version             2
  first change      model                            1
  first change      unrecorded                       1
top split points
  CALL_TOOL: RETRIEVE vs ANSWER                2
  ...
```

`--json report.json` writes the same report as JSON.

## Synthetic evaluation

`python -m agent_state_tracer evaluate` runs a synthetic support agent (400 inputs, 5 replays each, 2000 runs, seed 7) through the real tracer and OpenTelemetry SDK. Each input gets one scenario: nothing changes, model sampling at temperature 0.7, tool data drift, search index refresh, model upgrade, prompt change, model plus prompt deploy, edited input, a hidden flag at temperature 0, or a hidden flag at temperature 0.7. Stable inputs get injected tool timeouts and deadline stops. After export, 1 percent of traces lose a state span and 0.5 percent lose the root span. Full results are in `examples/eval_results.md` and `examples/eval_results.json`.

**All numbers below are synthetic.** They show how the mechanism behaves on a simulator built for this purpose, not how your agent will behave.

**Naming the cause (283 splits with a known cause, synthetic)**

| Method | Correct cause |
|---|---|
| Earliest evidence change (this repo) | 270/283 (95.4%) |
| Diff at the split step only | 245/283 (86.6%) |

The split-step diff got every edited-input case wrong (0/17) and 8 of 29 tool-data cases wrong, because it blamed downstream symptoms. Both methods missed the same 13 cases: a hidden flag flipping at temperature 0.7 looks exactly like model sampling (see limits). The cause stamped in-process on the root span was right for 274/287 divergent runs.

**Sampling at the same budget (290 of 2000 traces kept, synthetic)**

| | Head sampling | Tail sampling with hints |
|---|---|---|
| Spans kept | 1671 | 1803 |
| JSON bytes kept | 934809 | 1040530 |
| Failed or stopped runs kept | 2/14 | 14/14 |
| Runs on rare paths kept | 9/44 | 29/44 |
| Rare paths with at least one example | 5/10 | 10/10 |
| Splits caused by a change you can explain | 7/149 | 137/149 |
| Splits caused by model sampling you can explain | 4/138 | 16/138 |

Tail sampling keeps longer traces, so it stores about 11 percent more bytes than head sampling for the same number of traces. Sampling-caused splits are deliberately not prioritised. All 12 change-caused splits it misses come from the hidden flag at temperature 0.7: the tracer labels them `sampling`, so they get no `divergent` hint.

**Completeness checks (synthetic):** 37 traces were damaged on purpose, 37 were flagged, 0 false alarms.

**Cost of the extra attributes (synthetic, JSON bytes as a proxy):** the `agent_state` attributes make exported spans 66.6 percent bigger. Tail sampling then keeps 26.5 percent of the bytes you would store by keeping every trace without them. OTLP protobuf is smaller than JSON, so measure your own ratio with your exporter.

### Measure it on your own agent

1. Instrument the agent and export spans for a week (or replay an evaluation set several times).
2. Run `analyze --json report.json` and look at `inputs_with_more_than_one_path`, `fork_causes` and `integrity_issues`.
3. To compare sampling policies on your data, load your export with `loader.load_spans` and call `sampler.apply_policy` with `HeadPolicy` and `TailPolicy`.

## Collector setup

`examples/collector-tail-sampling.yaml` (from `collector-config`) mirrors `TailPolicy`:

```yaml
processors:
  tail_sampling:
    decision_wait: 30s
    num_traces: 50000
    policies:
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: keep-agent-hints
        type: string_attribute
        string_attribute:
          key: agent_state.sample_hint
          values: ["error", "divergent", "rare_path"]
      - name: baseline-sample
        type: probabilistic
        probabilistic:
          sampling_percentage: 5.0
```

The processor needs every span of a trace on the same Collector instance. With more than one Collector, put a load-balancing exporter keyed on trace id in front. `decision_wait` must be longer than your longest agent run, or the decision is made before the root span arrives. This snippet was checked for structure against the processor's documented policy types; it was not run against a live Collector here.

## Limits and failure modes

* **Hidden inputs at temperature above zero.** If something you do not record changes and the model is sampling, the split is labelled `sampling`. Record anything that can steer a decision (feature flags, clock-dependent rules, config) with `run.observe`.
* **Per-process memory.** Divergence and rarity are judged against runs seen by the same process. With many replicas, runs of the same input can land on different processes and a split is missed. Route by `run_key`, or replace `DigestStore` with a shared store.
* **Previous run, not majority path.** The in-process stamp compares with the previous run of the same input. The analyzer does the same over the export. A flip-flopping input produces one fork per flip.
* **Fingerprints show that something changed, not what.** To see the actual values, keep them in your own access-controlled store keyed by fingerprint.
* **Fingerprints of low-entropy values can be guessed.** A hash of `true` or a short enum is easy to reverse. Do not put secrets in short evidence values.
* **The first runs of every path are kept.** `rare_min_count=3` keeps the first three runs of each distinct path. An agent with unbounded path variety will keep a lot; lower the count or cap path length.
* **Size.** The extra attributes are not free (66.6 percent more JSON bytes in the synthetic run). Drop evidence fields you never use.
* **Conventions may move.** GenAI semantic conventions are at Development status.

## Rollback and recovery

* The tracer only adds spans and attributes. To roll back, stop calling `StateTracer` or remove the `keep-agent-hints` policy from the Collector; nothing else depends on it.
* If the Collector drops too much or too little, change `sampling_percentage` or remove a hint from `values` and reload the Collector. No application change is needed.
* If `analyze` starts reporting incomplete traces, check exporter queue drops and `decision_wait` before trusting any path statistics.

## Adoption checklist

- [ ] Pick a `run_key` that is identical for repeats of the same logical input.
- [ ] Name states in UPPER_SNAKE_CASE and keep the set small and stable.
- [ ] Record every input that can steer a decision (tool results, flags, model config).
- [ ] Never pass raw prompts as safe config fields; only `model`, `prompt_version`, `temperature`, `seed` and `policy_version` are stored as-is.
- [ ] Set `quiet_fork_causes` to match how much model randomness you expect.
- [ ] Deploy the Collector `tail_sampling` snippet with `decision_wait` above your longest run.
- [ ] Route all spans of a trace to one Collector instance.
- [ ] Run `analyze` on a sample of exported traces in CI or on a schedule and alert on exit code 3.
- [ ] Measure the attribute size overhead with your real exporter.

## Repo layout

```
README.md
assets/architecture.gif
ARTIFACT/
  pyproject.toml
  requirements.txt
  src/agent_state_tracer/
    conventions.py      attribute keys (GenAI conventions + agent_state.*)
    fingerprint.py      payload-free evidence and path signatures
    fork.py             fork finder and first-change rule (shared by tracer and analyzer)
    tracer.py           StateTracer, AgentRun, DigestStore
    exporters.py        JSON-lines span exporter
    loader.py           JSON-lines and OTLP JSON loader
    analyze.py          completeness checks, path stats, fork report
    sampler.py          head and tail policies, Collector config
    simulate.py         synthetic agent and scenarios
    evaluate.py         synthetic evaluation
    cli.py              analyze, simulate, evaluate, collector-config
  tests/                47 tests
  examples/
    instrument_your_agent.py
    sample_spans.jsonl        synthetic export, 120 runs
    expected_report.json      analyzer output for the sample (regression test)
    otlp_sample.json          same data in OTLP JSON form
    collector-tail-sampling.yaml
    eval_results.json / .md   synthetic evaluation
```
