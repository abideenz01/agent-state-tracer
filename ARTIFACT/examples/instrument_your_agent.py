"""Minimal example: instrument a tiny agent and write spans to a local file.

Run from the ARTIFACT folder:
    PYTHONPATH=src python examples/instrument_your_agent.py
    PYTHONPATH=src python -m agent_state_tracer analyze examples/out_spans.jsonl

To send spans to a Collector instead, install opentelemetry-exporter-otlp-proto-http
and swap the exporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
"""

import os

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent_state_tracer import StateTracer
from agent_state_tracer.exporters import JsonlSpanExporter

OUT = os.path.join(os.path.dirname(__file__), "out_spans.jsonl")
if os.path.exists(OUT):
    os.remove(OUT)

provider = TracerProvider(resource=Resource.create({"service.name": "refund-agent"}))
provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter(path=OUT)))
tracer = StateTracer("refund-agent", tracer_provider=provider)

MODEL = {"model": "model-a", "prompt_version": "v3", "temperature": 0.0}


def lookup_order(order_id: str, stock_feed: int) -> dict:
    # Stand-in for a real API. stock_feed simulates upstream data changing between runs.
    return {"order": order_id, "status": "shipped" if stock_feed == 1 else "lost"}


def run_agent(ticket: dict, stock_feed: int) -> None:
    # Same logical input -> same run_key, so runs can be compared.
    with tracer.run(run_key=f"ticket-{ticket['id']}", input_value=ticket) as run:
        run.transition("PLAN", reason="start", decided_by="rule")
        run.transition("CALL_TOOL", reason="needs_order_status", decided_by="model", evidence=MODEL)
        with run.tool("orders_api", call_id="call-1") as call:
            order = lookup_order(ticket["order"], stock_feed)
            call.record_result(order)  # only a fingerprint is exported
        if order["status"] == "shipped":
            run.transition("ANSWER", reason="enough_evidence", decided_by="model", evidence=MODEL)
        else:
            run.transition("ESCALATE", reason="needs_human", decided_by="model", evidence=MODEL)
    print(f"{run.trace_id}  path={'>'.join(run.path)}  hint={run.hint}"
          + (f"  first change={run.fork.cause_label}" if run.fork else ""))


ticket = {"id": 41, "order": "A-1009", "text": "Where is my refund?"}
run_agent(ticket, stock_feed=1)
run_agent(ticket, stock_feed=1)
run_agent(ticket, stock_feed=2)  # upstream data changed: path splits, root span says why
provider.shutdown()
print(f"spans written to {OUT}")
