import os
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_state_tracer import DigestStore, StateTracer  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1_000_000_000

    def __call__(self):
        self.t += 1_000_000
        return self.t


@pytest.fixture
def otel():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


@pytest.fixture
def tracer(otel):
    provider, _ = otel
    return StateTracer("test-agent", tracer_provider=provider, store=DigestStore(), clock=Clock())
