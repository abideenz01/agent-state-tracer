"""A JSON-lines span exporter for local runs and tests.

In production, export with the OTLP exporter to a Collector instead. The
analyzer reads both this format and the Collector's OTLP JSON file output.
"""

from __future__ import annotations

import json
import threading
from typing import IO, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode


def span_to_dict(span: ReadableSpan) -> dict:
    ctx = span.get_span_context()
    parent = span.parent
    status = span.status.status_code if span.status else StatusCode.UNSET
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "name": span.name,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "status": status.name,
        "attributes": dict(span.attributes or {}),
        "service": (span.resource.attributes.get("service.name") if span.resource else None),
    }


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: str | None = None, stream: IO[str] | None = None):
        if (path is None) == (stream is None):
            raise ValueError("give exactly one of path or stream")
        self._own = stream is None
        self._fh = stream if stream is not None else open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._closed:
            return SpanExportResult.FAILURE
        try:
            with self._lock:
                for s in spans:
                    self._fh.write(json.dumps(span_to_dict(s), sort_keys=True) + "\n")
                self._fh.flush()
            return SpanExportResult.SUCCESS
        except OSError:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._closed = True
        if self._own:
            self._fh.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
