"""Load exported spans from JSON lines or OpenTelemetry OTLP JSON.

Supported inputs:
  * one span per line, as written by ``JsonlSpanExporter``;
  * OTLP JSON (``{"resourceSpans": [...]}``), one document per line as written
    by the Collector's file exporter, or a single pretty-printed document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

_STATUS_BY_CODE = {0: "UNSET", 1: "OK", 2: "ERROR"}


@dataclass
class SpanRec:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ns: int
    end_ns: int
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)


class LoadError(ValueError):
    pass


def _otlp_value(v: dict) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_otlp_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def _otlp_status(st: Any) -> str:
    if not st:
        return "UNSET"
    code = st.get("code", 0)
    if isinstance(code, str):
        code = code.replace("STATUS_CODE_", "")
        return code if code in ("UNSET", "OK", "ERROR") else "UNSET"
    return _STATUS_BY_CODE.get(int(code), "UNSET")


def _norm_id(raw: Any, width: int) -> str | None:
    if raw in (None, ""):
        return None
    s = str(raw).lower()
    if len(s) != width or any(ch not in "0123456789abcdef" for ch in s):
        raise LoadError(f"expected a {width}-char hex id, got {raw!r} (OTLP JSON uses hex ids, not base64)")
    return s


def _from_otlp(doc: dict) -> Iterable[SpanRec]:
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                attrs = {a["key"]: _otlp_value(a.get("value", {})) for a in sp.get("attributes", [])}
                yield SpanRec(
                    trace_id=_norm_id(sp.get("traceId"), 32),
                    span_id=_norm_id(sp.get("spanId"), 16),
                    parent_span_id=_norm_id(sp.get("parentSpanId"), 16),
                    name=sp.get("name", ""),
                    start_ns=int(sp.get("startTimeUnixNano", 0)),
                    end_ns=int(sp.get("endTimeUnixNano", 0)),
                    status=_otlp_status(sp.get("status")),
                    attributes=attrs,
                )


def _from_flat(d: dict) -> SpanRec:
    for key in ("trace_id", "span_id", "name"):
        if key not in d:
            raise LoadError(f"span record missing {key!r}")
    return SpanRec(
        trace_id=_norm_id(d["trace_id"], 32),
        span_id=_norm_id(d["span_id"], 16),
        parent_span_id=_norm_id(d.get("parent_span_id"), 16),
        name=d["name"],
        start_ns=int(d.get("start_ns") or 0),
        end_ns=int(d.get("end_ns") or 0),
        status=str(d.get("status") or "UNSET"),
        attributes=dict(d.get("attributes") or {}),
    )


def parse_records(records: Iterable[dict]) -> list[SpanRec]:
    out: list[SpanRec] = []
    for r in records:
        if "resourceSpans" in r:
            out.extend(_from_otlp(r))
        else:
            out.append(_from_flat(r))
    return out


def load_spans(path: str) -> list[SpanRec]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    stripped = text.strip()
    if not stripped:
        return []
    try:
        whole = json.loads(stripped)
        if isinstance(whole, dict):
            return parse_records([whole])
        if isinstance(whole, list):
            return parse_records(whole)
    except json.JSONDecodeError:
        pass
    records = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LoadError(f"{path}:{n}: invalid JSON ({exc.msg})") from None
    return parse_records(records)
