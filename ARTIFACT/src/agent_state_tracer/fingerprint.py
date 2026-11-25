"""Turn decision evidence into payload-free span attributes.

Raw prompts, tool results and documents never leave the process. Each value is
reduced to a short SHA-256 fingerprint, which is enough to tell whether two
runs saw the same thing. A small allowlist of low-cardinality config fields
(model name, prompt version, temperature, seed) is recorded as-is because the
raw value is what an engineer needs to read during an incident.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

SAFE_RAW_FIELDS = frozenset({"model", "prompt_version", "temperature", "seed", "policy_version"})
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_:.\-]{0,63}$")
FP_LEN = 16


def fingerprint(value: Any) -> str:
    """Stable short hash of any JSON-serialisable value (key order ignored)."""
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:  # pragma: no cover - default=str covers most cases
        raise ValueError(f"evidence value is not serialisable: {type(value).__name__}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:FP_LEN]


def check_field(name: str) -> str:
    if not isinstance(name, str) or not _FIELD_RE.match(name):
        raise ValueError(
            f"invalid evidence field name {name!r}; use lowercase letters, digits, '_', ':', '.', '-' (max 64)"
        )
    return name


def encode_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return {field: recorded_value}. Safe fields keep scalar values, the rest are hashed."""
    out: dict[str, Any] = {}
    for name, value in evidence.items():
        check_field(name)
        if name in SAFE_RAW_FIELDS and isinstance(value, (str, int, float, bool)) and not (
            isinstance(value, str) and len(value) > 128
        ):
            out[name] = value
        else:
            out[name] = fingerprint(value)
    return out


def path_signature(states: list[str] | tuple[str, ...]) -> str:
    return fingerprint(list(states))[:12]
