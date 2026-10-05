"""
Structured, redacted logging of every message passed between agents.

This is what makes the multi-agent collaboration traceable/demoable: the
orchestrator calls `log_message` at every agent boundary, and the resulting
`trace` list (of TraceEntry-shaped dicts) is returned alongside the final
result so it can be printed or rendered in the UI.
"""

from __future__ import annotations

import json

from common.security import redact_for_log


def log_message(trace: list[dict], from_agent: str, to_agent: str, schema_name: str, payload) -> None:
    step = len(trace) + 1
    preview = redact_for_log(payload)
    entry = {
        "step": step,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "schema": schema_name,
        "preview": preview,
    }
    trace.append(entry)
    print(f"[TRACE #{step}] {from_agent} -> {to_agent} :: {schema_name}")
    print(json.dumps(preview, indent=2, default=str)[:600])
