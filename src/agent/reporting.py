from __future__ import annotations

from typing import Any


def build_agent_report(turn_payload: dict[str, Any]) -> dict[str, Any]:
    """Create a consistent report structure for notebook display and grading."""

    return {
        "query": turn_payload.get("query"),
        "state_trace": turn_payload.get("state_trace", []),
        "tool_usage": [
            {
                "tool": item.get("tool", "unknown"),
                "summary": {
                    key: value
                    for key, value in item.items()
                    if key in {"decision", "reason", "avg_similarity", "num_results", "error"}
                },
            }
            for item in turn_payload.get("tool_outputs", [])
        ],
        "final_answer": turn_payload.get("final_answer", ""),
        "citations": turn_payload.get("citations", []),
    }
