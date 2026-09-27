"""Append-only event log so the dashboard can watch a crew run in real time.

CrewAI emits callbacks as agents think and tasks finish, but they only ever
reach stdout. Writing them to JSONL means the dashboard can tail the run from
another process, and it leaves an audit trail of what each agent actually did
long after the terminal is closed.
"""

from __future__ import annotations

import datetime as dt
import json
import threading

import config

EVENT_LOG = config.DATA_DIR / "events.jsonl"

_lock = threading.Lock()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def emit(event_type: str, **fields) -> None:
    """Write one event. Never raises: telemetry must not break a run."""
    record = {"ts": _now(), "type": event_type, **fields}
    try:
        with _lock, EVENT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def read_events(since: int = 0, limit: int = 500) -> list[dict]:
    """Return events after index `since`, each tagged with its index."""
    if not EVENT_LOG.exists():
        return []
    try:
        with EVENT_LOG.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return []

    out = []
    for i, line in enumerate(lines):
        if i < since:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        record["index"] = i
        out.append(record)
    return out[-limit:]


def event_count() -> int:
    if not EVENT_LOG.exists():
        return 0
    try:
        with EVENT_LOG.open("r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def start_run(label: str) -> None:
    """Mark a fresh run so the dashboard can separate it from history."""
    emit("run_started", label=label)


def _text(value, limit: int = 600) -> str:
    text = str(value or "").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def make_step_callback(default_agent: str = "agent"):
    """Callback for each agent reasoning step, including tool usage.

    CrewAI's step payload shape has changed between versions, so this
    duck-types rather than assuming a class.
    """

    def callback(step) -> None:
        try:
            tool = getattr(step, "tool", None)
            tool_input = getattr(step, "tool_input", None)
            output = getattr(step, "output", None) or getattr(step, "result", None)
            thought = getattr(step, "thought", None)
            agent = getattr(step, "agent", None)
            agent_name = getattr(agent, "role", None) or default_agent

            if tool:
                emit(
                    "tool_call",
                    agent=agent_name,
                    tool=str(tool),
                    tool_input=_text(tool_input, 300),
                    output=_text(output, 400),
                )
            else:
                emit("agent_step", agent=agent_name, thought=_text(thought or output))
        except Exception:  # noqa: BLE001 - telemetry must never break the crew
            pass

    return callback


def make_task_callback():
    """Callback fired when a task completes."""

    def callback(task_output) -> None:
        try:
            agent = getattr(task_output, "agent", None)
            agent_name = getattr(agent, "role", None) or str(agent or "unknown")
            emit(
                "task_completed",
                agent=agent_name,
                task=_text(getattr(task_output, "name", None) or getattr(task_output, "description", ""), 160),
                summary=_text(getattr(task_output, "raw", None) or task_output, 1200),
            )
        except Exception:  # noqa: BLE001
            pass

    return callback


def clear() -> None:
    try:
        EVENT_LOG.unlink(missing_ok=True)
    except OSError:
        pass
