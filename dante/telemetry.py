from __future__ import annotations

import contextvars
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


_context: contextvars.ContextVar["TraceContext | None"] = contextvars.ContextVar("dante_trace", default=None)
_SENSITIVE = re.compile(r"(secret|token|password|authorization|api[_-]?key)", re.IGNORECASE)


@dataclass(frozen=True)
class TraceContext:
    request_id: str
    task_id: str
    trace_id: str
    correlation_id: str

    @classmethod
    def create(cls, task_id: str, trace_id: str) -> "TraceContext":
        return cls(f"req_{uuid4().hex}", task_id, trace_id, f"cor_{uuid4().hex}")


def use_trace(context: TraceContext) -> contextvars.Token:
    return _context.set(context)


def reset_trace(token: contextvars.Token) -> None:
    _context.reset(token)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _SENSITIVE.search(str(key)) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class JsonlAudit:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, event: str, **metadata: Any) -> None:
        context = _context.get()
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **({
                "request_id": context.request_id,
                "task_id": context.task_id,
                "trace_id": context.trace_id,
                "correlation_id": context.correlation_id,
            } if context else {}),
            **redact(metadata),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
