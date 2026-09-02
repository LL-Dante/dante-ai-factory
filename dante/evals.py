from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from dante.contracts import EvalRun


class EvalRunner:
    def run(self, suite: str, checks: dict[str, Callable[[], bool]]) -> EvalRun:
        run = EvalRun(suite=suite)
        run.results = [{"name": name, "passed": bool(check())} for name, check in checks.items()]
        run.status = "passed" if all(item["passed"] for item in run.results) else "failed"
        run.completed_at = datetime.now(timezone.utc)
        return run
