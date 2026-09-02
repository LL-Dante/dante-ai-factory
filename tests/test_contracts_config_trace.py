from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dante.config import load_config, redacted_config
from dante.contracts import PrivacyClass, Task, TaskStatus
from dante.telemetry import JsonlAudit, TraceContext, reset_trace, use_trace


class ContractsConfigTraceTests(unittest.TestCase):
    def test_contract_and_typed_config(self) -> None:
        task = Task(goal="test", workspace="workspace")
        self.assertEqual(task.status, TaskStatus.CREATED)
        self.assertEqual(task.privacy_class, PrivacyClass.INTERNAL)
        config = load_config(Path("config/dante/base.yaml"), overrides={"workspace": "test-workspace"})
        self.assertEqual(config.max_automatic_cost, 0)
        self.assertEqual(redacted_config(config)["litellm_api_key"], "env://LITELLM_MASTER_KEY")

    def test_non_loopback_and_nonzero_cost_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            load_config(Path("config/dante/base.yaml"), overrides={"bind": "0.0.0.0"})
        with self.assertRaises(ValueError):
            load_config(Path("config/dante/base.yaml"), overrides={"max_automatic_cost": 0.01})

    def test_trace_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit = JsonlAudit(path)
            context = TraceContext.create("tsk_test", "trc_test")
            token = use_trace(context)
            try:
                audit.write("test", api_key="do-not-log", result="ok")
            finally:
                reset_trace(token)
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["api_key"], "[REDACTED]")
            self.assertEqual(event["trace_id"], "trc_test")
            self.assertNotIn("do-not-log", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
