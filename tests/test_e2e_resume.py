from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DanteE2EResumeTests(unittest.TestCase):
    def test_real_tool_persistence_restart_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env["PYTHON_DOTENV_DISABLED"] = "1"
            start = subprocess.run([sys.executable, "scripts/dante_e2e.py", "start", directory], text=True, capture_output=True, env=env, check=False)
            self.assertEqual(start.returncode, 0, start.stderr)
            started = json.loads(start.stdout.strip().splitlines()[-1])
            self.assertEqual(started["status"], "waiting_tool")
            self.assertTrue(started["fallback"])
            self.assertEqual(started["cost"], 0)
            resumed = subprocess.run([sys.executable, "scripts/dante_e2e.py", "resume", directory], text=True, capture_output=True, env=env, check=False)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            report = json.loads(resumed.stdout.strip().splitlines()[-1])
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["eval_status"], "passed")
            self.assertEqual(started["trace_id"], report["trace_id"])
            self.assertTrue((Path(directory) / "e2e-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
