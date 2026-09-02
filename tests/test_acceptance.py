from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dante.acceptance import AcceptanceContract, DeterministicAcceptanceVerifier, ObservedAcceptance
from dante.agent_host import AgentHostFoundation
from dante.contracts import PrivacyClass, TaskStatus
from dante.ledger import TaskLedger


class AcceptanceVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = AcceptanceContract(required_tools={"write", "exists"}, required_artifacts={"report"}, required_final_fields={"absolute_path"})
        self.verifier = DeterministicAcceptanceVerifier()

    def test_all_checks_present(self) -> None:
        result = self.verifier.verify(self.contract, ObservedAcceptance(tools={"write", "exists"}, artifacts={"report"}, final_fields={"absolute_path": "C:\\work\\report.md"}))
        self.assertTrue(result.acceptance_complete)

    def test_missing_tool_is_incomplete(self) -> None:
        result = self.verifier.verify(self.contract, ObservedAcceptance(tools={"write"}, artifacts={"report"}, final_fields={"absolute_path": "C:\\work\\report.md"}))
        self.assertFalse(result.acceptance_complete)
        self.assertEqual(result.missing_checks, ("exists",))

    def test_missing_final_field_is_incomplete(self) -> None:
        result = self.verifier.verify(self.contract, ObservedAcceptance(tools={"write", "exists"}, artifacts={"report"}))
        self.assertFalse(result.acceptance_complete)
        self.assertEqual(result.missing_final_fields, ("absolute_path",))

    def test_empty_field_is_not_false_positive(self) -> None:
        result = self.verifier.verify(self.contract, ObservedAcceptance(tools={"write", "exists"}, artifacts={"report"}, final_fields={"absolute_path": "  "}))
        self.assertFalse(result.acceptance_complete)


class AcceptanceIntegrationTests(unittest.TestCase):
    def test_missing_only_continuation_then_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = TaskLedger(Path(directory) / "tasks.db")
            host = AgentHostFoundation(ledger, None, None, None, None)  # type: ignore[arg-type]
            task = host.start("create report", directory, PrivacyClass.PUBLIC)
            prompts: list[str] = []

            def continuation(prompt: str) -> ObservedAcceptance:
                prompts.append(prompt)
                return ObservedAcceptance(tools={"exists"}, final_fields={"absolute_path": str(Path(directory) / "report.md")})

            contract = AcceptanceContract(required_tools={"write", "exists"}, required_artifacts={"report"}, required_final_fields={"absolute_path"})
            finished, result = host.complete_with_one_continuation(task.task_id, contract, ObservedAcceptance(tools={"write"}, artifacts={"report"}), continuation)
            self.assertEqual(finished.status, TaskStatus.COMPLETED)
            self.assertTrue(result.acceptance_complete)
            self.assertEqual(len(prompts), 1)
            self.assertIn("tool/action: exists", prompts[0])
            self.assertIn("final field: absolute_path", prompts[0])
            self.assertNotIn("tool/action: write", prompts[0])

    def test_still_incomplete_never_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = TaskLedger(Path(directory) / "tasks.db")
            host = AgentHostFoundation(ledger, None, None, None, None)  # type: ignore[arg-type]
            task = host.start("create report", directory, PrivacyClass.PUBLIC)
            finished, result = host.complete_with_one_continuation(task.task_id, AcceptanceContract(required_tools={"exists"}), ObservedAcceptance(), lambda _: ObservedAcceptance())
            self.assertEqual(finished.status, TaskStatus.FAILED_RETRYABLE)
            self.assertFalse(result.acceptance_complete)


if __name__ == "__main__":
    unittest.main()
