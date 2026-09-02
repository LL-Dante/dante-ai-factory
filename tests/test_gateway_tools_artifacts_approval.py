from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from dante.approvals import ApprovalManager
from dante.artifacts import ArtifactStore
from dante.contracts import CostClass, LifecycleState, ModelRef, PrivacyClass, RouteDecision, Task, ToolManifest
from dante.inference import DeterministicFreeAdapter, InferenceGateway
from dante.ledger import TaskLedger
from dante.registry import ModelRegistry
from dante.tool_broker import ToolBroker, ToolDenied, workspace_method
from tools.ai_cloud_workspace import Tools


class GatewayToolArtifactApprovalTests(unittest.TestCase):
    def test_gateway_fallback_stays_zero_cost(self) -> None:
        models = [ModelRef(model_id=name, provider_id=provider, logical_alias=name, version="1", capabilities=frozenset(), cost_class=CostClass.ZERO, privacy_eligibility=frozenset({PrivacyClass.PUBLIC}), lifecycle=LifecycleState.APPROVED, runtime="test") for name, provider in (("primary", "p1"), ("fallback", "p2"))]
        registry = ModelRegistry(models)
        decision = RouteDecision(task_id="tsk", trace_id="trc", selected_model=models[0], candidates=("primary", "fallback"), reasons=("test",), policy_version="test")
        gateway = InferenceGateway(registry, [DeterministicFreeAdapter("p1", unavailable=True), DeterministicFreeAdapter("p2")])
        result = gateway.complete(decision, [{"role": "user", "content": "test"}])
        self.assertTrue(result.fallback)
        self.assertEqual(result.cost, 0)

    def test_workspace_wrapper_artifact_and_one_time_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["AI_CLOUD_WORKSPACE"] = str(Path(directory) / "workspace")
            os.environ["AI_CLOUD_AGENT_AUDIT_LOG"] = str(Path(directory) / "tool-audit.jsonl")
            ledger = TaskLedger(Path(directory) / "dante.db")
            task = ledger.create_task(Task(goal="tool", workspace=os.environ["AI_CLOUD_WORKSPACE"], privacy_class=PrivacyClass.PUBLIC))
            broker = ToolBroker(ledger=ledger)
            tools = Tools()
            manifest = ToolManifest(tool_id="workspace.write_text", permissions=frozenset({"write_workspace"}), risk="R1", filesystem_scope=os.environ["AI_CLOUD_WORKSPACE"])
            broker.register(manifest, workspace_method(tools, "write_text_file"))
            result = broker.execute(task, manifest.tool_id, {"relative_path": "e2e/result.txt", "content": "meaningful output"})
            self.assertTrue(Path(result["path"]).is_file())
            artifact = ArtifactStore(Path(directory) / "artifacts", ledger).create(task.task_id, "result.txt", b"meaningful output", media_type="text/plain", producer="workspace.write_text", provenance={"tool_id": manifest.tool_id})
            self.assertEqual(ledger.get_artifact(artifact.artifact_id).checksum_sha256, artifact.checksum_sha256)
            manager = ApprovalManager(ledger)
            approval, token = manager.request(task.task_id, "publish", "test-target", "plan-hash")
            self.assertEqual(manager.decide(approval.approval_id, token, "plan-hash", "human", approve=True).status, "approved")
            with self.assertRaises(PermissionError):
                manager.decide(approval.approval_id, token, "plan-hash", "human", approve=True)

    def test_tool_requires_manifest_approval(self) -> None:
        broker = ToolBroker()
        manifest = ToolManifest(tool_id="danger", permissions=frozenset(), risk="R3", filesystem_scope="none", approval_policy="always")
        broker.register(manifest, lambda: {"ok": True})
        with self.assertRaises(ToolDenied):
            broker.execute(Task(goal="x", workspace="x"), "danger", {})


if __name__ == "__main__":
    unittest.main()
