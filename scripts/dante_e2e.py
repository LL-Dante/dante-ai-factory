from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dante.agent_host import AgentHostFoundation
from dante.artifacts import ArtifactStore
from dante.contracts import CostClass, LifecycleState, ModelRef, PrivacyClass, TaskStatus, ToolManifest
from dante.evals import EvalRunner
from dante.inference import AICloudLiteLLMAdapter, DeterministicFreeAdapter, InferenceGateway
from dante.ledger import TaskLedger
from dante.privacy import PrivacyGate
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter, RouteDenied
from dante.telemetry import JsonlAudit
from dante.tool_broker import ToolBroker, workspace_method
from tools.ai_cloud_workspace import Tools


def model(model_id: str, provider: str, alias: str) -> ModelRef:
    return ModelRef(model_id=model_id, provider_id=provider, logical_alias=alias, version="42e7524", capabilities=frozenset({"tool_calling", "code"}), cost_class=CostClass.ZERO, privacy_eligibility=frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL}), lifecycle=LifecycleState.APPROVED, runtime="litellm", local=False)


def build(root: Path, adapters) -> tuple[AgentHostFoundation, TaskLedger, ArtifactStore]:
    audit = JsonlAudit(root / "audit.jsonl")
    ledger = TaskLedger(root / "dante.db", audit)
    models = [model("fixture/primary", "fixture-primary", "00-primary"), model("ai-cloud/agent-coding-free", "ai-cloud-free", "agent-coding-free")]
    registry = ModelRegistry(models)
    gateway = InferenceGateway(registry, adapters, audit)
    broker = ToolBroker(audit)
    workspace_tools = Tools()
    broker.register(ToolManifest(tool_id="workspace.write_text", permissions=frozenset({"write_workspace"}), risk="R1", filesystem_scope=str(root / "workspace")), workspace_method(workspace_tools, "write_text_file"))
    return AgentHostFoundation(ledger, PrivacyGate(), RuleBasedRouter(registry), gateway, broker), ledger, ArtifactStore(root / "artifacts", ledger)


class GatewayFixture(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        payload = json.dumps({"model": "agent-coding-free", "choices": [{"message": {"content": "AI_CLOUD_FREE_OK"}}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("x-litellm-response-cost", "0")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        return


def start(root: Path) -> dict:
    os.environ["AI_CLOUD_WORKSPACE"] = str(root / "workspace")
    os.environ["AI_CLOUD_AGENT_AUDIT_LOG"] = str(root / "workspace-tools.jsonl")
    os.environ["DANTE_E2E_GATEWAY_KEY"] = "ephemeral-test-only"
    server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayFixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        live_adapter = AICloudLiteLLMAdapter(f"http://127.0.0.1:{server.server_port}/v1", "env://DANTE_E2E_GATEWAY_KEY")
        host, ledger, artifacts = build(root, [DeterministicFreeAdapter("fixture-primary", unavailable=True), live_adapter])
        task = host.start("DANTE pre-beast E2E", str(root / "workspace"), PrivacyClass.PUBLIC)
        inference = host.infer(task.task_id, "public fixture", {"tool_calling", "code"})
        tool = host.execute_tool_and_checkpoint(task.task_id, "workspace.write_text", {"relative_path": "dante-e2e/output.txt", "content": inference.content}, stop_after=True)
        artifact = artifacts.create(task.task_id, "output.txt", Path(tool["path"]).read_bytes(), media_type="text/plain", producer="workspace.write_text", provenance={"route_model": inference.model_id, "fallback": inference.fallback})
        (root / "task-id.txt").write_text(task.task_id, encoding="ascii")
        state = ledger.get_task(task.task_id)
        return {"task_id": task.task_id, "status": state.status.value, "trace_id": state.trace_id, "artifact_id": artifact.artifact_id, "fallback": inference.fallback, "cost": inference.cost}
    finally:
        server.shutdown()
        server.server_close()
        os.environ.pop("DANTE_E2E_GATEWAY_KEY", None)


def resume(root: Path) -> dict:
    os.environ["AI_CLOUD_WORKSPACE"] = str(root / "workspace")
    os.environ["AI_CLOUD_AGENT_AUDIT_LOG"] = str(root / "workspace-tools.jsonl")
    host, ledger, _artifacts = build(root, [])
    task_id = (root / "task-id.txt").read_text(encoding="ascii")
    before = ledger.get_task(task_id)
    if before.status != TaskStatus.WAITING_TOOL:
        raise RuntimeError(f"Unexpected recovery status: {before.status.value}")
    route = ledger.get_route(before.selected_route or "")
    artifact = ledger.get_artifact(before.artifacts[0])
    host.resume(task_id)
    completed = host.complete(task_id)
    confidential_denied = False
    try:
        host.infer(task_id, "fattura cliente", {"tool_calling"})
    except RouteDenied:
        confidential_denied = True
    evaluation = EvalRunner().run("dante-pre-beast-e2e", {
        "completed": lambda: completed.status == TaskStatus.COMPLETED,
        "trace_coherent": lambda: route.trace_id == completed.trace_id,
        "artifact_persisted": lambda: Path(artifact.path).is_file(),
        "zero_cost": lambda: route.automatic_cost == 0,
        "confidential_cloud_denied": lambda: confidential_denied,
    })
    report = {"task_id": task_id, "status": completed.status.value, "trace_id": completed.trace_id, "artifact_id": artifact.artifact_id, "route_id": route.route_id, "eval_status": evaluation.status, "checks": evaluation.results}
    (root / "e2e-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "resume"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    result = start(args.root) if args.phase == "start" else resume(args.root)
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
