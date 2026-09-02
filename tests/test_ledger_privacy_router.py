from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dante.contracts import CostClass, LifecycleState, ModelRef, PrivacyClass, Task, TaskStatus
from dante.ledger import InvalidTransition, TaskLedger
from dante.privacy import PrivacyGate
from dante.registry import ModelRegistry
from dante.contracts.runtime import LocalModelMetadata
from dante.routing import RouteDenied, RuleBasedRouter


def model(model_id: str, *, local: bool, privacy: set[PrivacyClass], cost: CostClass = CostClass.ZERO) -> ModelRef:
    return ModelRef(model_id=model_id, provider_id="test", logical_alias=model_id, version="1", capabilities=frozenset({"tool_calling"}), cost_class=cost, privacy_eligibility=frozenset(privacy), lifecycle=LifecycleState.APPROVED, runtime="test", local=local,
        local_metadata=LocalModelMetadata(exact_identity='fixture-revision', runtime_reference=model_id,
            identity_verified=True, identity_limitations=('deterministic fixture identity',),
            license_id='fixture-license', license_status='verified', qualification='qualified',
            machine_profiles={'default'}, context_tokens=1024, tool_use=True, eval_refs=('fixture-eval',)) if local else None)


class LedgerPrivacyRouterTests(unittest.TestCase):
    def test_state_machine_is_persistent_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dante.db"
            ledger = TaskLedger(path)
            task = ledger.create_task(Task(goal="persist", workspace=directory))
            ledger.transition(task.task_id, TaskStatus.PLANNED)
            running = ledger.transition(task.task_id, TaskStatus.RUNNING)
            same = ledger.transition(task.task_id, TaskStatus.RUNNING)
            self.assertEqual(running.attempts, same.attempts)
            ledger.checkpoint(task.task_id, "tool-ready", {"value": 7})
            ledger.transition(task.task_id, TaskStatus.WAITING_TOOL)
            restarted = TaskLedger(path)
            recovered = restarted.get_task(task.task_id)
            self.assertEqual(recovered.status, TaskStatus.WAITING_TOOL)
            self.assertEqual(recovered.checkpoint, {"value": 7})
            self.assertEqual(len(restarted.recoverable_tasks()), 1)
            with self.assertRaises(InvalidTransition):
                restarted.transition(task.task_id, TaskStatus.COMPLETED)

    def test_privacy_gate_is_deterministic(self) -> None:
        gate = PrivacyGate()
        self.assertTrue(gate.classify("public release", PrivacyClass.PUBLIC).cloud_allowed)
        self.assertFalse(gate.classify("fattura cliente", PrivacyClass.INTERNAL).cloud_allowed)
        secret = gate.classify("API_KEY=sk-abcdefghijklmnop", PrivacyClass.PUBLIC)
        self.assertEqual(secret.classification, PrivacyClass.SECRET)
        self.assertFalse(secret.model_allowed)

    def test_router_enforces_privacy_and_zero_cost(self) -> None:
        local = model("local", local=True, privacy=set(PrivacyClass))
        cloud = model("cloud-free", local=False, privacy={PrivacyClass.PUBLIC, PrivacyClass.INTERNAL})
        paid = model("paid", local=False, privacy={PrivacyClass.PUBLIC}, cost=CostClass.PAID)
        router = RuleBasedRouter(ModelRegistry([cloud, paid, local]))
        task = Task(goal="route", workspace="workspace", privacy_class=PrivacyClass.PUBLIC)
        decision = router.decide(task, PrivacyGate().classify("public", PrivacyClass.PUBLIC), {"tool_calling"})
        self.assertEqual(decision.selected_model.model_id, "local")
        self.assertEqual(decision.automatic_cost, 0)
        confidential = PrivacyGate().classify("fattura", PrivacyClass.CONFIDENTIAL)
        self.assertEqual(router.decide(task, confidential, {"tool_calling"}).selected_model.model_id, "local")
        with self.assertRaises(RouteDenied):
            RuleBasedRouter(ModelRegistry([cloud])).decide(task, confidential, {"tool_calling"})


if __name__ == "__main__":
    unittest.main()
