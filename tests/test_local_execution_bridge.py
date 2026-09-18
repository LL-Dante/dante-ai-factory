"""End-to-end P8 bridge tests use synthetic adapters and no runtime process."""
from pathlib import Path
import tempfile
import unittest

from dante.agent_host import AgentHostFoundation
from dante.contracts import PrivacyClass, Task
from dante.contracts.continuity import BackendState, ContinuityPolicy, ExecutionPolicy
from dante.contracts.inference import AssistantMessage, InferenceResponse
from dante.contracts.qualification import QualificationState
from dante.continuity import ContinuityManager, ContinuitySignal, Disposition
from dante.inference import InferenceGateway, InvalidResponse, QuotaExhausted
from dante.ledger import TaskLedger
from dante.privacy import PrivacyGate
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter
from continuity_process import FakeAdapter, models


class SyntheticP7Gate:
    def __init__(self, state=QualificationState.QUALIFIED):
        self.state = state

    def __call__(self, model, *, tool_use=False):
        if self.state != QualificationState.QUALIFIED:
            raise ValueError('Synthetic P7 state is not qualified')


class MismatchedAdapter(FakeAdapter):
    def __init__(self, provider_id, wrong_model):
        super().__init__(provider_id)
        self.wrong_model = wrong_model

    def complete(self, request):
        self.calls.append(request)
        return InferenceResponse(model=self.wrong_model,
            assistant_message=AssistantMessage(content='fixture'), finish_reason='stop')


class LocalExecutionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root / 'tasks.db')
        self.task = self.ledger.create_task(Task(goal='bridge fixture', workspace=str(self.root)))
        self.gate = SyntheticP7Gate()
        self.registry = ModelRegistry(models(), machine_profile='old-pc', qualification_gate=self.gate)
        self.cloud = FakeAdapter('fake-cloud')
        self.local = FakeAdapter('fake-local')
        self.now = 1000.0

    def host(self, mode, *, adapters=None):
        gateway = InferenceGateway(self.registry, [self.cloud, self.local] if adapters is None else adapters)
        router = RuleBasedRouter(self.registry)
        manager = ContinuityManager(self.ledger, router, gateway,
            ContinuityPolicy(mode=mode, max_route_attempts=20), clock=lambda: self.now)
        for adapter in (self.cloud, self.local):
            manager.observe(adapter.provider_id, BackendState.AVAILABLE)
        return AgentHostFoundation(self.ledger, PrivacyGate(), router, gateway, None,
                                   continuity=manager), manager

    def test_local_only_executes_selected_local_adapter_through_gateway(self):
        host, _ = self.host(ExecutionPolicy.LOCAL_ONLY)
        response = host.infer(self.task.task_id, 'fixture', set())
        self.assertEqual(response.provider_id, 'fake-local')
        self.assertEqual(len(self.local.calls), 1)
        self.assertEqual(self.cloud.calls, [])
        self.assertEqual(self.local.calls[0].task_id, self.task.task_id)

    def test_local_preferred_preserves_local_selection(self):
        host, _ = self.host(ExecutionPolicy.LOCAL_PREFERRED)
        self.assertEqual(host.infer(self.task.task_id, 'fixture', set()).provider_id, 'fake-local')
        self.assertEqual(self.cloud.calls, [])

    def test_cloud_quota_executes_eligible_local_adapter_end_to_end(self):
        self.cloud.error = QuotaExhausted(retry_after=60)
        host, manager = self.host(ExecutionPolicy.CLOUD_PREFERRED)
        response = host.infer(self.task.task_id, 'fixture', set())
        self.assertEqual(response.provider_id, 'fake-local')
        self.assertTrue(response.fallback)
        self.assertEqual((len(self.cloud.calls), len(self.local.calls)), (1, 1))
        self.assertEqual(manager.observation('fake-cloud')['reason'], 'quota_exhausted')

    def test_invalid_local_response_enters_existing_runtime_failure_path(self):
        mismatch = MismatchedAdapter('fake-local', self.registry.get('cloud'))
        host, manager = self.host(ExecutionPolicy.LOCAL_ONLY, adapters=[mismatch])
        with self.assertRaises(ContinuitySignal) as raised:
            host.infer(self.task.task_id, 'fixture', set())
        self.assertEqual((raised.exception.outcome.disposition, raised.exception.outcome.reason),
                         (Disposition.RETRY_LATER, 'routes_temporarily_unavailable'))
        self.assertEqual(manager.observation('fake-local')['reason'], 'invalid_response')

    def test_no_adapter_has_deterministic_terminal_result(self):
        host, _ = self.host(ExecutionPolicy.LOCAL_ONLY, adapters=[])
        reasons = []
        for _ in range(2):
            with self.assertRaises(ContinuitySignal) as raised:
                host.infer(self.task.task_id, 'fixture', set())
            reasons.append((raised.exception.outcome.disposition, raised.exception.outcome.reason))
        self.assertEqual(reasons, [(Disposition.TERMINAL, 'no_eligible_route')] * 2)

    def test_nonqualified_local_route_never_reaches_adapter(self):
        self.gate.state = QualificationState.STALE
        host, _ = self.host(ExecutionPolicy.LOCAL_ONLY)
        with self.assertRaises(ContinuitySignal) as raised:
            host.infer(self.task.task_id, 'fixture', set())
        self.assertEqual(raised.exception.outcome.reason, 'no_eligible_route')
        self.assertEqual(self.local.calls, [])


if __name__ == '__main__':
    unittest.main()
