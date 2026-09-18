"""P8 execution-policy tests use synthetic adapters and qualification states only."""
from pathlib import Path
import tempfile
import unittest

from dante.contracts import PrivacyClass, Task
from dante.contracts.continuity import BackendState, ContinuityPolicy, ExecutionPolicy
from dante.contracts.qualification import QualificationState
from dante.continuity import ContinuityManager, Disposition
from dante.inference import InferenceGateway, QuotaExhausted, RateLimitUnavailable
from dante.ledger import TaskLedger
from dante.privacy import PrivacyGate
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter
from continuity_process import FakeAdapter, models


class SyntheticP7Gate:
    """A state-only fixture: it never creates hardware qualification evidence."""
    def __init__(self, state=QualificationState.QUALIFIED):
        self.state = state

    def __call__(self, model, *, tool_use=False):
        if self.state != QualificationState.QUALIFIED:
            raise ValueError('Synthetic local qualification is not valid')


class LocalFirstRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root / 'tasks.db')
        self.task = self.ledger.create_task(Task(goal='P8 fixture', workspace=str(self.root)))
        self.gate = SyntheticP7Gate()
        self.registry = ModelRegistry(models(), machine_profile='old-pc', qualification_gate=self.gate)
        self.cloud = FakeAdapter('fake-cloud')
        self.local = FakeAdapter('fake-local')
        self.gateway = InferenceGateway(self.registry, [self.cloud, self.local])
        self.router = RuleBasedRouter(self.registry)
        self.now = 1000.0
        self.privacy = PrivacyGate().classify('fixture', PrivacyClass.INTERNAL)

    def manager(self, mode):
        manager = ContinuityManager(self.ledger, self.router, self.gateway,
                                    ContinuityPolicy(mode=mode), clock=lambda: self.now)
        manager.observe('fake-cloud', BackendState.AVAILABLE)
        manager.observe('fake-local', BackendState.AVAILABLE)
        return manager

    def infer(self, mode):
        return self.manager(mode).infer(self.task, self.privacy,
            [{'role': 'user', 'content': 'fixture'}], set())

    def test_four_execution_policies_select_expected_boundary(self):
        expected = {
            ExecutionPolicy.LOCAL_ONLY: 'fake-local',
            ExecutionPolicy.LOCAL_PREFERRED: 'fake-local',
            ExecutionPolicy.CLOUD_PREFERRED: 'fake-cloud',
            ExecutionPolicy.CLOUD_ONLY: 'fake-cloud',
        }
        for policy, provider in expected.items():
            with self.subTest(policy=policy):
                outcome = self.infer(policy)
                self.assertEqual(outcome.disposition, Disposition.CONTINUE_NOW)
                self.assertEqual(outcome.response.provider_id, provider)

    def test_legacy_preference_names_remain_compatible(self):
        self.assertEqual(self.infer('LOCAL_FIRST').response.provider_id, 'fake-local')
        self.assertEqual(self.infer('CLOUD_FIRST_WITH_LOCAL_FALLBACK').response.provider_id, 'fake-cloud')

    def test_nonqualified_p7_states_are_rejected(self):
        for state in (QualificationState.UNKNOWN, QualificationState.FAILED, QualificationState.STALE):
            with self.subTest(state=state):
                self.gate.state = state
                outcome = self.infer(ExecutionPolicy.LOCAL_ONLY)
                self.assertEqual((outcome.disposition, outcome.reason),
                                 (Disposition.TERMINAL, 'no_eligible_route'))
        self.assertEqual(self.local.calls, [])

    def test_cloud_quota_falls_back_only_when_policy_allows_local(self):
        for error in (QuotaExhausted(), RateLimitUnavailable()):
            with self.subTest(error=type(error).__name__):
                self.cloud.error = error
                outcome = self.infer(ExecutionPolicy.CLOUD_PREFERRED)
                self.assertEqual(outcome.response.provider_id, 'fake-local')
                self.assertTrue(outcome.response.fallback)

    def test_cloud_only_quota_does_not_cross_policy_boundary(self):
        self.cloud.error = QuotaExhausted()
        outcome = self.infer(ExecutionPolicy.CLOUD_ONLY)
        self.assertEqual((outcome.disposition, outcome.reason),
                         (Disposition.RETRY_LATER, 'provider_quota_cooldown'))
        self.assertEqual(self.local.calls, [])

    def test_no_backend_available_is_deterministic(self):
        self.gate.state = QualificationState.UNKNOWN
        first = self.infer(ExecutionPolicy.LOCAL_ONLY)
        second = self.infer(ExecutionPolicy.LOCAL_ONLY)
        self.assertEqual((first.disposition, first.reason), (second.disposition, second.reason))
        self.assertEqual((first.disposition, first.reason),
                         (Disposition.TERMINAL, 'no_eligible_route'))


if __name__ == '__main__':
    unittest.main()
