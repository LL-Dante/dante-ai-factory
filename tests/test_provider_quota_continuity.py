"""Persistent provider cooldown tests use controlled time and synthetic adapters."""
from pathlib import Path
import tempfile
import unittest

from dante.contracts import PrivacyClass, Task
from dante.contracts.continuity import BackendState, ContinuityPolicy, ExecutionPolicy
from dante.continuity import ContinuityManager, Disposition
from dante.inference import InferenceGateway, QuotaExhausted, RateLimitUnavailable, reset_at_seconds
from dante.ledger import TaskLedger
from dante.privacy import PrivacyGate
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter
from continuity_process import FakeAdapter, models


class ProviderQuotaContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root / 'tasks.db')
        self.task = self.ledger.create_task(Task(goal='quota fixture', workspace=str(self.root)))
        self.registry = ModelRegistry(models(), machine_profile='old-pc')
        self.cloud = FakeAdapter('fake-cloud')
        self.local = FakeAdapter('fake-local')
        self.gateway = InferenceGateway(self.registry, [self.cloud, self.local])
        self.router = RuleBasedRouter(self.registry)
        self.now = 1000.0
        self.policy = ContinuityPolicy(mode=ExecutionPolicy.CLOUD_PREFERRED, max_route_attempts=20)
        self.privacy = PrivacyGate().classify('fixture', PrivacyClass.INTERNAL)
        self.manager = self.make_manager()
        for backend in ('fake-cloud', 'fake-local'):
            self.manager.observe(backend, BackendState.AVAILABLE)

    def make_manager(self):
        return ContinuityManager(self.ledger, self.router, self.gateway,
                                 self.policy, clock=lambda: self.now)

    def infer(self):
        return self.manager.infer(self.task, self.privacy,
            [{'role': 'user', 'content': 'fixture'}], set())

    def test_quota_state_survives_restart_and_skips_provider(self):
        self.cloud.error = QuotaExhausted(retry_after=120)
        self.assertEqual(self.infer().response.provider_id, 'fake-local')
        stored = self.manager.observation('fake-cloud')
        self.assertEqual((stored['state'], stored['reason'], stored['cooldown_until']),
                         ('quota_exhausted', 'quota_exhausted', 1120.0))
        self.manager = self.make_manager()
        self.assertEqual(self.infer().response.provider_id, 'fake-local')
        self.assertEqual(len(self.cloud.calls), 1)

    def test_rate_limit_state_and_reset_time_are_persisted(self):
        self.cloud.error = RateLimitUnavailable(retry_after=10, retry_at=1150)
        self.assertEqual(self.infer().response.provider_id, 'fake-local')
        stored = self.manager.observation('fake-cloud')
        self.assertEqual((stored['state'], stored['reason'], stored['cooldown_until']),
                         ('rate_limited', 'rate_limit', 1150.0))

    def test_provider_recovers_deterministically_after_cooldown(self):
        self.cloud.error = QuotaExhausted(retry_at=1010)
        self.assertEqual(self.infer().response.provider_id, 'fake-local')
        self.cloud.error = None
        self.now = 1009
        self.assertEqual(self.infer().response.provider_id, 'fake-local')
        self.now = 1010
        self.assertEqual(self.infer().response.provider_id, 'fake-cloud')
        self.assertEqual(self.manager.observation('fake-cloud')['state'], 'available')

    def test_no_backend_reports_normalized_cooldown_reason(self):
        self.policy = ContinuityPolicy(mode=ExecutionPolicy.CLOUD_ONLY, max_route_attempts=20)
        self.manager = self.make_manager()
        self.manager.observe('fake-cloud', BackendState.AVAILABLE)
        self.cloud.error = RateLimitUnavailable(retry_after=30)
        first = self.infer()
        second = self.infer()
        self.assertEqual((first.disposition, first.reason),
                         (Disposition.RETRY_LATER, 'provider_rate_limit_cooldown'))
        self.assertEqual((second.disposition, second.reason),
                         (Disposition.RETRY_LATER, 'provider_rate_limit_cooldown'))
        self.assertEqual(len(self.cloud.calls), 1)

    def test_reset_timestamp_parser_is_bounded(self):
        self.assertEqual(reset_at_seconds('1234.5'), 1234.5)
        for value in (None, '-1', 'NaN', 'inf', 'invalid'):
            with self.subTest(value=value):
                self.assertIsNone(reset_at_seconds(value))


if __name__ == '__main__':
    unittest.main()
