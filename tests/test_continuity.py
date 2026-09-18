"""Deterministic continuity policy, persistence and process-restart tests. No network."""
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from dante.contracts import Task, TaskStatus, PrivacyClass, CostClass
from dante.contracts.continuity import ContinuityPolicy, BackendState as State
from dante.continuity import ContinuityManager, Disposition, classify
from dante.inference import (InferenceGateway, InferenceTimeout, AdapterUnavailable, RateLimitUnavailable,
    QuotaExhausted, ContextExceeded, InvalidResponse, PolicyDenied, QualificationDenied,
    ConfigurationDenied, InvalidRequest, retry_after_seconds)
from dante.ledger import TaskLedger
from dante.privacy import PrivacyGate
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter
from dante.task_queue import TaskQueue, ExecutionPlan, Action, lease_context, LeaseLost
from dante.worker import Worker
from continuity_process import models, FakeAdapter, host_factory


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root / 'tasks.db')
        self.task = self.ledger.create_task(Task(goal='continuity fixture', workspace=str(self.root)))
        self.now = 1000.
        self.registry = ModelRegistry(models(), machine_profile='old-pc')
        self.cloud, self.local = FakeAdapter('fake-cloud'), FakeAdapter('fake-local')
        self.gateway = InferenceGateway(self.registry, [self.cloud, self.local])
        self.manager = ContinuityManager(self.ledger, RuleBasedRouter(self.registry), self.gateway,
            ContinuityPolicy(mode='CLOUD_FIRST_WITH_LOCAL_FALLBACK'), clock=lambda: self.now)
        for adapter in (self.cloud, self.local):
            self.manager.observe(adapter.provider_id, State.AVAILABLE)

    def infer(self, capabilities=None, privacy=PrivacyClass.INTERNAL, context=None):
        return self.manager.infer(self.task, PrivacyGate().classify('fixture', privacy),
            [{'role':'user', 'content':'confidential prompt marker'}], capabilities or set(), context_tokens=context)

    def fallback(self, error):
        self.cloud.error = error
        outcome = self.infer()
        self.assertEqual(outcome.disposition, Disposition.CONTINUE_NOW)
        self.assertEqual(outcome.response.model.provider_id, 'fake-local')
        self.assertTrue(outcome.response.fallback)
        self.assertEqual(len(self.cloud.calls), 1)
        self.assertEqual(len(self.local.calls), 1)
        return outcome

    def change(self, name, **changes):
        self.registry.register(self.registry.get(name).model_copy(update=changes))

    def test_cloud_success_no_fallback(self):
        result = self.infer()
        self.assertEqual(result.response.provider_id, 'fake-cloud')
        self.assertFalse(result.response.fallback)
        self.assertEqual(self.local.calls, [])
        self.assertIsNone(result.response.cost)

    def test_timeout_local_continuation(self):
        self.fallback(InferenceTimeout())
        self.assertEqual(self.manager.observation('fake-cloud')['state'], State.DEGRADED)

    def test_unavailable_local_continuation(self):
        self.fallback(AdapterUnavailable())

    def test_rate_limit_local_continuation(self):
        self.fallback(RateLimitUnavailable())
        self.assertEqual(self.manager.observation('fake-cloud')['state'], State.RATE_LIMITED)

    def test_quota_local_continuation(self):
        self.fallback(QuotaExhausted())
        self.assertEqual(self.manager.observation('fake-cloud')['state'], State.QUOTA_EXHAUSTED)

    def test_same_task_identity(self):
        self.fallback(QuotaExhausted())
        self.assertEqual(self.cloud.calls[0].task_id, self.task.task_id)
        self.assertEqual(self.local.calls[0].task_id, self.task.task_id)
        self.assertEqual(self.local.calls[0].trace_id, self.task.trace_id)

    def test_capability_mismatch_not_selected(self):
        self.cloud.error = AdapterUnavailable()
        self.change('local-fixture', capabilities=frozenset())
        self.assertEqual(self.infer({'tool_calling'}).disposition, Disposition.RETRY_LATER)
        self.assertEqual(self.local.calls, [])

    def test_privacy_local_only(self):
        self.assertEqual(self.infer(privacy=PrivacyClass.CONFIDENTIAL).response.provider_id, 'fake-local')
        self.assertEqual(self.cloud.calls, [])

    def test_unqualified_local_denied(self):
        self.change('local-fixture', local_metadata=None)
        self.manager.policy = ContinuityPolicy()
        self.assertEqual(self.infer().disposition, Disposition.TERMINAL)
        self.assertEqual(self.local.calls, [])

    def test_unhealthy_runtime_denied(self):
        self.manager.policy = ContinuityPolicy()
        self.manager.observe('fake-local', State.UNAVAILABLE)
        self.assertEqual(self.infer().disposition, Disposition.RETRY_LATER)
        self.assertEqual(self.local.calls, [])

    def test_unknown_cloud_cost_denied(self):
        self.change('cloud', cost_verification='COST_UNVERIFIED')
        self.assertEqual(self.infer().response.provider_id, 'fake-local')
        self.assertEqual(self.cloud.calls, [])

    def test_paid_denied(self):
        self.change('cloud', cost_class=CostClass.PAID)
        self.assertEqual(self.infer().response.provider_id, 'fake-local')
        self.assertEqual(self.cloud.calls, [])

    def test_retry_after_respected(self):
        self.fallback(RateLimitUnavailable(retry_after=120))
        self.assertEqual(self.manager.observation('fake-cloud')['cooldown_until'], self.now+120)
        self.infer()
        self.assertEqual(len(self.cloud.calls), 1)

    def test_all_unavailable_persistent_nonterminal(self):
        self.cloud.error = AdapterUnavailable()
        self.local.error = AdapterUnavailable()
        outcome = self.infer()
        self.assertEqual(outcome.disposition, Disposition.RETRY_LATER)
        self.assertGreater(outcome.next_attempt_at, self.now)
        restored = TaskLedger(self.ledger.path)
        self.assertEqual(restored.get_task(self.task.task_id).task_id, self.task.task_id)
        with closing(restored._connect()) as db:
            row = db.execute('SELECT * FROM task_continuity').fetchone()
        self.assertEqual(row['next_attempt_at'], outcome.next_attempt_at)
        self.assertEqual(row['disposition'], 'retry_later')

    def test_permanent_mismatch_terminal(self):
        self.assertEqual(self.infer({'impossible'}).disposition, Disposition.TERMINAL)
        self.assertEqual(self.cloud.calls+self.local.calls, [])

    def test_budget_persists_across_manager_restart(self):
        self.manager.policy = ContinuityPolicy(mode='CLOUD_FIRST_WITH_LOCAL_FALLBACK', max_route_attempts=1)
        self.cloud.error = InferenceTimeout()
        result = self.infer()
        self.assertEqual(result.reason, 'route_attempt_budget')
        self.manager = ContinuityManager(TaskLedger(self.ledger.path), self.manager.router, self.gateway,
                                        self.manager.policy, clock=lambda:self.now)
        self.assertEqual(self.infer().reason, 'route_attempt_budget')
        self.assertEqual(self.local.calls, [])
        self.now += 60
        self.cloud.error = None
        self.assertEqual(self.infer().disposition, Disposition.CONTINUE_NOW)

    def test_success_restores_backend(self):
        self.fallback(InferenceTimeout())
        self.now += 6
        self.cloud.error = None
        self.assertEqual(self.infer().response.provider_id, 'fake-cloud')
        observed = self.manager.observation('fake-cloud')
        self.assertEqual((observed['state'], observed['consecutive_failures'], observed['last_success']),
                         ('available', 0, self.now))
        self.assertIsNone(observed['cooldown_until'])

    def test_audit_sanitized(self):
        self.cloud.error = AdapterUnavailable('confidential exception marker')
        self.local.error = InferenceTimeout()
        self.infer()
        events = [e for e in self.ledger.events(self.task.task_id) if e['event'].startswith('continuity.')]
        text = json.dumps(events)
        for expected in ('continuity.candidates','continuity.selected','continuity.failure','retry_later','cooldown'):
            self.assertIn(expected, text)
        self.assertNotIn('confidential', text)
        self.assertTrue(all(e['task_id'] == self.task.task_id for e in events))

    def test_context_capacity_denied(self):
        self.cloud.error = AdapterUnavailable()
        self.assertEqual(self.infer(context=8192).disposition, Disposition.TERMINAL)
        self.assertEqual(self.local.calls+self.cloud.calls, [])

    def test_machine_profile_denied(self):
        self.registry.machine_profile = 'other-machine'
        self.manager.policy = ContinuityPolicy()
        self.assertEqual(self.infer().disposition, Disposition.TERMINAL)
        self.assertEqual(self.local.calls, [])

    def test_policy_modes(self):
        for mode in ('LOCAL_ONLY','LOCAL_FIRST','SPECIFIC_ALLOWED_BACKENDS'):
            self.manager.policy = ContinuityPolicy(mode=mode, allowed_backends={'fake-local'})
            self.assertEqual(self.infer().response.provider_id, 'fake-local')
        self.assertEqual(self.cloud.calls, [])

    def test_unknown_and_stale_health_fail_closed(self):
        self.manager.policy = ContinuityPolicy()
        self.manager.observe('fake-local', State.UNKNOWN)
        self.assertEqual(self.infer().disposition, Disposition.RETRY_LATER)
        self.manager.observe('fake-local', State.AVAILABLE)
        self.now += 301
        self.assertEqual(self.infer().disposition, Disposition.RETRY_LATER)
        self.assertEqual(self.local.calls, [])

    def test_classifications(self):
        for error, reason in ((ContextExceeded(),'context_exceeded'), (InvalidResponse(),'invalid_response'),
                (PolicyDenied(),'policy_denied'), (QualificationDenied(),'qualification_denied'),
                (ConfigurationDenied(),'configuration_denied'), (InvalidRequest(),'invalid_request')):
            self.assertEqual(classify(error)[0], reason)
        self.cloud.error = InvalidRequest()
        self.assertEqual(self.infer().reason, 'invalid_request')
        self.assertEqual(self.local.calls, [])

    def test_context_error_reevaluates_other_route(self):
        self.fallback(ContextExceeded())
        self.assertEqual(self.manager.observation('fake-cloud')['state'], 'available')

    def test_invalid_response_cooldown(self):
        self.fallback(InvalidResponse())
        self.assertGreater(self.manager.observation('fake-cloud')['cooldown_until'], self.now)

    def test_retry_after_parser(self):
        self.assertEqual(retry_after_seconds('42'),42)
        self.assertIsNone(retry_after_seconds('NaN'))
        self.assertIsNone(retry_after_seconds('invalid'))

    def test_additive_v2_migration_preserves_task(self):
        with closing(self.ledger._connect()) as db, db:
            db.execute('DROP TABLE backend_observations')
            db.execute('DROP TABLE task_continuity')
            db.execute('PRAGMA user_version=2')
        restored = TaskLedger(self.ledger.path)
        self.assertEqual(restored.get_task(self.task.task_id), self.task)
        with closing(restored._connect()) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],4)

    def test_http_error_classification_and_retry_after(self):
        import io
        import urllib.error
        from dante.inference import AICloudLiteLLMAdapter
        for code, expected in (('insufficient_quota', QuotaExhausted), ('rate_limit', RateLimitUnavailable)):
            error = urllib.error.HTTPError('http://fixture',429,'',{'Retry-After':'90'},
                                          io.BytesIO(json.dumps({'error':{'code':code}}).encode()))
            mapped = AICloudLiteLLMAdapter._http_error(error)
            self.assertIsInstance(mapped,expected)
            self.assertEqual(mapped.retry_after,90)
        outage = urllib.error.HTTPError('http://fixture',503,'',{'Retry-After':'75'},io.BytesIO(b'{}'))
        self.assertEqual(AICloudLiteLLMAdapter._http_error(outage).retry_after,75)
        invalid = urllib.error.HTTPError('http://fixture',400,'',{},io.BytesIO(b'{}'))
        self.assertIsInstance(AICloudLiteLLMAdapter._http_error(invalid),InvalidRequest)

    def test_bounded_exponential_backoff(self):
        self.manager.policy = ContinuityPolicy(mode='SPECIFIC_ALLOWED_BACKENDS',
            allowed_backends={'fake-cloud'},base_backoff_s=2,max_backoff_s=8,max_route_attempts=100)
        self.cloud.error = AdapterUnavailable()
        for delay in (2,4,8,8):
            outcome = self.infer()
            self.assertEqual(outcome.next_attempt_at,self.now+delay)
            self.now += delay
        self.assertEqual(len(self.cloud.calls),4)

    def test_configuration_policy_validation(self):
        from dante.config import DanteConfig
        from pydantic import ValidationError
        config = DanteConfig(workspace=self.root,continuity={'mode':'LOCAL_FIRST'})
        self.assertEqual(config.continuity.mode,'LOCAL_FIRST')
        for values in ({'max_route_attempts':0},{'window_s':0},{'base_backoff_s':10,'max_backoff_s':1}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ContinuityPolicy(**values)

    def test_explicit_allowlist_default_deny(self):
        self.manager.policy = ContinuityPolicy(mode='SPECIFIC_ALLOWED_BACKENDS')
        self.assertEqual(self.infer().disposition,Disposition.TERMINAL)
        self.assertEqual(self.cloud.calls+self.local.calls,[])

    def test_policy_error_reevaluates_without_health_poisoning(self):
        self.fallback(PolicyDenied())
        self.assertEqual(self.manager.observation('fake-cloud')['state'],'available')

    def test_missing_adapter_fails_closed(self):
        self.gateway.adapters.clear()
        self.assertEqual(self.infer().disposition,Disposition.TERMINAL)
        self.assertEqual(self.cloud.calls+self.local.calls,[])


class ContinuityProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root/'tasks.db')
        self.queue = TaskQueue(self.ledger)
        self.task = self.queue.submit(Task(goal='offline continuity',workspace=str(self.root)),
                                      ExecutionPlan(actions=[Action(tool_id='counter')]))

    def process(self, mode, expected=0):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1',
                   DANTE_CONTINUITY_FIXTURE=mode)
        result = subprocess.run([sys.executable,'tests/continuity_process.py',str(self.ledger.path)],
                                env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,expected,result.stdout+result.stderr)

    def count(self):
        path = self.root/'counter.txt'
        return len(path.read_text().splitlines()) if path.exists() else 0

    def test_quota_switch_checkpoint_kill_restart_no_duplicate(self):
        self.process('checkpoint',73)
        self.assertEqual(self.count(),1)
        steps = self.ledger.steps(self.task.task_id)
        self.assertEqual(len(steps),1)
        self.assertEqual(steps[0].state,'succeeded')
        with closing(self.ledger._connect()) as db, db:
            db.execute('UPDATE task_queue SET lease_expires_at=0')
        self.process('resume')
        self.assertEqual(self.count(),1)
        task = self.ledger.get_task(self.task.task_id)
        self.assertEqual(task.task_id,self.task.task_id)
        self.assertEqual(task.status,TaskStatus.COMPLETED)
        self.assertTrue(any(e['event'] == 'acceptance.checked' and json.loads(e['metadata'])['acceptance_complete'] is True for e in self.ledger.events(task.task_id)))
        failures = [e for e in self.ledger.events(task.task_id) if e['event']=='continuity.failure']
        self.assertEqual(len(failures),1)
        self.assertIn('quota_exhausted',failures[0]['metadata'])

    def test_defer_exit_later_available_restart(self):
        self.process('defer')
        status = self.queue.status(self.task.task_id)
        self.assertEqual(status['state'],'ready')
        self.assertEqual(status['task_status'],'failed_retryable')
        self.assertGreater(status['next_attempt_at'],time.time())
        self.assertEqual(self.count(),0)
        self.assertIsNone(self.queue.claim('too-soon',30))
        with closing(self.ledger._connect()) as db, db:
            db.execute('UPDATE task_queue SET next_attempt_at=0')
            db.execute('UPDATE task_continuity SET next_attempt_at=0')
        self.process('restore')
        self.assertEqual(self.ledger.get_task(self.task.task_id).status,TaskStatus.COMPLETED)
        self.assertEqual(self.count(),1)

    def test_retry_decision_crash_before_release_honors_date(self):
        lease = self.queue.claim('old',30)
        with lease_context(lease):
            manager = ContinuityManager(self.ledger, None, None, ContinuityPolicy())
            manager._outcome(self.task,Disposition.RETRY_LATER,'outage',retry_at=time.time()+100)
        with closing(self.ledger._connect()) as db, db:
            db.execute('UPDATE task_queue SET lease_expires_at=0')
        self.queue.recover_expired()
        self.assertIsNone(self.queue.claim('new',30))
        with lease_context(lease), self.assertRaises(LeaseLost):
            manager._outcome(self.task,Disposition.TERMINAL,'stale_owner')

    def test_worker_terminal_for_permanent_policy(self):
        from dante.continuity import ContinuitySignal, ContinuityOutcome
        def denied(ledger,task):
            raise ContinuitySignal(ContinuityOutcome(Disposition.TERMINAL,'no_eligible_route'))
        Worker(self.ledger,denied).run_once()
        self.assertEqual(self.queue.status(self.task.task_id)['state'],'done')
        self.assertEqual(self.ledger.get_task(self.task.task_id).status,TaskStatus.FAILED_TERMINAL)
        self.assertEqual(self.count(),0)

    def test_cancelled_task_never_calls_adapter(self):
        from dante.task_queue import CancellationRequested
        lease = self.queue.claim('owner',30)
        self.queue.cancel(self.task.task_id)
        registry = ModelRegistry(models(),machine_profile='old-pc')
        adapter = FakeAdapter('fake-local')
        manager = ContinuityManager(self.ledger,RuleBasedRouter(registry),
                                    InferenceGateway(registry,[adapter]),ContinuityPolicy())
        manager.observe('fake-local',State.AVAILABLE)
        with lease_context(lease), self.assertRaises(CancellationRequested):
            manager.infer(self.task,PrivacyGate().classify('fixture'),[{'role':'user','content':'fixture'}],set())
        self.assertEqual(adapter.calls,[])
