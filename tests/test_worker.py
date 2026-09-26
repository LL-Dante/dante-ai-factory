import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

from dante.agent_host import AgentHostFoundation
from dante.contracts import Task, TaskStatus, ToolManifest
from dante.ledger import TaskLedger
from dante.task_queue import Action, ExecutionPlan, LeaseLost, TaskQueue, lease_context
from dante.tool_broker import ToolBroker
from dante.worker import Worker
from worker_process import host_factory


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'tasks.db'
        self.ledger = TaskLedger(self.db)
        self.queue = TaskQueue(self.ledger)
        self.task = self.queue.submit(Task(goal='offline counter', workspace=str(self.root)),
                                      ExecutionPlan(actions=[Action(tool_id='counter')]))

    def expire(self):
        # Deterministic logical expiry; no sleeps or dependence on scheduler timing.
        with closing(self.ledger._connect()) as db, db:
            db.execute('UPDATE task_queue SET lease_expires_at=? WHERE task_id=?', (time.time()-1, self.task.task_id))

    def due(self):
        with closing(self.ledger._connect()) as db, db:
            db.execute('UPDATE task_queue SET next_attempt_at=0 WHERE task_id=?', (self.task.task_id,))

    def count(self):
        path = self.root / 'counter.txt'
        return len(path.read_text().splitlines()) if path.exists() else 0

    def process(self, crash='', mode='run', expected=0):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1', DANTE_WORKER_FIXTURE_CRASH=crash)
        result = subprocess.run([sys.executable, 'tests/worker_process.py', str(self.db), mode],
                                env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
        return result

    def cli(self, *args):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1',
                   AI_CLOUD_WORKSPACE=str(self.root), AI_CLOUD_AGENT_AUDIT_LOG=str(self.root/'audit.jsonl'))
        result = subprocess.run([sys.executable, '-m', 'dante', '--db', str(self.db), *args],
                                env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        return json.loads(result.stdout)

    def test_atomic_claim_only_one_worker(self):
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda worker: self.queue.claim(worker, 30), ['one', 'two']))
        self.assertEqual(sum(result is not None for result in results), 1)
        status = self.queue.status(self.task.task_id)
        self.assertEqual((status['state'], status['attempt'], status['generation']), ('leased', 1, 1))
        self.assertIsNone(self.queue.claim('third', 30))

    def test_heartbeat_extends_ownership(self):
        lease = self.queue.claim('one', 30)
        before = self.queue.status(self.task.task_id)['lease_expires_at']
        self.queue.heartbeat(lease, 60)
        after = self.queue.status(self.task.task_id)
        self.assertGreater(after['lease_expires_at'], before)
        self.assertGreaterEqual(after['heartbeat_at'], after['claimed_at'])
        self.assertEqual(self.queue.recover_expired(), 0)
        self.assertIsNone(self.queue.claim('two', 30))

    def test_expired_lease_requires_explicit_recovery_and_fences_old_owner(self):
        old = self.queue.claim('one', 30)
        self.expire()
        self.assertIsNone(self.queue.claim('two', 30))
        self.assertEqual(self.queue.recover_expired(), 1)
        new = self.queue.claim('two', 30)
        self.assertEqual(new.generation, old.generation+1)
        with self.assertRaises(LeaseLost):
            self.queue.heartbeat(old, 30)
        with lease_context(old), self.assertRaises(LeaseLost):
            self.ledger.prepare_step(self.task.task_id, 'counter', '1', {})
        with self.assertRaises(LeaseLost):
            self.queue.release(old)

    def test_crashed_worker_claim_recovered_in_new_process(self):
        self.process(mode='claim', expected=73)
        original = self.queue.status(self.task.task_id)['worker_id']
        self.expire()
        self.process()
        status = self.queue.status(self.task.task_id)
        self.assertEqual(status['task_status'], 'completed')
        self.assertNotEqual(status['worker_id'], original)
        self.assertEqual(self.count(), 1)
        self.assertEqual(status['attempt'], 2)

    def test_verified_effect_not_duplicated_after_process_takeover(self):
        self.process(crash='verified', expected=73)
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.ledger.steps(self.task.task_id)[0].state, 'succeeded')
        self.expire()
        self.process()
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.queue.status(self.task.task_id)['task_status'], 'completed')

    def test_uncertain_effect_blocks_replay_after_process_takeover(self):
        self.process(crash='effect', expected=73)
        self.assertEqual(self.count(), 1)
        self.expire()
        self.process()
        status = self.queue.status(self.task.task_id)
        self.assertEqual((status['state'], status['last_failure']), ('blocked', 'reconciliation_required'))
        self.assertEqual(status['task_status'], 'failed_retryable')
        self.assertEqual(self.ledger.steps(self.task.task_id)[0].state, 'uncertain')
        self.queue.retry(self.task.task_id, delay_s=1)
        self.due()
        self.process()
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.queue.status(self.task.task_id)['state'], 'blocked')

    def test_cancelled_queued_task_never_claimed(self):
        self.queue.cancel(self.task.task_id)
        self.assertEqual(self.ledger.get_task(self.task.task_id).status, TaskStatus.CANCELLED)
        self.assertIsNone(self.queue.claim('one', 30))
        self.assertEqual(self.queue.list(runnable=True), [])

    def test_active_cancellation_observed_at_safe_boundary(self):
        plan = ExecutionPlan(actions=[Action(tool_id='counter'), Action(tool_id='counter')])
        task = self.queue.submit(Task(goal='cancel active', workspace=str(self.root)), plan)
        self.queue.cancel(self.task.task_id)
        def factory(ledger, current):
            host = host_factory(ledger, current)
            original = host.tools._tools['counter'][1]
            def effect():
                result = original()
                self.queue.cancel(task.task_id)
                return result
            manifest = host.tools.manifest('counter')
            host.tools.register(manifest, effect, required_permissions={'write_workspace'})
            return host
        Worker(self.ledger, factory).run_once()
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.queue.status(task.task_id)['task_status'], 'cancelled')
        self.assertEqual(self.queue.status(task.task_id)['next_action'], 1)

    def test_cancel_requested_then_crash_expiry_finalizes_cancellation(self):
        self.queue.claim('one', 30)
        self.queue.cancel(self.task.task_id)
        self.expire()
        self.queue.recover_expired()
        self.assertIsNone(self.queue.claim('two', 30))
        self.assertEqual(self.queue.status(self.task.task_id)['task_status'], 'cancelled')

    def test_retry_respects_time_and_persists_failure_reason(self):
        lease = self.queue.claim('one', 30)
        self.queue.release(lease, blocked='fixture_failure')
        self.queue.retry(self.task.task_id, delay_s=60)
        reopened = TaskQueue(TaskLedger(self.db))
        status = reopened.status(self.task.task_id)
        self.assertEqual(status['retry_count'], 1)
        self.assertEqual(status['last_failure'], 'fixture_failure')
        self.assertGreater(status['next_attempt_at'], time.time())
        self.assertIsNone(reopened.claim('two', 30))
        self.due()
        self.assertIsNotNone(reopened.claim('two', 30))

    def test_graceful_shutdown_preserves_resumable_boundary(self):
        self.queue.cancel(self.task.task_id)
        task = self.queue.submit(Task(goal='two steps', workspace=str(self.root)),
                                 ExecutionPlan(actions=[Action(tool_id='counter'), Action(tool_id='counter')]))
        worker = None
        def factory(ledger, current):
            host = host_factory(ledger, current)
            original = host.tools._tools['counter'][1]
            def effect():
                result = original()
                worker.request_stop()
                return result
            host.tools.register(host.tools.manifest('counter'), effect, required_permissions={'write_workspace'})
            return host
        worker = Worker(self.ledger, factory)
        worker.run()
        status = self.queue.status(task.task_id)
        self.assertEqual((status['state'], status['next_action']), ('ready', 1))
        self.assertNotEqual(status['task_status'], 'completed')
        self.assertEqual(self.count(), 1)
        Worker(TaskLedger(self.db), host_factory).run_once()
        self.assertEqual(self.count(), 2)
        self.assertEqual(self.queue.status(task.task_id)['task_status'], 'completed')

    def test_worker_heartbeat_during_active_handler(self):
        entered, release = threading.Event(), threading.Event()
        def factory(ledger, task):
            host = host_factory(ledger, task)
            def effect():
                entered.set()
                release.wait(5)
                return {'ok': True}
            host.tools.register(host.tools.manifest('counter'), effect, required_permissions={'write_workspace'})
            return host
        worker = Worker(self.ledger, factory, lease_s=1)
        thread = threading.Thread(target=worker.run_once)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            first = self.queue.status(self.task.task_id)['heartbeat_at']
            deadline = time.monotonic()+3
            while self.queue.status(self.task.task_id)['heartbeat_at'] <= first and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertGreater(self.queue.status(self.task.task_id)['heartbeat_at'], first)
            self.assertIsNone(self.queue.claim('other', 30))
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_stale_worker_cannot_persist_effect_success(self):
        lease = self.queue.claim('old', 30)
        with lease_context(lease):
            step = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {})
            self.ledger.claim_step(self.task.task_id, step.step_id)
        self.expire()
        self.queue.recover_expired()
        self.queue.claim('new', 30)
        with lease_context(lease), self.assertRaises(LeaseLost):
            self.ledger.finish_step(self.task.task_id, step.step_id, {'ok': True})
        self.assertEqual(self.ledger.get_step(self.task.task_id, step.step_id).state, 'uncertain')

    def test_live_stale_handler_cannot_override_takeover(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def factory(ledger, task):
            host = host_factory(ledger, task)
            def effect():
                calls.append('effect')
                entered.set()
                release.wait(5)
                return {'ok': True}
            host.tools.register(host.tools.manifest('counter'), effect, required_permissions={'write_workspace'})
            return host
        old = Worker(self.ledger, factory, lease_s=30)
        thread = threading.Thread(target=old.run_once)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.expire()
            replacement = Worker(TaskLedger(self.db), factory)
            replacement.run_once()
            self.assertEqual(self.queue.status(self.task.task_id)['last_failure'], 'reconciliation_required')
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(old.stop.is_set())
        self.assertEqual(calls, ['effect'])
        self.assertEqual(self.ledger.steps(self.task.task_id)[0].state, 'uncertain')
        self.assertEqual(self.queue.status(self.task.task_id)['state'], 'blocked')

    def test_persistent_loop_waits_and_accepts_later_submission(self):
        self.queue.cancel(self.task.task_id)
        worker = Worker(self.ledger, host_factory, poll_s=.05)
        thread = threading.Thread(target=worker.run)
        thread.start()
        try:
            task = self.queue.submit(Task(goal='later', workspace=str(self.root)),
                                     ExecutionPlan(actions=[Action(tool_id='counter')]))
            deadline = time.monotonic()+5
            while self.queue.status(task.task_id)['state'] != 'done' and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(self.queue.status(task.task_id)['task_status'], 'completed')
            self.assertEqual(self.count(), 1)
        finally:
            worker.request_stop()
            thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_unleased_host_cannot_execute_queued_task(self):
        with self.assertRaises(LeaseLost):
            self.ledger.prepare_step(self.task.task_id, 'counter', '1', {})
        self.assertEqual(self.count(), 0)

    def test_cli_submitter_exits_then_offline_worker_completes_and_status_lists(self):
        self.queue.cancel(self.task.task_id)
        plan = self.root / 'plan.json'
        plan.write_text(json.dumps({'actions': [{'tool_id': 'workspace.write_text_file',
                        'arguments': {'relative_path': 'result.txt', 'content': 'offline result'}}]}))
        submitted = self.cli('submit', '--goal', 'offline file', '--plan', str(plan))
        task_id = submitted['task_id']
        self.assertEqual(submitted['task_status'], 'created')
        self.assertIn(task_id, [r['task_id'] for r in self.cli('list', '--runnable')])
        self.cli('worker', '--once')
        self.assertEqual((self.root/'result.txt').read_text(), 'offline result')
        status = self.cli('status', task_id)
        self.assertEqual((status['state'], status['task_status']), ('done', 'completed'))
        self.assertIn(task_id, [r['task_id'] for r in self.cli('list')])

    def test_local_offline_task_acceptance_and_audit(self):
        worker = Worker(self.ledger, host_factory)
        self.assertTrue(worker.run_once())
        self.assertFalse(worker.run_once())
        self.assertEqual(self.count(), 1)
        events = {r['event'] for r in self.ledger.events(self.task.task_id)}
        self.assertTrue({'worker.claimed','worker.released','worker.checkpoint','step.succeeded','acceptance.checked'} <= events)
        self.assertEqual(self.ledger.get_task(self.task.task_id).status, TaskStatus.COMPLETED)

    def test_no_hot_retry_and_stopped_worker_does_not_claim(self):
        worker = Worker(self.ledger, lambda *_: (_ for _ in ()).throw(ValueError('fixture')))
        self.assertTrue(worker.run_once())
        self.assertFalse(worker.run_once())
        self.assertEqual(self.queue.status(self.task.task_id)['state'], 'blocked')
        self.queue.retry(self.task.task_id)
        self.due()
        worker.request_stop()
        self.assertFalse(worker.run_once())

    def test_reject_secret_plan_without_partial_task(self):
        task = Task(goal='no secrets', workspace=str(self.root))
        with self.assertRaises(ValueError):
            self.queue.submit(task, ExecutionPlan(actions=[Action(tool_id='counter', arguments={'password': 'fixture'})]))
        with self.assertRaises(KeyError):
            self.ledger.get_task(task.task_id)

    def test_additive_migration_preserves_existing_p3_data(self):
        older = self.root/'old.db'
        ledger = TaskLedger(older)
        historical = ledger.create_task(Task(goal='historical', workspace=str(self.root)))
        with closing(sqlite3.connect(older)) as db, db:
            db.execute('DROP TABLE backend_observations')
            db.execute('DROP TABLE task_continuity')
            db.execute('DROP TABLE task_queue')
            db.execute('PRAGMA user_version=1')
        migrated = TaskLedger(older)
        self.assertEqual(migrated.get_task(historical.task_id), historical)
        self.assertEqual(migrated.events(historical.task_id)[0]['event'], 'task.created')
        with closing(sqlite3.connect(older)) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 5)
        self.assertEqual(TaskQueue(migrated).list(), [])


if __name__ == '__main__':
    unittest.main()
