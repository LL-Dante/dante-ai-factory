import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

from dante.acceptance import AcceptanceContract, ObservedAcceptance
from dante.agent_host import AgentHostFoundation
from dante.artifacts import ArtifactStore
from dante.contracts import PrivacyClass, Task, TaskStatus, ToolManifest
from dante.ledger import TaskLedger, InvalidTransition
from dante.recovery import ReconciliationRequired, verified_receipt
from dante.tool_broker import ToolBroker


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root / 'tasks.db')
        self.host = AgentHostFoundation(self.ledger, None, None, None, ToolBroker())
        self.task = self.host.start('fixture', str(self.root), PrivacyClass.PUBLIC,
                                    acceptance=AcceptanceContract(required_tools={'counter'}))

    def process(self, window, expected=0):
        env = dict(os.environ, PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1')
        run = subprocess.run([sys.executable, 'tests/recovery_process.py', str(self.root), self.task.task_id, window],
                             capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(run.returncode, expected, run.stderr + run.stdout)
        return run

    def count(self):
        path = self.root / 'counter.txt'
        return len(path.read_text().splitlines()) if path.exists() else 0

    def step(self):
        return self.ledger.steps(self.task.task_id)[0]

    def reconcile(self, outcome):
        step = self.step()
        certificate = {'step_id': step.step_id, 'arguments_digest': step.arguments_digest,
                       'outcome': outcome, 'result': {'ok': True}}
        artifact = ArtifactStore(self.root / 'artifacts', self.ledger).create(
            self.task.task_id, 'reconciliation.json', json.dumps(certificate).encode(),
            media_type='application/json', producer='fixture-counter-inspector', provenance={}, step_id=step.step_id)
        return self.ledger.reconcile_step(self.task.task_id, step.step_id, artifact.artifact_id,
                                          actor='fixture-inspector', executor_stopped=True)

    def test_crash_A_intent_before_effect_can_execute(self):
        self.process('A', 71)
        self.assertEqual(self.count(), 0)
        self.assertEqual(self.step().state, 'intent')
        self.assertEqual(self.step().attempt, 0)
        self.process('normal')
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.step().state, 'succeeded')

    def test_crash_B_started_without_outcome_is_uncertain(self):
        self.process('B', 71)
        self.assertEqual(self.count(), 0)
        self.process('normal', 3)
        self.assertEqual(self.count(), 0)
        self.assertEqual(self.ledger.get_task(self.task.task_id).status, TaskStatus.FAILED_RETRYABLE)
        self.reconcile('not_applied')
        self.process('normal')
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.step().attempt, 2)

    def test_crash_C_after_effect_does_not_blindly_replay(self):
        self.process('C', 71)
        self.assertEqual(self.count(), 1)
        self.process('normal', 3)
        self.assertEqual(self.count(), 1)
        self.reconcile('applied')
        self.process('normal')
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.step().attempt, 1)

    def test_crash_D_persisted_result_before_checkpoint_reused(self):
        self.process('D', 71)
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.step().state, 'succeeded')
        self.assertEqual(self.ledger.get_task(self.task.task_id).checkpoint, {})
        self.process('normal')
        self.process('normal')
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.ledger.get_task(self.task.task_id).checkpoint['step_id'], self.step().step_id)

    def test_acceptance_survives_process_restart(self):
        result = self.process('inspect')
        self.assertEqual(json.loads(result.stdout)['required_tools'], ['counter'])

    def test_omitted_contract_and_direct_completion_are_blocked(self):
        reopened = TaskLedger(self.root / 'tasks.db')
        host = AgentHostFoundation(reopened, None, None, None, None)
        finished = host.complete(self.task.task_id)
        self.assertEqual(finished.status, TaskStatus.FAILED_RETRYABLE)
        host.resume(self.task.task_id)
        reopened.transition(self.task.task_id, TaskStatus.VERIFYING)
        with self.assertRaises(InvalidTransition):
            reopened.transition(self.task.task_id, TaskStatus.COMPLETED)
        with self.assertRaises(ValueError):
            reopened.bind_acceptance(self.task.task_id, AcceptanceContract())

    def test_in_memory_claims_cannot_satisfy_operational_acceptance(self):
        result = self.host.complete(self.task.task_id, observed=ObservedAcceptance(tools={'counter'}))
        self.assertEqual(result.status, TaskStatus.FAILED_RETRYABLE)

    def test_complete_from_persisted_tool_artifact_and_final_evidence(self):
        # A separate operational task with an explicit complete evidence contract.
        task = self.host.start('evidence', str(self.root), PrivacyClass.PUBLIC, acceptance=AcceptanceContract(
            required_tools={'write'}, required_artifacts={'result.json'}, required_final_fields={'answer'}))
        broker = ToolBroker()
        broker.register(ToolManifest(tool_id='write', permissions=set(), risk='R1', filesystem_scope='none', arguments_schema={'type': 'object'}), lambda: {'ok': True}, required_permissions=set())
        host = AgentHostFoundation(self.ledger, None, None, None, broker)
        host.execute_tool_and_checkpoint(task.task_id, 'write', {})
        step = self.ledger.steps(task.task_id)[0]
        artifact = ArtifactStore(self.root / 'artifacts', self.ledger).create(task.task_id, 'result.json', b'{"answer":"verified"}',
            media_type='application/json', producer='fixture-verifier', provenance={}, step_id=step.step_id)
        self.ledger.record_final_evidence(task.task_id, artifact.artifact_id)
        reopened = TaskLedger(self.root / 'tasks.db')
        finished = AgentHostFoundation(reopened, None, None, None, None).complete(task.task_id)
        self.assertEqual(finished.status, TaskStatus.COMPLETED)
        self.assertEqual(artifact.provenance['step_id'], step.step_id)

    def test_legacy_task_enters_evidence_mode_before_first_effect(self):
        task = self.host.start('legacy entry', str(self.root), PrivacyClass.PUBLIC)
        self.ledger.prepare_step(task.task_id, 'counter', '1', {})
        self.assertEqual(self.ledger.get_acceptance(task.task_id).required_tools, {'counter'})
        result = self.host.complete(task.task_id, observed=ObservedAcceptance(tools={'counter'}))
        self.assertEqual(result.status, TaskStatus.FAILED_RETRYABLE)

    def test_handler_failure_after_effect_is_not_retried_automatically(self):
        count = []
        def fail():
            count.append('effect')
            raise RuntimeError('private failure details')
        broker = ToolBroker()
        broker.register(ToolManifest(tool_id='counter', permissions=set(), risk='R1', filesystem_scope='none', arguments_schema={'type': 'object'}), fail, required_permissions=set())
        host = AgentHostFoundation(self.ledger, None, None, None, broker)
        with self.assertRaises(RuntimeError):
            host.execute_tool_and_checkpoint(self.task.task_id, 'counter', {})
        self.assertEqual(self.step().state, 'uncertain')
        self.assertEqual(self.step().error, 'RuntimeError')
        with self.assertRaises(ReconciliationRequired):
            host.execute_tool_and_checkpoint(self.task.task_id, 'counter', {})
        self.assertEqual(count, ['effect'])

    def test_same_key_different_arguments_rejected(self):
        first = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {'x': 1}, 'key')
        again = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {'x': 1}, 'key')
        self.assertEqual(first.step_id, again.step_id)
        with self.assertRaises(ValueError):
            self.ledger.prepare_step(self.task.task_id, 'counter', '1', {'x': 2}, 'key')

    def test_atomic_claim_only_one_executor(self):
        step = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {}, 'key')
        def claim():
            try:
                self.ledger.claim_step(self.task.task_id, step.step_id)
                return True
            except ReconciliationRequired:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(lambda _: claim(), range(2))), [False, True])

    def test_journal_does_not_store_secret_arguments_or_arbitrary_output(self):
        secret = 'fixture-secret-not-for-storage'
        step = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {'password': secret})
        self.ledger.claim_step(self.task.task_id, step.step_id)
        self.ledger.finish_step(self.task.task_id, step.step_id, {'ok': True, 'password': secret})
        self.assertNotIn(secret, self.ledger.get_step(self.task.task_id, step.step_id).model_dump_json())

    def test_uncertain_blocks_completion_even_when_required_tool_succeeded(self):
        self.process('normal')
        step = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {}, 'second')
        self.ledger.claim_step(self.task.task_id, step.step_id)
        self.ledger.mark_uncertain(self.task.task_id, step.step_id, 'FixtureError')
        result = self.host.complete(self.task.task_id)
        self.assertEqual(result.status, TaskStatus.FAILED_RETRYABLE)

    def test_reconciliation_requires_quiescence_and_matching_evidence(self):
        self.process('B', 71)
        with self.assertRaises(ReconciliationRequired):
            self.ledger.reconcile_step(self.task.task_id, self.step().step_id, 'missing', actor='fixture', executor_stopped=False)
        artifact = ArtifactStore(self.root / 'artifacts', self.ledger).create(self.task.task_id, 'wrong.json', b'{}',
            media_type='application/json', producer='fixture', provenance={})
        with self.assertRaises(ValueError):
            self.ledger.reconcile_step(self.task.task_id, self.step().step_id, artifact.artifact_id, actor='fixture', executor_stopped=True)

    def test_lost_reconciliation_evidence_blocks_reuse(self):
        step = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {})
        self.ledger.claim_step(self.task.task_id, step.step_id)
        saved = self.reconcile('applied')
        artifact = self.ledger.get_artifact(saved.evidence_ref)
        Path(artifact.path).write_bytes(b'corrupt')
        with self.assertRaises(ReconciliationRequired):
            self.ledger.verified_step_result(self.task.task_id, step.step_id)
        self.assertFalse(self.ledger.verify_persistent_acceptance(self.task.task_id).acceptance_complete)

    def test_lost_absence_evidence_blocks_retry(self):
        step = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {})
        self.ledger.claim_step(self.task.task_id, step.step_id)
        saved = self.reconcile('not_applied')
        artifact = self.ledger.get_artifact(saved.evidence_ref)
        Path(artifact.path).write_bytes(b'corrupt')
        with self.assertRaises(ReconciliationRequired):
            self.ledger.claim_step(self.task.task_id, step.step_id)
        self.assertEqual(self.step().attempt, 1)

    def test_file_receipt_detects_changed_evidence(self):
        path = self.root / 'effect.txt'
        path.write_text('original')
        step = self.ledger.prepare_step(self.task.task_id, 'counter', '1', {})
        self.ledger.claim_step(self.task.task_id, step.step_id)
        saved = self.ledger.finish_step(self.task.task_id, step.step_id, {'ok': True, 'path': str(path)})
        path.write_text('tampered')
        with self.assertRaises(ReconciliationRequired):
            verified_receipt(saved)
        self.assertFalse(self.ledger.verify_persistent_acceptance(self.task.task_id).acceptance_complete)


class ArtifactMigrationTests(unittest.TestCase):
    def test_different_content_same_name_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = TaskLedger(root / 'db')
            task = ledger.create_task(Task(goal='artifact', workspace=directory))
            store = ArtifactStore(root / 'cas', ledger)
            a = store.create(task.task_id, 'same.txt', b'first', media_type='text/plain', producer='fixture', provenance={})
            b = store.create(task.task_id, 'same.txt', b'second', media_type='text/plain', producer='fixture', provenance={})
            self.assertNotEqual(a.path, b.path)
            self.assertEqual(Path(a.path).read_bytes(), b'first')
            self.assertEqual(Path(b.path).read_bytes(), b'second')
            self.assertEqual(ledger.get_artifact(a.artifact_id).checksum_sha256, a.checksum_sha256)

    def test_identical_content_deduplicates_across_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = TaskLedger(root / 'db')
            one = ledger.create_task(Task(goal='one', workspace=directory))
            two = ledger.create_task(Task(goal='two', workspace=directory))
            store = ArtifactStore(root / 'cas', ledger)
            a = store.create(one.task_id, 'a.txt', b'same', media_type='text/plain', producer='fixture', provenance={})
            b = store.create(two.task_id, 'b.txt', b'same', media_type='text/plain', producer='fixture', provenance={})
            self.assertEqual(a.path, b.path)
            self.assertNotEqual(a.artifact_id, b.artifact_id)
            self.assertNotEqual(a.task_id, b.task_id)
            self.assertEqual(len(list((root / 'cas' / 'blobs').rglob(a.checksum_sha256))), 1)

    def test_corrupt_cas_blob_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = TaskLedger(Path(directory) / 'db')
            task = ledger.create_task(Task(goal='one', workspace=directory))
            store = ArtifactStore(Path(directory) / 'cas', ledger)
            a = store.create(task.task_id, 'a', b'same', media_type='text/plain', producer='fixture', provenance={})
            Path(a.path).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                store.create(task.task_id, 'a', b'same', media_type='text/plain', producer='fixture', provenance={})
            self.assertEqual(Path(a.path).read_bytes(), b'corrupt')

    def test_migrate_unversioned_foundation_preserves_task_events_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.db'
            task = Task(goal='historical', workspace=directory, checkpoint={'old': 'kept'})
            with closing(sqlite3.connect(path)) as db, db:
                db.executescript('''
                    CREATE TABLE tasks(task_id TEXT PRIMARY KEY,payload TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE task_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT NOT NULL,event TEXT NOT NULL,
                      from_status TEXT,to_status TEXT,trace_id TEXT NOT NULL,timestamp_utc TEXT NOT NULL,metadata TEXT NOT NULL);
                    CREATE TABLE routes(route_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,payload TEXT NOT NULL);
                    CREATE TABLE artifacts(artifact_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,payload TEXT NOT NULL);
                    CREATE TABLE approvals(approval_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,payload TEXT NOT NULL,token_hash TEXT NOT NULL,consumed INTEGER NOT NULL DEFAULT 0);
                ''')
                db.execute('INSERT INTO tasks VALUES(?,?,?)', (task.task_id, task.model_dump_json(), 7))
                db.execute('INSERT INTO task_events(task_id,event,trace_id,timestamp_utc,metadata) VALUES(?,?,?,?,?)',
                           (task.task_id, 'historical', task.trace_id, '2026-09-01', '{}'))
            ledger = TaskLedger(path)
            again = TaskLedger(path)
            self.assertEqual(again.get_task(task.task_id), task)
            self.assertEqual(ledger.events(task.task_id)[0]['event'], 'historical')
            self.assertEqual(again.get_acceptance(task.task_id), AcceptanceContract())
            with closing(sqlite3.connect(path)) as db, db:
                self.assertEqual(db.execute('SELECT revision FROM tasks').fetchone()[0], 7)
                self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 5)
                self.assertEqual(db.execute('SELECT count(*) FROM task_acceptance').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
