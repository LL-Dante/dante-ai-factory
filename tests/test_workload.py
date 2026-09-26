from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from dante.workload import (IdempotencyConflict, JobState, JobTransitionError, ModelRequirement,
    Node0InferenceExecutor, OrchestratorAlreadyRunning, WorkloadOrchestrator, WorkloadSpec, WorkloadStore)


DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'


def spec(**changes):
    data = dict(model=ModelRequirement(model_id='qwen3:4b', runtime_reference='qwen3:4b',
        digest_sha256=DIGEST, context_tokens=4096), prompt='Say local ready.', max_output_tokens=24,
        timeout_s=2, maximum_attempts=2, retry_base_s=0.1, retry_max_s=0.2)
    data.update(changes)
    return WorkloadSpec(**data)


class FakeExecutor:
    def __init__(self, *, text='local answer', error=None, block_until_cancel=False):
        self.text, self.error, self.block_until_cancel = text, error, block_until_cancel
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def execute(self, request, cancellation):
        self.started.set()
        if self.block_until_cancel:
            while not cancellation.wait(0.01):
                pass
            self.cancelled.set()
        if self.error:
            raise self.error
        return type('Result', (), {'text': self.text, 'model_id': 'qwen3:4b', 'provider_id': 'ollama',
            'fallback': False, 'runtime': 'ollama', 'metadata': {'qualification_id': 'physical-test'}})()


class GatedExecutor:
    def __init__(self, *, wait_for_cancel=False):
        self.started = threading.Event()
        self.cancel_seen = threading.Event()
        self.release = threading.Event()
        self.wait_for_cancel = wait_for_cancel

    def execute(self, request, cancellation):
        self.started.set()
        if self.wait_for_cancel:
            cancellation.wait(2)
            self.cancel_seen.set()
        self.release.wait(3)
        if cancellation.is_set():
            from dante.inference import InferenceCancelled
            raise InferenceCancelled('cancelled')
        return type('Result', (), {'text': 'gated answer', 'model_id': 'qwen3:4b', 'provider_id': 'ollama',
            'fallback': False, 'runtime': 'ollama', 'metadata': {}})()


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class WorkloadStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'ledger.db'
        self.store = WorkloadStore(self.db, busy_timeout_ms=1000)

    def tearDown(self):
        self.temp.cleanup()

    def test_submit_reopen_and_safe_status_omits_prompt(self):
        created = self.store.submit(spec(prompt='private prompt value'))
        reopened = WorkloadStore(self.db)
        self.assertEqual(reopened.get(created.job_id).state, JobState.QUEUED)
        self.assertNotIn('private prompt value', reopened.get(created.job_id).model_dump_json())
        self.assertEqual(reopened.spec(created.job_id).prompt, 'private prompt value')

    def test_idempotency_returns_same_job_or_conflicts(self):
        first = self.store.submit(spec(), idempotency_key='client-1')
        second = self.store.submit(spec(), idempotency_key='client-1')
        self.assertEqual(first.job_id, second.job_id)
        with self.assertRaises(IdempotencyConflict):
            self.store.submit(spec(prompt='different input'), idempotency_key='client-1')
        self.assertEqual(len(self.store.list()), 1)

    def test_priority_order_and_terminal_jobs_are_not_requeued(self):
        low = self.store.submit(spec(priority=-1))
        high = self.store.submit(spec(priority=5))
        claimed, _ = self.store.claim('worker-a')
        self.assertEqual(claimed.job_id, high.job_id)
        self.assertEqual(self.store.request_cancel(low.job_id).state, JobState.CANCELLED)
        self.assertIsNone(self.store.claim('worker-b'))

    def test_cancel_wins_success_race(self):
        job = self.store.submit(spec())
        claimed, attempt = self.store.claim('worker')
        self.assertTrue(self.store.start_attempt(job.job_id, attempt))
        self.assertEqual(self.store.request_cancel(job.job_id).state, JobState.CANCEL_REQUESTED)
        result = self.store.succeed(job.job_id, attempt, {'text': 'discard this'})
        self.assertEqual(result.state, JobState.CANCELLED)
        self.assertIsNone(result.result)
        with self.assertRaises(JobTransitionError):
            self.store.succeed(job.job_id, attempt, {'text': 'again'})

    def test_cancel_queued_claimed_and_retry_wait(self):
        queued = self.store.submit(spec())
        self.assertEqual(self.store.request_cancel(queued.job_id).state, JobState.CANCELLED)
        claimed_job = self.store.submit(spec())
        claimed, _attempt = self.store.claim('worker')
        self.assertEqual(claimed.job_id, claimed_job.job_id)
        self.assertEqual(self.store.request_cancel(claimed_job.job_id).state, JobState.CANCELLED)

        retry_job = self.store.submit(spec())
        _job, attempt = self.store.claim('worker')
        self.store.start_attempt(retry_job.job_id, attempt)
        self.assertEqual(self.store.fail(retry_job.job_id, attempt, code='runtime_unavailable', retryable=True).state,
                         JobState.RETRY_WAIT)
        self.assertEqual(self.store.request_cancel(retry_job.job_id).state, JobState.CANCELLED)

    def test_retry_is_bounded_and_uses_new_attempt_id(self):
        job = self.store.submit(spec(maximum_attempts=2, retry_base_s=0.1, retry_max_s=0.1))
        _claimed, first = self.store.claim('worker')
        self.store.start_attempt(job.job_id, first)
        retry = self.store.fail(job.job_id, first, code='runtime_timeout', retryable=True)
        self.assertEqual(retry.state, JobState.RETRY_WAIT)
        with closing(self.store._connect()) as db, db:
            db.execute('UPDATE workload_jobs SET eligible_at=? WHERE job_id=?', (time.time() - 1, job.job_id))
        _claimed, second = self.store.claim('worker')
        self.assertNotEqual(first, second)
        self.store.start_attempt(job.job_id, second)
        final = self.store.fail(job.job_id, second, code='runtime_timeout', retryable=True)
        self.assertEqual(final.state, JobState.FAILED)
        self.assertEqual([a.state for a in self.store.attempts(job.job_id)], ['failed', 'failed'])

    def test_nonretryable_failure_and_expired_deadline(self):
        invalid = self.store.submit(spec())
        _job, attempt = self.store.claim('worker')
        self.store.start_attempt(invalid.job_id, attempt)
        self.assertEqual(self.store.fail(invalid.job_id, attempt, code='qualification_denied', retryable=False).state,
                         JobState.FAILED)
        expired = self.store.submit(spec(deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
        self.assertIsNone(self.store.claim('worker'))
        self.assertEqual(self.store.get(expired.job_id).state, JobState.TIMED_OUT)

    def test_result_size_is_bounded(self):
        job = self.store.submit(spec())
        _job, attempt = self.store.claim('worker')
        self.store.start_attempt(job.job_id, attempt)
        with self.assertRaises(ValueError):
            self.store.succeed(job.job_id, attempt, {'text': 'x' * 70000})
        self.assertEqual(self.store.get(job.job_id).state, JobState.RUNNING)

    def test_transactions_rollback_and_schema_version(self):
        with closing(self.store._connect()) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 5)
            count = db.execute('SELECT COUNT(*) FROM workload_jobs').fetchone()[0]
            with self.assertRaises(RuntimeError):
                with db:
                    db.execute("INSERT INTO workload_jobs(job_id,request_digest,payload,state,priority,created_at,created_epoch,updated_at,eligible_at,attempt_count,cancel_requested) VALUES('rollback','d','{}','queued',0,'now',0,'now',0,0,0)")
                    raise RuntimeError('rollback')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM workload_jobs').fetchone()[0], count)

    def test_concurrent_claim_has_one_winner(self):
        self.store.submit(spec())
        barrier = threading.Barrier(8)
        def claim(index):
            barrier.wait(timeout=2)
            return self.store.claim(f'worker-{index}')
        pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='claim-test')
        futures = []
        try:
            for index in range(8):
                futures.append(pool.submit(claim, index))
            done, pending = wait(futures, timeout=6)
            self.assertFalse(pending, 'Concurrent claim workers exceeded their deadline')
            results = [future.result(timeout=0) for future in futures]
        finally:
            barrier.abort()
            pool.shutdown(wait=True, cancel_futures=True)
        self.assertEqual(sum(item is not None for item in results), 1)

    def test_active_owner_prevents_duplicate_orchestrator(self):
        first = WorkloadOrchestrator(self.store, FakeExecutor())
        first.start()
        second = WorkloadOrchestrator(WorkloadStore(self.db), FakeExecutor())
        with self.assertRaises(OrchestratorAlreadyRunning):
            second.start()
        first.stop()

    def test_interrupted_running_attempt_is_retried_and_preserved(self):
        job = self.store.submit(spec())
        _claimed, attempt = self.store.claim('dead-worker')
        self.store.start_attempt(job.job_id, attempt)
        recovered = self.store.recover_interrupted()
        self.assertEqual(recovered, [job.job_id])
        self.assertEqual(self.store.get(job.job_id).state, JobState.RETRY_WAIT)
        self.assertEqual(self.store.attempts(job.job_id)[0].state, 'interrupted')

    def test_audit_never_contains_prompt_or_result_text(self):
        job = self.store.submit(spec(prompt='audit-private-prompt'))
        _claimed, attempt = self.store.claim('worker')
        self.store.start_attempt(job.job_id, attempt)
        self.store.succeed(job.job_id, attempt, {'text': 'audit-private-response'})
        encoded = json.dumps(self.store.events(job.job_id))
        self.assertNotIn('audit-private-prompt', encoded)
        self.assertNotIn('audit-private-response', encoded)


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WorkloadStore(Path(self.temp.name) / 'ledger.db')

    def tearDown(self):
        self.temp.cleanup()

    def test_dispatch_persists_success_and_releases_resource(self):
        job = self.store.submit(spec())
        executor = FakeExecutor()
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        self.assertTrue(orchestrator.run_once())
        orchestrator.stop()
        record = self.store.get(job.job_id)
        self.assertEqual(record.state, JobState.SUCCEEDED)
        self.assertEqual(record.attempt_count, 1)
        self.assertEqual(record.result['text'], 'local answer')
        self.assertTrue(orchestrator.resource_gate.acquire(timeout=0))
        orchestrator.resource_gate.release()
        self.assertIn('route_selected', [e['event'] for e in self.store.events(job.job_id)])

    def test_execution_timeout_is_terminal_or_bounded_retry_not_success(self):
        job = self.store.submit(spec(timeout_s=0.1, maximum_attempts=1))
        executor = FakeExecutor(block_until_cancel=True)
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        orchestrator.run_once()
        orchestrator.stop()
        self.assertEqual(self.store.get(job.job_id).state, JobState.FAILED)
        self.assertNotEqual(self.store.get(job.job_id).state, JobState.SUCCEEDED)

    def test_lifecycle_normal_return_releases_worker_and_gpu_slot(self):
        job = self.store.submit(spec())
        orchestrator = WorkloadOrchestrator(self.store, FakeExecutor())
        orchestrator.start()
        try:
            self.assertTrue(orchestrator.run_once())
            self.assertEqual(self.store.get(job.job_id).state, JobState.SUCCEEDED)
            self.assertTrue(orchestrator.resource_gate.acquire(timeout=0))
            orchestrator.resource_gate.release()
            self.assertFalse(any(t.name.startswith('dante-local-inference-') or
                                 t.name.startswith('dante-inference-reaper-') for t in threading.enumerate()))
        finally:
            orchestrator.stop()

    def test_lifecycle_timeout_returns_without_waiting_for_stuck_inference(self):
        job = self.store.submit(spec(timeout_s=0.08, maximum_attempts=1))
        executor = GatedExecutor()
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        start = time.monotonic()
        try:
            orchestrator.run_once()
            self.assertLess(time.monotonic() - start, 1.0)
            self.assertEqual(self.store.get(job.job_id).state, JobState.FAILED)
            self.assertTrue(executor.started.is_set())
            self.assertFalse(orchestrator.resource_gate.acquire(timeout=0))
        finally:
            executor.release.set()
            self.assertTrue(wait_until(lambda: not orchestrator._active))
            orchestrator.stop()

    def test_lifecycle_cooperative_cancellation_finishes_durably(self):
        job = self.store.submit(spec(timeout_s=5))
        executor = GatedExecutor(wait_for_cancel=True)
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        runner = threading.Thread(target=orchestrator.run_once)
        runner.start()
        try:
            self.assertTrue(executor.started.wait(1))
            self.assertEqual(orchestrator.cancel(job.job_id).state, JobState.CANCEL_REQUESTED)
            self.assertTrue(executor.cancel_seen.wait(1))
            executor.release.set()
            runner.join(2)
            self.assertFalse(runner.is_alive())
            self.assertEqual(self.store.get(job.job_id).state, JobState.CANCELLED)
        finally:
            executor.release.set()
            runner.join(2)
            orchestrator.stop()

    def test_lifecycle_delayed_cancellation_keeps_attempt_active(self):
        job = self.store.submit(spec(timeout_s=5))
        executor = GatedExecutor(wait_for_cancel=True)
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        runner = threading.Thread(target=orchestrator.run_once)
        runner.start()
        try:
            self.assertTrue(executor.started.wait(1))
            orchestrator.cancel(job.job_id)
            self.assertTrue(executor.cancel_seen.wait(1))
            time.sleep(0.05)
            self.assertTrue(runner.is_alive())
            self.assertEqual(self.store.get(job.job_id).state, JobState.CANCEL_REQUESTED)
            self.assertFalse(orchestrator.resource_gate.acquire(timeout=0))
        finally:
            executor.release.set()
            runner.join(2)
            orchestrator.stop()

    def test_lifecycle_stop_with_active_inference_retains_owner(self):
        job = self.store.submit(spec(timeout_s=5))
        executor = GatedExecutor()
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        runner = threading.Thread(target=orchestrator.run_once)
        runner.start()
        try:
            self.assertTrue(executor.started.wait(1))
            orchestrator.stop(drain_timeout_s=0.05)
            self.assertEqual(orchestrator.state, 'DEGRADED')
            with closing(self.store._connect()) as db:
                self.assertEqual(db.execute('SELECT owner_id FROM workload_control').fetchone()[0], orchestrator.owner_id)
            self.assertTrue(runner.is_alive())
        finally:
            executor.release.set()
            runner.join(2)
            self.assertTrue(wait_until(lambda: orchestrator.state == 'STOPPED'))

    def test_lifecycle_gpu_slot_remains_owned_until_execution_ends(self):
        self.store.submit(spec(timeout_s=0.08, maximum_attempts=1))
        executor = GatedExecutor()
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        try:
            orchestrator.run_once()
            self.assertFalse(orchestrator.resource_gate.acquire(timeout=0))
            executor.release.set()
            self.assertTrue(wait_until(lambda: not orchestrator._active))
            self.assertTrue(orchestrator.resource_gate.acquire(timeout=0))
            orchestrator.resource_gate.release()
        finally:
            executor.release.set()
            orchestrator.stop()

    def test_lifecycle_owner_released_only_after_worker_terminates(self):
        self.store.submit(spec(timeout_s=5))
        executor = GatedExecutor()
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        runner = threading.Thread(target=orchestrator.run_once)
        runner.start()
        try:
            self.assertTrue(executor.started.wait(1))
            orchestrator.request_stop()
            orchestrator.stop(drain_timeout_s=0.02)
            with closing(self.store._connect()) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM workload_control WHERE owner_id=?',
                                             (orchestrator.owner_id,)).fetchone()[0], 1)
            executor.release.set()
            runner.join(2)
            self.assertTrue(wait_until(lambda: orchestrator.state == 'STOPPED'))
            with closing(self.store._connect()) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM workload_control WHERE owner_id=?',
                                             (orchestrator.owner_id,)).fetchone()[0], 0)
        finally:
            executor.release.set()
            runner.join(2)

    def test_lifecycle_timeout_cannot_succeed_after_late_return(self):
        job = self.store.submit(spec(timeout_s=0.08, maximum_attempts=1))
        executor = GatedExecutor()
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        try:
            orchestrator.run_once()
            self.assertEqual(self.store.get(job.job_id).state, JobState.FAILED)
            executor.release.set()
            self.assertTrue(wait_until(lambda: not orchestrator._active))
            self.assertEqual(self.store.get(job.job_id).state, JobState.FAILED)
            self.assertIsNone(self.store.get(job.job_id).result)
        finally:
            executor.release.set()
            orchestrator.stop()

    def test_lifecycle_cancel_success_race_cannot_succeed(self):
        job = self.store.submit(spec(timeout_s=5))
        executor = GatedExecutor()
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        runner = threading.Thread(target=orchestrator.run_once)
        runner.start()
        try:
            self.assertTrue(executor.started.wait(1))
            self.store.request_cancel(job.job_id)
            executor.release.set()
            runner.join(2)
            self.assertFalse(runner.is_alive())
            self.assertEqual(self.store.get(job.job_id).state, JobState.CANCELLED)
            self.assertIsNone(self.store.get(job.job_id).result)
        finally:
            executor.release.set()
            runner.join(2)
            orchestrator.stop()

    def test_lifecycle_no_worker_or_reaper_remains_after_completion(self):
        job = self.store.submit(spec())
        orchestrator = WorkloadOrchestrator(self.store, FakeExecutor())
        orchestrator.start()
        try:
            orchestrator.run_once()
            self.assertEqual(self.store.get(job.job_id).state, JobState.SUCCEEDED)
            self.assertTrue(wait_until(lambda: not any(t.name.startswith('dante-local-inference-') or
                t.name.startswith('dante-inference-reaper-') for t in threading.enumerate())))
        finally:
            orchestrator.stop()

    def test_running_cancel_propagates_and_never_succeeds(self):
        job = self.store.submit(spec(timeout_s=5))
        executor = FakeExecutor(block_until_cancel=True)
        orchestrator = WorkloadOrchestrator(self.store, executor)
        orchestrator.start()
        thread = threading.Thread(target=orchestrator.run_once)
        thread.start()
        self.assertTrue(executor.started.wait(2))
        self.assertEqual(self.store.request_cancel(job.job_id).state, JobState.CANCEL_REQUESTED)
        thread.join(3)
        orchestrator.stop()
        self.assertFalse(thread.is_alive())
        self.assertTrue(executor.cancelled.is_set())
        self.assertEqual(self.store.get(job.job_id).state, JobState.CANCELLED)
        self.assertIsNone(self.store.get(job.job_id).result)

    def test_node0_executor_uses_existing_supervisor_interface(self):
        metadata = type('Metadata', (), {'runtime_reference': 'qwen3:4b', 'runtime_digest': DIGEST})()
        model = type('Model', (), {'model_id': 'qwen3-4b-node0', 'provider_id': 'ollama',
            'runtime': 'ollama', 'local_metadata': metadata})()
        response = type('Response', (), {'fallback': False, 'provider_id': 'ollama',
            'model': model, 'content': 'ready'})()
        class Supervisor:
            calls = []
            def operator_status(self):
                return {'ready_for_local_routing': True, 'gate_accepted': True}
            def infer(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return response
            def operator_health(self):
                return {'healthy': True, 'gpu_execution_verified': True, 'qualification_id': 'q-1',
                    'gpu_uuid': 'GPU-test', 'gpu_vram_bytes': 1024, 'runtime_version': 'fixture'}
        supervisor = Supervisor()
        executor = Node0InferenceExecutor(supervisor, expected_model_id='qwen3-4b-node0',
            expected_runtime_reference='qwen3:4b', expected_digest=DIGEST)
        request = spec(model=ModelRequirement(model_id='qwen3-4b-node0', runtime_reference='qwen3:4b',
            digest_sha256=DIGEST, context_tokens=4096))
        result = executor.execute(request, threading.Event())
        self.assertEqual(result.text, 'ready')
        self.assertEqual(supervisor.calls[0][0], 'Say local ready.')
        self.assertEqual(supervisor.calls[0][1]['model_reference'], 'qwen3:4b')
        self.assertEqual(result.metadata['gpu_identity'], 'GPU-test')

    def test_node0_executor_rejects_unready_production_supervisor(self):
        class Supervisor:
            def operator_status(self):
                return {'ready_for_local_routing': False, 'gate_accepted': False}
            def infer(self, *_args, **_kwargs):
                raise AssertionError('unready supervisor must not be called')
        executor = Node0InferenceExecutor(Supervisor(), expected_model_id='qwen3-4b-node0',
            expected_runtime_reference='qwen3:4b', expected_digest=DIGEST)
        with self.assertRaises(Exception):
            executor.execute(spec(model=ModelRequirement(model_id='qwen3-4b-node0',
                runtime_reference='qwen3:4b', digest_sha256=DIGEST, context_tokens=4096)), threading.Event())


if __name__ == '__main__':
    unittest.main()
