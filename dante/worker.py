"""One local task at a time, with independent lease renewal and P2 recovery."""
import threading
from uuid import uuid4

from dante.contracts import TaskStatus
from dante.continuity import ContinuitySignal, Disposition
from dante.recovery import ReconciliationRequired
from dante.task_queue import CancellationRequested, LeaseLost, TaskQueue, lease_context, positive
from dante.tool_broker import ToolDenied


class Worker:
    def __init__(self, ledger, host_factory, *, poll_s: float = 1, lease_s: float = 30):
        self.ledger, self.queue, self.host_factory = ledger, TaskQueue(ledger), host_factory
        self.poll_s, self.lease_s = positive(poll_s), positive(lease_s)
        self.worker_id = 'worker_' + uuid4().hex
        self.stop = threading.Event()

    def request_stop(self):
        self.stop.set()

    def _fail(self, task_id):
        task = self.ledger.get_task(task_id)
        if task.status in {TaskStatus.RUNNING, TaskStatus.WAITING_TOOL, TaskStatus.VERIFYING}:
            self.ledger.transition(task_id, TaskStatus.FAILED_RETRYABLE, current_step='worker-blocked')

    def _release_after_failure(self, lease, reason=None):
        try:
            if reason is not None:
                self._fail(lease.task_id)
            self.queue.release(lease, blocked=reason)
        except LeaseLost:
            self.stop.set()

    def run_once(self) -> bool:
        if self.stop.is_set():
            return False
        self.queue.recover_expired()
        lease = self.queue.claim(self.worker_id, self.lease_s)
        if lease is None:
            return False
        heartbeat_stop = threading.Event()

        def renew():
            while not heartbeat_stop.wait(self.lease_s / 3):
                try:
                    self.queue.heartbeat(lease, self.lease_s)
                except Exception:
                    self.stop.set()
                    return

        heartbeat = threading.Thread(target=renew, name='dante-lease', daemon=True)
        heartbeat.start()
        with lease_context(lease):
            try:
                self.queue.boundary(lease)
                task = self.ledger.get_task(lease.task_id)
                if task.status == TaskStatus.CREATED:
                    task = self.ledger.transition(task.task_id, TaskStatus.PLANNED)
                if task.status == TaskStatus.PLANNED:
                    task = self.ledger.transition(task.task_id, TaskStatus.RUNNING)
                # Inspect every prior effect before any new action, including those
                # beyond a checkpoint. Expiry is never evidence of non-execution.
                for step in self.ledger.steps(task.task_id):
                    if step.state != 'intent':
                        self.ledger.verified_step_result(task.task_id, step.step_id)
                host = self.host_factory(self.ledger, task)
                task = host.resume(task.task_id)
                plan = self.queue.plan(task.task_id)
                index = self.queue.status(task.task_id)['next_action']
                while index < len(plan.actions):
                    self.queue.boundary(lease)
                    if self.stop.is_set():
                        self.queue.release(lease)
                        return True
                    action = plan.actions[index]
                    if host.tools.manifest(action.tool_id).version != action.version:
                        raise ToolDenied('Tool version changed')
                    host.execute_tool_and_checkpoint(task.task_id, action.tool_id, action.arguments,
                        idempotency_key=f'worker-action:{index}', approval_id=action.approval_id, plan_hash=action.plan_hash)
                    self.queue.advance(lease, index)
                    index += 1
                self.queue.boundary(lease)
                if self.stop.is_set():
                    self.queue.release(lease)
                    return True
                finished = host.complete(task.task_id)
                self.queue.release(lease, blocked=None if finished.status == TaskStatus.COMPLETED else 'acceptance_incomplete')
            except ContinuitySignal as exc:
                try:
                    outcome = exc.outcome
                    if outcome.disposition == Disposition.RETRY_LATER:
                        self._fail(lease.task_id)
                        self.queue.release(lease, blocked=outcome.reason, retry_at=outcome.next_attempt_at)
                    elif outcome.disposition == Disposition.TERMINAL:
                        current = self.ledger.get_task(lease.task_id)
                        if current.status == TaskStatus.FAILED_RETRYABLE:
                            self.ledger.transition(lease.task_id, TaskStatus.RUNNING)
                        self.ledger.transition(lease.task_id, TaskStatus.FAILED_TERMINAL,
                                               current_step=outcome.reason)
                        self.queue.release(lease)
                    else:
                        self._release_after_failure(lease, 'invalid_continuity_signal')
                except LeaseLost:
                    self.stop.set()
            except CancellationRequested:
                self._release_after_failure(lease)
            except ReconciliationRequired:
                self._release_after_failure(lease, 'reconciliation_required')
            except ToolDenied as exc:
                self._release_after_failure(lease, exc.result.status.value)
            except LeaseLost:
                self.stop.set()  # The new owner alone may persist continuation.
            except Exception:
                self._release_after_failure(lease, 'worker_failure')
            finally:
                heartbeat_stop.set()
                heartbeat.join()
        return True

    def run(self):
        while not self.stop.is_set():
            if not self.run_once():
                self.stop.wait(self.poll_s)
