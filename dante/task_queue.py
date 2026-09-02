"""Persistent scheduling and fenced ownership; no execution or provider calls."""
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import math
from typing import Any
import time

from pydantic import Field
from dante.acceptance import AcceptanceContract
from dante.contracts import StrictModel, Task, TaskStatus, PrivacyClass
from dante.privacy import PrivacyGate
from dante.recovery import digest


class LeaseLost(RuntimeError):
    pass


class CancellationRequested(RuntimeError):
    pass


class Action(StrictModel):
    tool_id: str = Field(min_length=1)
    version: str = '1'
    arguments: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None
    plan_hash: str = ''


class ExecutionPlan(StrictModel):
    actions: list[Action] = Field(min_length=1, max_length=1000)
    acceptance: AcceptanceContract = Field(default_factory=AcceptanceContract)


@dataclass(frozen=True)
class Lease:
    task_id: str
    worker_id: str
    generation: int


_owner: ContextVar[Lease | None] = ContextVar('dante_worker_lease', default=None)
TERMINAL = {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED_TERMINAL}


@contextmanager
def lease_context(lease: Lease):
    token = _owner.set(lease)
    try:
        yield
    finally:
        _owner.reset(token)


def assert_lease(connection, task_id: str, *, before_effect=False, lease: Lease | None = None):
    row = connection.execute('SELECT * FROM task_queue WHERE task_id=?', (task_id,)).fetchone()
    if row is None:
        return
    owner = lease or _owner.get()
    if (owner is None or owner.task_id != task_id or row['state'] != 'leased'
            or row['worker_id'] != owner.worker_id or row['generation'] != owner.generation
            or row['lease_expires_at'] <= time.time()):
        raise LeaseLost('Worker no longer owns this task')
    if before_effect and row['cancel_requested']:
        raise CancellationRequested('Cancellation requested')
    return row


def positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Interval must be finite and positive')
    return value


class TaskQueue:
    def __init__(self, ledger):
        self.ledger = ledger

    def submit(self, task: Task, plan: ExecutionPlan) -> Task:
        # Plans must survive the client, but never accept credential-bearing payloads.
        plan = ExecutionPlan.model_validate_json(plan.model_dump_json())
        digest(plan.model_dump(mode='json'))
        def sensitive(value):
            if isinstance(value, dict):
                return any(k.lower() in {'password', 'secret', 'api_key', 'token', 'access_token'} or sensitive(v)
                           for k, v in value.items())
            return isinstance(value, list) and any(sensitive(v) for v in value)
        if (task.status != TaskStatus.CREATED or sensitive(plan.model_dump(mode='json'))
                or PrivacyGate().classify(task.goal + plan.model_dump_json(), task.privacy_class).classification == PrivacyClass.SECRET):
            raise ValueError('Only new tasks with non-secret local plans can be queued')
        acceptance = plan.acceptance.model_copy(update={
            'required_tools': plan.acceptance.required_tools | {a.tool_id for a in plan.actions}})
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO tasks(task_id,payload) VALUES(?,?)', (task.task_id, task.model_dump_json()))
            db.execute('INSERT INTO task_acceptance(task_id,contract,strict,locked) VALUES(?,?,1,1)',
                       (task.task_id, acceptance.model_dump_json()))
            db.execute('INSERT INTO task_queue(task_id,plan) VALUES(?,?)', (task.task_id, plan.model_dump_json()))
            self.ledger._event(db, task, 'task.created', None, {})
            self.ledger._event(db, task, 'worker.submitted', None, {'action_count': len(plan.actions)})
        return task

    def status(self, task_id: str) -> dict:
        with closing(self.ledger._connect()) as db:
            row = db.execute('SELECT * FROM task_queue WHERE task_id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        result = dict(row)
        result.pop('plan')
        result['task_status'] = self.ledger.get_task(task_id).status.value
        return result

    def list(self, *, runnable=False) -> list[dict]:
        with closing(self.ledger._connect()) as db:
            ids = [r[0] for r in db.execute('SELECT task_id FROM task_queue ORDER BY rowid')]
        rows = [self.status(task_id) for task_id in ids]
        now = time.time()
        return [r for r in rows if not runnable or (not r['cancel_requested'] and
                ((r['state'] == 'ready' and r['next_attempt_at'] <= now) or
                 (r['state'] == 'leased' and r['lease_expires_at'] <= now)))]

    def plan(self, task_id: str) -> ExecutionPlan:
        with closing(self.ledger._connect()) as db:
            row = db.execute('SELECT plan FROM task_queue WHERE task_id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return ExecutionPlan.model_validate_json(row[0])

    def _event(self, db, task_id, event, metadata):
        row = db.execute('SELECT payload FROM tasks WHERE task_id=?', (task_id,)).fetchone()
        task = Task.model_validate_json(row[0])
        self.ledger._event(db, task, event, task.status, metadata)

    def recover_expired(self) -> int:
        """Expiry revokes ownership, not proof that an old Python handler stopped."""
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT * FROM task_queue WHERE state='leased' AND lease_expires_at<=?", (time.time(),)).fetchall()
            for row in rows:
                db.execute("UPDATE execution_steps SET state='uncertain',error='lease_expired' WHERE task_id=? AND state='started'",
                           (row['task_id'],))
                task = Task.model_validate_json(db.execute('SELECT payload FROM tasks WHERE task_id=?', (row['task_id'],)).fetchone()[0])
                if row['cancel_requested'] and task.status not in TERMINAL:
                    self._cancel_task(db, task)
                state = 'done' if task.status in TERMINAL else 'ready'
                db.execute('UPDATE task_queue SET state=?,lease_expires_at=NULL WHERE task_id=?', (state, row['task_id']))
                db.execute('''UPDATE task_queue SET next_attempt_at=MAX(next_attempt_at,
                    COALESCE((SELECT next_attempt_at FROM task_continuity WHERE task_id=?),0))
                    WHERE task_id=?''', (row['task_id'], row['task_id']))
                self._event(db, row['task_id'], 'worker.recovered',
                            {'worker_id': row['worker_id'], 'generation': row['generation'], 'reason': 'lease_expired'})
            return len(rows)

    def claim(self, worker_id: str, lease_s: float) -> Lease | None:
        positive(lease_s)
        if not worker_id.strip():
            raise ValueError('Worker identity required')
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            now = time.time()
            row = db.execute("""SELECT q.* FROM task_queue q JOIN tasks t USING(task_id)
                WHERE q.state='ready' AND q.cancel_requested=0 AND q.next_attempt_at<=?
                AND COALESCE((SELECT next_attempt_at FROM task_continuity c WHERE c.task_id=q.task_id),0)<=?
                AND json_extract(t.payload,'$.status') NOT IN ('completed','cancelled','failed_terminal')
                ORDER BY q.next_attempt_at,q.rowid LIMIT 1""", (now, now)).fetchone()
            if row is None:
                return None
            lease = Lease(row['task_id'], worker_id, row['generation'] + 1)
            db.execute("""UPDATE task_queue SET state='leased',worker_id=?,generation=?,claimed_at=?,
                lease_expires_at=?,heartbeat_at=?,attempt=attempt+1 WHERE task_id=?""",
                (worker_id, lease.generation, now, now + lease_s, now, lease.task_id))
            self._event(db, lease.task_id, 'worker.claimed', {'worker_id': worker_id, 'generation': lease.generation})
            return lease

    def heartbeat(self, lease: Lease, lease_s: float):
        positive(lease_s)
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            assert_lease(db, lease.task_id, lease=lease)
            now = time.time()
            db.execute('UPDATE task_queue SET heartbeat_at=?,lease_expires_at=? WHERE task_id=?',
                       (now, now + lease_s, lease.task_id))
            self._event(db, lease.task_id, 'worker.heartbeat', {'worker_id': lease.worker_id, 'generation': lease.generation})

    def boundary(self, lease: Lease):
        with closing(self.ledger._connect()) as db:
            assert_lease(db, lease.task_id, before_effect=True, lease=lease)

    def advance(self, lease: Lease, index: int):
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = assert_lease(db, lease.task_id, lease=lease)
            if row['next_action'] != index:
                raise ValueError('Unexpected action checkpoint')
            db.execute('UPDATE task_queue SET next_action=? WHERE task_id=?', (index + 1, lease.task_id))
            self._event(db, lease.task_id, 'worker.checkpoint', {'generation': lease.generation, 'next_action': index + 1})

    def release(self, lease: Lease, *, blocked: str | None = None, retry_at: float | None = None):
        if retry_at is not None and (not math.isfinite(retry_at) or retry_at <= 0):
            raise ValueError('Invalid retry timestamp')
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = assert_lease(db, lease.task_id, lease=lease)
            task = Task.model_validate_json(db.execute('SELECT payload FROM tasks WHERE task_id=?', (lease.task_id,)).fetchone()[0])
            if row['cancel_requested'] and task.status not in TERMINAL:
                self._cancel_task(db, task)
            state = 'done' if task.status in TERMINAL else ('blocked' if blocked and retry_at is None else 'ready')
            db.execute('UPDATE task_queue SET state=?,lease_expires_at=NULL,last_failure=? WHERE task_id=?',
                       (state, blocked, lease.task_id))
            if retry_at is not None and state == 'ready':
                db.execute('UPDATE task_queue SET next_attempt_at=?,retry_count=retry_count+1 WHERE task_id=?',
                           (retry_at, lease.task_id))
                self._event(db, lease.task_id, 'worker.retry_scheduled',
                            {'next_attempt_at': retry_at, 'reason': blocked})
            self._event(db, lease.task_id, 'worker.released',
                        {'worker_id': lease.worker_id, 'generation': lease.generation, 'state': state, 'reason': blocked})

    def _cancel_task(self, db, task):
        from dante.ledger import TRANSITIONS
        if TaskStatus.CANCELLED not in TRANSITIONS[task.status]:
            raise ValueError('Task cannot be cancelled')
        previous = task.status
        task.status = TaskStatus.CANCELLED
        from dante.contracts import utc_now
        task.updated_at = utc_now()
        db.execute('UPDATE tasks SET payload=?,revision=revision+1 WHERE task_id=?', (task.model_dump_json(), task.task_id))
        self.ledger._event(db, task, 'task.transition', previous, {'reason': 'cancel_requested'})

    def cancel(self, task_id: str):
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM task_queue WHERE task_id=?', (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = Task.model_validate_json(db.execute('SELECT payload FROM tasks WHERE task_id=?', (task_id,)).fetchone()[0])
            if task.status in TERMINAL:
                return
            db.execute('UPDATE task_queue SET cancel_requested=1 WHERE task_id=?', (task_id,))
            if row['state'] != 'leased' or row['lease_expires_at'] <= time.time():
                self._cancel_task(db, task)
                db.execute("UPDATE task_queue SET state='done',lease_expires_at=NULL WHERE task_id=?", (task_id,))
            self._event(db, task_id, 'worker.cancel_requested', {})

    def retry(self, task_id: str, *, delay_s: float = 1):
        positive(delay_s)
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM task_queue WHERE task_id=?', (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = Task.model_validate_json(db.execute('SELECT payload FROM tasks WHERE task_id=?', (task_id,)).fetchone()[0])
            if row['cancel_requested'] or task.status in TERMINAL or row['state'] == 'leased':
                raise ValueError('Task is not available for retry')
            db.execute("UPDATE task_queue SET state='ready',retry_count=retry_count+1,next_attempt_at=? WHERE task_id=?",
                       (time.time() + delay_s, task_id))
            self._event(db, task_id, 'worker.retry_scheduled', {'delay_s': delay_s, 'retry_count': row['retry_count'] + 1})
