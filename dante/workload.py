"""Durable, bounded local inference jobs over the qualified Node 0 graph."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, NewType
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from dante.contracts import StrictModel
from dante.inference import InferenceCancelled, InferenceError, InferenceTimeout
from dante.ledger import TaskLedger


JobId = NewType('JobId', str)
AttemptId = NewType('AttemptId', str)
MAX_RESULT_BYTES = 64 * 1024


class JobState(StrEnum):
    QUEUED = 'queued'
    CLAIMED = 'claimed'
    RUNNING = 'running'
    CANCEL_REQUESTED = 'cancel_requested'
    RETRY_WAIT = 'retry_wait'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    TIMED_OUT = 'timed_out'


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.TIMED_OUT})
LEGAL_TRANSITIONS = {
    JobState.QUEUED: frozenset({JobState.CLAIMED, JobState.CANCELLED, JobState.TIMED_OUT}),
    JobState.CLAIMED: frozenset({JobState.RUNNING, JobState.CANCEL_REQUESTED, JobState.RETRY_WAIT,
                                 JobState.FAILED, JobState.CANCELLED, JobState.TIMED_OUT}),
    JobState.RUNNING: frozenset({JobState.CANCEL_REQUESTED, JobState.RETRY_WAIT, JobState.SUCCEEDED,
                                 JobState.FAILED, JobState.CANCELLED, JobState.TIMED_OUT}),
    JobState.CANCEL_REQUESTED: frozenset({JobState.CANCELLED, JobState.TIMED_OUT}),
    JobState.RETRY_WAIT: frozenset({JobState.CLAIMED, JobState.CANCELLED, JobState.TIMED_OUT}),
    JobState.SUCCEEDED: frozenset(), JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(), JobState.TIMED_OUT: frozenset(),
}


class JobTransitionError(RuntimeError):
    pass


class IdempotencyConflict(ValueError):
    pass


class OrchestratorAlreadyRunning(RuntimeError):
    pass


class Node0Unavailable(RuntimeError):
    pass


class QualificationRejected(RuntimeError):
    pass


class WorkloadExecutionTimeout(TimeoutError):
    pass


class ModelRequirement(StrictModel):
    model_id: str = Field(min_length=1, max_length=200)
    runtime_reference: str = Field(min_length=1, max_length=200)
    digest_sha256: str = Field(pattern=r'^[0-9a-fA-F]{64}$')
    context_tokens: int = Field(gt=0, le=131072)
    local_only: bool = True

    @field_validator('digest_sha256')
    @classmethod
    def normalize_digest(cls, value: str) -> str:
        return value.lower().removeprefix('sha256:')

    @field_validator('local_only')
    @classmethod
    def require_local(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError('Node 0 workload jobs require LOCAL_ONLY execution')
        return value


class WorkloadSpec(StrictModel):
    job_type: str = Field(default='LOCAL_INFERENCE', pattern=r'^LOCAL_INFERENCE$')
    model: ModelRequirement
    prompt: str = Field(min_length=1, max_length=20000)
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    max_output_tokens: int = Field(default=128, ge=1, le=2048)
    priority: int = Field(default=0, ge=-100, le=100)
    timeout_s: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)
    deadline_at: datetime | None = None
    maximum_attempts: int = Field(default=2, ge=1, le=5)
    retry_base_s: float = Field(default=2, ge=0.1, le=300, allow_inf_nan=False)
    retry_max_s: float = Field(default=30, ge=0.1, le=900, allow_inf_nan=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator('deadline_at')
    @classmethod
    def deadline_is_aware(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError('Deadline must include a timezone')
        return value.astimezone(timezone.utc) if value else value

    @model_validator(mode='after')
    def safe_metadata_and_retry(self):
        if self.retry_base_s > self.retry_max_s:
            raise ValueError('Retry base exceeds retry maximum')
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if any(word in str(key).lower() for word in ('password', 'secret', 'token', 'credential', 'api_key')):
                        raise ValueError('Sensitive metadata is not accepted')
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(self.metadata)
        json.dumps(self.metadata, allow_nan=False)
        return self


class JobRecord(StrictModel):
    job_id: JobId
    state: JobState
    priority: int
    created_at: datetime
    updated_at: datetime
    attempt_count: int
    current_attempt_id: AttemptId | None = None
    deadline_at: datetime | None = None
    cancel_requested: bool = False
    failure: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class AttemptRecord(StrictModel):
    attempt_id: AttemptId
    job_id: JobId
    attempt_number: int
    worker_id: str
    state: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    failure: dict[str, Any] | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError('UTC-aware timestamp required')
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _canonical_digest(spec: WorkloadSpec) -> str:
    raw = json.dumps(spec.model_dump(mode='json'), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class WorkloadStore:
    """Transactional workload state in the repository's versioned SQLite ledger."""
    def __init__(self, path: Path, *, audit=None, busy_timeout_ms: int = 5000):
        if not 1 <= busy_timeout_ms <= 10000:
            raise ValueError('SQLite busy timeout must be between 1 ms and 10 s')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Reuse the core ledger initializer and migration chain; no parallel schema owner.
        TaskLedger(self.path, audit)
        self.audit = audit
        self.busy_timeout_ms = busy_timeout_ms
        with closing(self._connect()) as db:
            db.execute('PRAGMA journal_mode=WAL')

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute(f'PRAGMA busy_timeout={self.busy_timeout_ms}')
        return db

    def _event(self, db, job_id: str, event: str, *, attempt_id: str | None = None, **metadata):
        safe = {k: v for k, v in metadata.items() if k not in {'prompt', 'response', 'result', 'content'}}
        db.execute('INSERT INTO workload_events(job_id,attempt_id,event,timestamp_utc,metadata) VALUES(?,?,?,?,?)',
                   (job_id, attempt_id, event, _iso(_now()), json.dumps(safe, sort_keys=True, separators=(',', ':'))))
        if self.audit:
            self.audit.write('workload.' + event, job_id=job_id, attempt_id=attempt_id, **safe)

    @staticmethod
    def _transition(db, row, state: JobState, *, failure=None, result=None, eligible_at=None):
        previous = JobState(row['state'])
        if state not in LEGAL_TRANSITIONS[previous]:
            raise JobTransitionError(f'Illegal workload transition {previous.value}->{state.value}')
        now = _now()
        db.execute('''UPDATE workload_jobs SET state=?,updated_at=?,eligible_at=?,failure_json=?,result_json=?
            WHERE job_id=?''', (state.value, _iso(now), eligible_at if eligible_at is not None else row['eligible_at'],
                                 json.dumps(failure, sort_keys=True) if failure is not None else None,
                                 json.dumps(result, sort_keys=True, ensure_ascii=False) if result is not None else None,
                                 row['job_id']))
        return previous

    def submit(self, spec: WorkloadSpec, *, idempotency_key: str | None = None) -> JobRecord:
        if idempotency_key is not None and (not idempotency_key.strip() or len(idempotency_key) > 200):
            raise ValueError('Invalid idempotency key')
        canonical = _canonical_digest(spec)
        created = _now()
        deadline_epoch = spec.deadline_at.timestamp() if spec.deadline_at else None
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if idempotency_key:
                old = db.execute('SELECT job_id,request_digest FROM workload_jobs WHERE idempotency_key=?',
                                  (idempotency_key,)).fetchone()
                if old:
                    if old['request_digest'] != canonical:
                        raise IdempotencyConflict('Idempotency key already identifies a different request')
                    return self.get(old['job_id'])
            job_id = 'job_' + uuid4().hex
            db.execute('''INSERT INTO workload_jobs(job_id,idempotency_key,request_digest,payload,state,priority,
                created_at,created_epoch,updated_at,eligible_at,deadline_epoch,attempt_count,cancel_requested)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)''',
                (job_id, idempotency_key, canonical, spec.model_dump_json(), JobState.QUEUED.value,
                 spec.priority, _iso(created), created.timestamp(), _iso(created), created.timestamp(), deadline_epoch, 0))
            self._event(db, job_id, 'submitted', request_digest=canonical, job_type=spec.job_type,
                        priority=spec.priority, maximum_attempts=spec.maximum_attempts)
        return self.get(job_id)

    def spec(self, job_id: str) -> WorkloadSpec:
        with closing(self._connect()) as db:
            row = db.execute('SELECT payload FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return WorkloadSpec.model_validate_json(row['payload'])

    def get(self, job_id: str) -> JobRecord:
        with closing(self._connect()) as db:
            row = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return JobRecord(job_id=JobId(row['job_id']), state=JobState(row['state']), priority=row['priority'],
            created_at=_parse(row['created_at']), updated_at=_parse(row['updated_at']),
            attempt_count=row['attempt_count'], current_attempt_id=AttemptId(row['current_attempt_id']) if row['current_attempt_id'] else None,
            deadline_at=datetime.fromtimestamp(row['deadline_epoch'], timezone.utc) if row['deadline_epoch'] else None,
            cancel_requested=bool(row['cancel_requested']),
            failure=json.loads(row['failure_json']) if row['failure_json'] else None,
            result=json.loads(row['result_json']) if row['result_json'] else None)

    def list(self, *, limit: int = 100) -> list[JobRecord]:
        if not 1 <= limit <= 500:
            raise ValueError('Limit must be between 1 and 500')
        with closing(self._connect()) as db:
            ids = [r[0] for r in db.execute('SELECT job_id FROM workload_jobs ORDER BY created_epoch DESC,job_id LIMIT ?', (limit,))]
        return [self.get(job_id) for job_id in ids]

    def attempts(self, job_id: str) -> list[AttemptRecord]:
        with closing(self._connect()) as db:
            rows = db.execute('SELECT * FROM workload_attempts WHERE job_id=? ORDER BY attempt_number', (job_id,)).fetchall()
        return [AttemptRecord(attempt_id=AttemptId(r['attempt_id']), job_id=JobId(r['job_id']),
            attempt_number=r['attempt_number'], worker_id=r['worker_id'], state=r['state'],
            started_at=_parse(r['started_at']), ended_at=_parse(r['ended_at']),
            failure=json.loads(r['failure_json']) if r['failure_json'] else None) for r in rows]

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as db:
            rows = db.execute('SELECT * FROM workload_events WHERE job_id=? ORDER BY event_id', (job_id,)).fetchall()
        return [{'event_id': r['event_id'], 'event': r['event'], 'attempt_id': r['attempt_id'],
                 'timestamp_utc': r['timestamp_utc'], 'metadata': json.loads(r['metadata'])} for r in rows]

    def record_event(self, job_id: str, event: str, *, attempt_id: str | None = None, **metadata):
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone():
                raise KeyError(job_id)
            self._event(db, job_id, event, attempt_id=attempt_id, **metadata)

    def _expire_due(self, db, row, now: float):
        if row['deadline_epoch'] is not None and row['deadline_epoch'] <= now and row['state'] in {
                JobState.QUEUED.value, JobState.RETRY_WAIT.value, JobState.CLAIMED.value}:
            previous = self._transition(db, row, JobState.TIMED_OUT, failure={'code': 'deadline_expired'})
            self._event(db, row['job_id'], 'timed_out', from_state=previous.value, reason='deadline_expired')
            return True
        return False

    def claim(self, worker_id: str) -> tuple[JobRecord, AttemptId] | None:
        if not worker_id or len(worker_id) > 128:
            raise ValueError('Worker identity required')
        now = time.time()
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            expired = db.execute("SELECT * FROM workload_jobs WHERE state IN ('queued','retry_wait') AND deadline_epoch IS NOT NULL AND deadline_epoch<=?", (now,)).fetchall()
            for row in expired:
                self._expire_due(db, row, now)
            row = db.execute("""SELECT * FROM workload_jobs WHERE state IN ('queued','retry_wait')
                AND eligible_at<=? AND (deadline_epoch IS NULL OR deadline_epoch>?)
                ORDER BY priority DESC,eligible_at,created_epoch,job_id LIMIT 1""", (now, now)).fetchone()
            if not row:
                return None
            previous = self._transition(db, row, JobState.CLAIMED)
            number = row['attempt_count'] + 1
            attempt_id = 'att_' + uuid4().hex
            db.execute('''INSERT INTO workload_attempts(attempt_id,job_id,attempt_number,worker_id,state)
                VALUES(?,?,?,?,?)''', (attempt_id, row['job_id'], number, worker_id, 'claimed'))
            db.execute('UPDATE workload_jobs SET attempt_count=?,current_attempt_id=? WHERE job_id=?',
                       (number, attempt_id, row['job_id']))
            self._event(db, row['job_id'], 'claimed', attempt_id=attempt_id, worker_id=worker_id,
                        attempt_number=number, from_state=previous.value)
        return self.get(row['job_id']), AttemptId(attempt_id)

    def start_attempt(self, job_id: str, attempt_id: str) -> bool:
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            if bool(row['cancel_requested']) or row['state'] == JobState.CANCEL_REQUESTED.value:
                self._finalize_cancel(db, row, attempt_id)
                return False
            if row['current_attempt_id'] != attempt_id or row['state'] != JobState.CLAIMED.value:
                raise JobTransitionError('Attempt no longer owns the claimed job')
            previous = self._transition(db, row, JobState.RUNNING)
            stamp = _iso(_now())
            db.execute("UPDATE workload_attempts SET state='running',started_at=? WHERE attempt_id=?", (stamp, attempt_id))
            self._event(db, job_id, 'attempt_started', attempt_id=attempt_id, from_state=previous.value)
            return True

    def is_cancel_requested(self, job_id: str, attempt_id: str) -> bool:
        with closing(self._connect()) as db:
            row = db.execute('SELECT cancel_requested,current_attempt_id FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
        return bool(row and row['cancel_requested'] and row['current_attempt_id'] == attempt_id)

    def request_cancel(self, job_id: str) -> JobRecord:
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            state = JobState(row['state'])
            if state in TERMINAL_STATES:
                return self.get(job_id)
            db.execute('UPDATE workload_jobs SET cancel_requested=1,updated_at=? WHERE job_id=?', (_iso(_now()), job_id))
            if state in {JobState.QUEUED, JobState.RETRY_WAIT, JobState.CLAIMED}:
                self._finalize_cancel(db, row, row['current_attempt_id'])
            elif state == JobState.RUNNING:
                self._transition(db, row, JobState.CANCEL_REQUESTED)
                self._event(db, job_id, 'cancel_requested', attempt_id=row['current_attempt_id'])
            elif state != JobState.CANCEL_REQUESTED:
                raise JobTransitionError('Job cannot be cancelled from its current state')
        return self.get(job_id)

    def finish_cancel(self, job_id: str, attempt_id: str) -> JobRecord:
        """Durably finish a requested cancellation after execution has observed it."""
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            if row['current_attempt_id'] != attempt_id:
                raise JobTransitionError('Attempt no longer owns the job')
            if row['state'] == JobState.CANCEL_REQUESTED.value or row['cancel_requested']:
                self._finalize_cancel(db, row, attempt_id)
            elif row['state'] != JobState.CANCELLED.value:
                raise JobTransitionError('Cancellation was not requested')
        return self.get(job_id)

    def _finalize_cancel(self, db, row, attempt_id):
        if row['state'] != JobState.CANCELLED.value:
            previous = self._transition(db, row, JobState.CANCELLED, failure={'code': 'cancelled'})
            self._event(db, row['job_id'], 'cancelled', attempt_id=attempt_id, from_state=previous.value)
        if attempt_id:
            db.execute("UPDATE workload_attempts SET state='cancelled',ended_at=? WHERE attempt_id=? AND ended_at IS NULL",
                       (_iso(_now()), attempt_id))

    def succeed(self, job_id: str, attempt_id: str, result: dict[str, Any]) -> JobRecord:
        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if len(rendered.encode('utf-8')) > MAX_RESULT_BYTES:
            raise ValueError('Persisted workload result exceeds 64 KiB')
        response_digest = hashlib.sha256(rendered.encode('utf-8')).hexdigest()
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
            if row['current_attempt_id'] != attempt_id or row['state'] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}:
                raise JobTransitionError('Attempt no longer owns the running job')
            if row['cancel_requested'] or row['state'] == JobState.CANCEL_REQUESTED.value:
                self._finalize_cancel(db, row, attempt_id)
            else:
                previous = self._transition(db, row, JobState.SUCCEEDED, result=result)
                db.execute("UPDATE workload_attempts SET state='succeeded',ended_at=?,response_digest=? WHERE attempt_id=?",
                           (_iso(_now()), response_digest, attempt_id))
                self._event(db, job_id, 'succeeded', attempt_id=attempt_id, from_state=previous.value,
                            result_bytes=len(rendered.encode('utf-8')), result_digest=response_digest)
        return self.get(job_id)

    def fail(self, job_id: str, attempt_id: str, *, code: str, retryable: bool) -> JobRecord:
        if not code or len(code) > 100:
            raise ValueError('A typed failure code is required')
        now = time.time()
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            if row['current_attempt_id'] != attempt_id or row['state'] not in {
                    JobState.RUNNING.value, JobState.CLAIMED.value, JobState.CANCEL_REQUESTED.value}:
                raise JobTransitionError('Attempt no longer owns the job')
            if row['cancel_requested'] or row['state'] == JobState.CANCEL_REQUESTED.value:
                self._finalize_cancel(db, row, attempt_id)
                return self.get(job_id)
            spec_row = db.execute('SELECT payload FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
            spec = WorkloadSpec.model_validate_json(spec_row['payload'])
            attempt = db.execute('SELECT attempt_number FROM workload_attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
            attempt_state = db.execute('SELECT state FROM workload_attempts WHERE attempt_id=?', (attempt_id,)).fetchone()['state']
            expired = row['deadline_epoch'] is not None and row['deadline_epoch'] <= now
            can_retry = retryable and not expired and attempt['attempt_number'] < spec.maximum_attempts
            if can_retry:
                delay = min(spec.retry_max_s, spec.retry_base_s * (2 ** (attempt['attempt_number'] - 1)))
                eligible = now + delay
                if row['deadline_epoch'] is not None and eligible >= row['deadline_epoch']:
                    can_retry = False
            failure = {'code': code, 'retryable': bool(retryable)}
            if attempt_state == 'interrupted':
                db.execute('UPDATE workload_attempts SET failure_json=? WHERE attempt_id=?',
                           (json.dumps(failure, sort_keys=True), attempt_id))
            else:
                db.execute("UPDATE workload_attempts SET state='failed',ended_at=?,failure_json=? WHERE attempt_id=?",
                           (_iso(_now()), json.dumps(failure, sort_keys=True), attempt_id))
            if expired:
                target = JobState.TIMED_OUT
            elif can_retry:
                target = JobState.RETRY_WAIT
            else:
                target = JobState.FAILED
            previous = self._transition(db, row, target, failure=failure,
                eligible_at=eligible if target == JobState.RETRY_WAIT else None)
            event = 'retry_scheduled' if target == JobState.RETRY_WAIT else target.value
            self._event(db, job_id, event, attempt_id=attempt_id, from_state=previous.value,
                        failure_code=code, retryable=bool(retryable), next_eligible_epoch=eligible if target == JobState.RETRY_WAIT else None)
        return self.get(job_id)

    def mark_timeout(self, job_id: str, attempt_id: str) -> JobRecord:
        return self.fail(job_id, attempt_id, code='attempt_timeout', retryable=True)

    def acquire_owner(self, owner_id: str, process_id: int) -> None:
        now = time.time()
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT * FROM workload_control WHERE singleton=1').fetchone()
            if current and current['owner_id'] != owner_id and self._process_alive(current['process_id']):
                raise OrchestratorAlreadyRunning('Another workload orchestrator owns this database')
            db.execute('''INSERT INTO workload_control(singleton,owner_id,process_id,heartbeat_epoch)
                VALUES(1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET owner_id=excluded.owner_id,
                process_id=excluded.process_id,heartbeat_epoch=excluded.heartbeat_epoch''',
                (owner_id, process_id, now))

    @staticmethod
    def _process_alive(pid: int) -> bool:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes

            if pid <= 0:
                return False

            # On Windows CPython implements os.kill(pid, 0) via TerminateProcess.
            # A zero-time wait on a synchronization handle checks liveness safely.
            synchronize = 0x00100000
            error_invalid_parameter = 87
            wait_object_0 = 0
            wait_timeout = 258
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            open_process.restype = wintypes.HANDLE
            wait_for_single_object = kernel32.WaitForSingleObject
            wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            wait_for_single_object.restype = wintypes.DWORD
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL

            handle = open_process(synchronize, False, pid)
            if not handle:
                return ctypes.get_last_error() != error_invalid_parameter
            try:
                result = wait_for_single_object(handle, 0)
                if result == wait_object_0:
                    return False
                if result == wait_timeout:
                    return True
                return True
            finally:
                close_handle(handle)

        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def heartbeat_owner(self, owner_id: str) -> None:
        with closing(self._connect()) as db, db:
            cur = db.execute('UPDATE workload_control SET heartbeat_epoch=? WHERE singleton=1 AND owner_id=?',
                             (time.time(), owner_id))
            if cur.rowcount != 1:
                raise OrchestratorAlreadyRunning('Workload ownership was lost')

    def release_owner(self, owner_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute('DELETE FROM workload_control WHERE singleton=1 AND owner_id=?', (owner_id,))

    def recover_interrupted(self) -> list[str]:
        """New process owner deterministically resolves work left mid-attempt."""
        with closing(self._connect()) as db:
            rows = db.execute("SELECT job_id,current_attempt_id,state FROM workload_jobs WHERE state IN ('claimed','running','cancel_requested') ORDER BY created_epoch,job_id").fetchall()
        recovered = []
        for row in rows:
            job_id, attempt_id, state = row['job_id'], row['current_attempt_id'], row['state']
            if not attempt_id:
                continue
            if state == JobState.CANCEL_REQUESTED.value:
                with closing(self._connect()) as db, db:
                    db.execute('BEGIN IMMEDIATE')
                    current = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
                    self._finalize_cancel(db, current, attempt_id)
                    self._event(db, job_id, 'recovered_cancelled', attempt_id=attempt_id)
            else:
                with closing(self._connect()) as db, db:
                    db.execute('BEGIN IMMEDIATE')
                    current = db.execute('SELECT * FROM workload_jobs WHERE job_id=?', (job_id,)).fetchone()
                    previous = JobState(current['state'])
                    db.execute("UPDATE workload_attempts SET state='interrupted',ended_at=?,failure_json=? WHERE attempt_id=?",
                               (_iso(_now()), json.dumps({'code': 'orchestrator_interrupted'}), attempt_id))
                    # Generation has no external transactional side effect, so retry is safe but at-least-once.
                    self._event(db, job_id, 'attempt_interrupted', attempt_id=attempt_id, previous_state=previous.value)
                self.fail(job_id, attempt_id, code='orchestrator_interrupted', retryable=True)
            recovered.append(job_id)
        return recovered


@dataclass(frozen=True)
class ExecutionResult:
    text: str
    model_id: str
    provider_id: str
    fallback: bool
    runtime: str
    metadata: dict[str, Any]


class ResourceGate:
    """One explicit GPU slot by default for the current Node 0 hardware."""
    def __init__(self, max_gpu_jobs: int = 1):
        if not 1 <= max_gpu_jobs <= 4:
            raise ValueError('GPU concurrency must be between one and four')
        self.max_gpu_jobs = max_gpu_jobs
        self._semaphore = threading.BoundedSemaphore(max_gpu_jobs)

    def acquire(self, timeout: float | None = None) -> bool:
        return self._semaphore.acquire(timeout=timeout)

    def release(self):
        self._semaphore.release()


@dataclass
class _ActiveAttempt:
    job_id: str
    attempt_id: str
    cancellation: threading.Event = field(default_factory=threading.Event)
    finish_lock: threading.Lock = field(default_factory=threading.Lock)
    timed_out: bool = False
    worker: threading.Thread | None = None
    reaper: threading.Thread | None = None


class Node0InferenceExecutor:
    """Adapter to the already-running production supervisor and its one graph."""
    def __init__(self, supervisor, *, expected_model_id: str,
                 expected_runtime_reference: str, expected_digest: str, workspace: str | None = None):
        self.supervisor = supervisor
        self.expected_model_id = expected_model_id
        self.expected_runtime_reference = expected_runtime_reference
        self.expected_digest = expected_digest.lower().removeprefix('sha256:')
        self.workspace = workspace or str(Path.home())

    def execute(self, spec: WorkloadSpec, cancellation: threading.Event) -> ExecutionResult:
        snapshot = self.supervisor.operator_status()
        if not snapshot.get('ready_for_local_routing') or not snapshot.get('gate_accepted'):
            raise Node0Unavailable('Node 0 qualification gate is not ready')
        metadata = spec.model
        if (metadata.model_id != self.expected_model_id
                or metadata.runtime_reference != self.expected_runtime_reference
                or metadata.digest_sha256 != self.expected_digest):
            raise QualificationRejected('Requested model identity differs from configured qualified model')
        response = self.supervisor.infer(spec.prompt, model_reference=metadata.runtime_reference,
            context_tokens=metadata.context_tokens, max_output_tokens=spec.max_output_tokens,
            cancellation=cancellation)
        if cancellation.is_set():
            raise InferenceCancelled('Local inference cancelled')
        if response.fallback or response.provider_id != 'ollama' or response.model.model_id != metadata.model_id:
            raise QualificationRejected('Inference did not use the required qualified local route')
        actual = response.model.local_metadata
        if not actual or actual.runtime_reference != metadata.runtime_reference or actual.runtime_digest != metadata.digest_sha256:
            raise QualificationRejected('Inference model identity does not match the submitted requirement')
        if not response.content.strip():
            raise RuntimeError('local_runtime_empty_response')
        health = self.supervisor.operator_health()
        if not health.get('healthy') or not health.get('gpu_execution_verified'):
            raise QualificationRejected('Inference GPU execution could not be verified')
        return ExecutionResult(response.content, response.model.model_id, response.provider_id,
            response.fallback, response.model.runtime,
            {'context_tokens': metadata.context_tokens, 'max_output_tokens': spec.max_output_tokens,
             'qualification_id': health.get('qualification_id'), 'gpu_identity': health.get('gpu_uuid'),
             'gpu_vram_bytes': health.get('gpu_vram_bytes'),
             'runtime_version': health.get('runtime_version')})


class WorkloadOrchestrator:
    def __init__(self, store: WorkloadStore, executor, *, poll_s: float = 0.25,
                 max_gpu_jobs: int = 1, worker_id: str | None = None):
        if not 0.05 <= poll_s <= 30:
            raise ValueError('Scheduler poll interval must be between 50 ms and 30 s')
        self.store, self.executor = store, executor
        self.poll_s = poll_s
        self.worker_id = worker_id or 'orchestrator_' + uuid4().hex
        self.owner_id = self.worker_id
        self.resource_gate = ResourceGate(max_gpu_jobs)
        self.state = 'STOPPED'
        self.stop_event = threading.Event()
        self._active: dict[str, _ActiveAttempt] = {}
        self._active_lock = threading.Lock()
        self._owner_released = True

    def start(self) -> list[str]:
        if self.state == 'RUNNING':
            return []
        with self._active_lock:
            if self._active:
                raise OrchestratorAlreadyRunning('Cannot restart while an owned inference is still active')
        self.state = 'STARTING'
        self.store.acquire_owner(self.owner_id, os.getpid())
        self._owner_released = False
        try:
            recovered = self.store.recover_interrupted()
            self.stop_event.clear()
            self.state = 'RUNNING'
            self.store.audit and self.store.audit.write('workload.orchestrator.started', worker_id=self.worker_id,
                                                         recovered_count=len(recovered))
            return recovered
        except BaseException:
            self.store.release_owner(self.owner_id)
            self.state = 'FAILED'
            raise

    def cancel(self, job_id: str) -> JobRecord:
        result = self.store.request_cancel(job_id)
        with self._active_lock:
            active = self._active.get(job_id)
            if active is not None:
                active.cancellation.set()
        return result

    def _run_claimed(self, job: JobRecord, attempt_id: str):
        job_id, attempt_id = str(job.job_id), str(attempt_id)
        active = _ActiveAttempt(job_id, attempt_id)
        with self._active_lock:
            self._active[job_id] = active
        acquired = False
        try:
            acquired = self.resource_gate.acquire(timeout=0)
            if not acquired:
                # Single-slot scheduling should make this unreachable; fail closed if violated.
                self.store.fail(job_id, attempt_id, code='gpu_slot_unavailable', retryable=True)
                with self._active_lock:
                    self._active.pop(job_id, None)
                return
            if not self.store.start_attempt(job_id, attempt_id):
                self.resource_gate.release()
                acquired = False
                with self._active_lock:
                    self._active.pop(job_id, None)
                return
            spec = self.store.spec(job_id)
            if spec.deadline_at and _now() >= spec.deadline_at:
                self.store.fail(job_id, attempt_id, code='deadline_expired', retryable=False)
                self.resource_gate.release()
                acquired = False
                with self._active_lock:
                    self._active.pop(job_id, None)
                return
            timeout = spec.timeout_s
            if spec.deadline_at:
                timeout = min(timeout, max(0.001, (spec.deadline_at - _now()).total_seconds()))
            self.store.record_event(job_id, 'route_selected', attempt_id=attempt_id,
                route='local_only', provider='ollama', model_id=spec.model.model_id,
                model_digest=spec.model.digest_sha256)
            def execute_and_persist():
                try:
                    outcome = self.executor.execute(spec, active.cancellation)
                    if not isinstance(outcome.text, str) or len(outcome.text.encode('utf-8')) > MAX_RESULT_BYTES:
                        raise ValueError('result_too_large')
                    with active.finish_lock:
                        if active.timed_out:
                            return
                        if active.cancellation.is_set() or self.store.is_cancel_requested(job_id, attempt_id):
                            self.store.request_cancel(job_id)
                            self.store.finish_cancel(job_id, attempt_id)
                            return
                        self.store.succeed(job_id, attempt_id, {
                            'text': outcome.text, 'model_id': outcome.model_id, 'provider_id': outcome.provider_id,
                            'runtime': outcome.runtime, 'fallback': outcome.fallback, 'metadata': outcome.metadata,
                        })
                except InferenceCancelled:
                    with active.finish_lock:
                        if not active.timed_out:
                            self.store.request_cancel(job_id)
                            self.store.finish_cancel(job_id, attempt_id)
                except BaseException as exc:
                    with active.finish_lock:
                        if active.timed_out:
                            return
                        retryable = isinstance(exc, (InferenceTimeout, Node0Unavailable)) or (
                            isinstance(exc, InferenceError) and getattr(exc, 'retry_after', None) is not None)
                        code = 'result_too_large' if str(exc) == 'result_too_large' else (
                            type(exc).__name__ if isinstance(exc, (InferenceError, Node0Unavailable))
                            else 'local_execution_failed')
                        self.store.fail(job_id, attempt_id, code=code, retryable=retryable)

            def reap_worker():
                assert active.worker is not None
                active.worker.join()
                if acquired:
                    self.resource_gate.release()
                with self._active_lock:
                    self._active.pop(job_id, None)
                    no_active = not self._active
                if self.stop_event.is_set() and no_active:
                    self._finish_stop()

            active.worker = threading.Thread(target=execute_and_persist,
                name='dante-local-inference-' + attempt_id, daemon=False)
            active.reaper = threading.Thread(target=reap_worker,
                name='dante-inference-reaper-' + attempt_id, daemon=False)
            # Start execution first: joining an unstarted Thread raises RuntimeError.
            active.worker.start()
            # The reaper owns resource release even if execution outlives its deadline.
            active.reaper.start()
            deadline = time.monotonic() + timeout
            while active.worker.is_alive():
                if self.store.is_cancel_requested(job_id, attempt_id):
                    active.cancellation.set()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with active.finish_lock:
                        if active.worker.is_alive() and not active.timed_out:
                            active.timed_out = True
                            self.store.mark_timeout(job_id, attempt_id)
                            active.cancellation.set()
                    break
                active.worker.join(min(0.025, remaining))
            if not active.worker.is_alive() and active.reaper is not None:
                active.reaper.join()
            elif active.worker.is_alive():
                # Allow cooperative cancellation a short bounded opportunity to finish.
                active.worker.join(0.25)
                if not active.worker.is_alive() and active.reaper is not None:
                    active.reaper.join()
                else:
                    self.state = 'DEGRADED'
        except BaseException:
            if acquired and (active.worker is None or not active.worker.is_alive()):
                self.resource_gate.release()
            with self._active_lock:
                if active.worker is None or not active.worker.is_alive():
                    self._active.pop(job_id, None)
            raise

    def run_once(self) -> bool:
        if self.state != 'RUNNING':
            raise RuntimeError('Orchestrator must be started before dispatch')
        self.store.heartbeat_owner(self.owner_id)
        claimed = self.store.claim(self.worker_id)
        if claimed is None:
            return False
        job, attempt_id = claimed
        self._run_claimed(job, str(attempt_id))
        return True

    def serve(self):
        if self.state != 'RUNNING':
            self.start()
        try:
            while not self.stop_event.is_set():
                self.store.heartbeat_owner(self.owner_id)
                if not self.run_once():
                    self.stop_event.wait(self.poll_s)
        finally:
            self.stop()

    def request_stop(self):
        if self.state == 'RUNNING':
            self.state = 'DRAINING'
        self.stop_event.set()

    def stop(self, *, drain_timeout_s: float = 65):
        self.request_stop()
        deadline = time.monotonic() + drain_timeout_s
        while time.monotonic() < deadline:
            with self._active_lock:
                active = list(self._active.values())
            if not active:
                self._finish_stop()
                return
            time.sleep(min(0.025, max(0, deadline - time.monotonic())))
        with self._active_lock:
            active = list(self._active.values())
        if active:
            for attempt in active:
                self.store.request_cancel(attempt.job_id)
                attempt.cancellation.set()
            self.state = 'DEGRADED'
            self.store.audit and self.store.audit.write('workload.orchestrator.stopped', worker_id=self.worker_id,
                                                         state=self.state, active_count=len(active))
            # The reaper retains owner and GPU slot until every owned call actually exits.
            return
        self._finish_stop()

    def _finish_stop(self):
        with self._active_lock:
            if self._active:
                return
            release = not self._owner_released
            self._owner_released = True
            self.state = 'STOPPED'
        self.store.audit and self.store.audit.write('workload.orchestrator.stopped', worker_id=self.worker_id,
                                                     state=self.state)
        if release:
            self.store.release_owner(self.owner_id)


__all__ = [
    'AttemptId', 'AttemptRecord', 'ExecutionResult', 'IdempotencyConflict', 'JobId', 'JobRecord',
    'JobState', 'JobTransitionError', 'ModelRequirement', 'Node0InferenceExecutor',
    'Node0Unavailable', 'OrchestratorAlreadyRunning', 'QualificationRejected', 'ResourceGate',
    'WorkloadOrchestrator', 'WorkloadSpec', 'WorkloadStore',
]
