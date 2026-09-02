from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dante.contracts import Artifact, RouteDecision, Task, TaskStatus, new_id
from dante.acceptance import AcceptanceContract, AcceptanceResult, ObservedAcceptance, DeterministicAcceptanceVerifier
from dante.migrations import migrate
from dante.recovery import StepRecord, ReconciliationRequired, digest, file_digest, receipt, verified_receipt
from dante.telemetry import JsonlAudit


TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.PLANNED, TaskStatus.CANCELLED}),
    TaskStatus.PLANNED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.WAITING_TOOL, TaskStatus.WAITING_APPROVAL, TaskStatus.VERIFYING, TaskStatus.FAILED_RETRYABLE, TaskStatus.FAILED_TERMINAL, TaskStatus.CANCELLED}),
    TaskStatus.WAITING_TOOL: frozenset({TaskStatus.RUNNING, TaskStatus.FAILED_RETRYABLE, TaskStatus.FAILED_TERMINAL, TaskStatus.CANCELLED}),
    TaskStatus.WAITING_APPROVAL: frozenset({TaskStatus.RUNNING, TaskStatus.FAILED_TERMINAL, TaskStatus.CANCELLED}),
    TaskStatus.VERIFYING: frozenset({TaskStatus.CANCELLED, TaskStatus.COMPLETED, TaskStatus.FAILED_RETRYABLE, TaskStatus.FAILED_TERMINAL}),
    TaskStatus.FAILED_RETRYABLE: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED_TERMINAL: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTransition(RuntimeError):
    pass


class TaskLedger:
    def __init__(self, path: Path, audit: JsonlAudit | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.audit = audit
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, payload TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    event TEXT NOT NULL, from_status TEXT, to_status TEXT,
                    trace_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL, metadata TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE TABLE IF NOT EXISTS routes (
                    route_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, payload TEXT NOT NULL,
                    token_hash TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0
                );
                """
            )

        with closing(self._connect()) as connection:
            migrate(connection)

    @staticmethod
    def _dump(model: Any) -> str:
        return model.model_dump_json()

    def _event(self, connection: sqlite3.Connection, task: Task, event: str, previous: TaskStatus | None, metadata: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO task_events(task_id,event,from_status,to_status,trace_id,timestamp_utc,metadata) VALUES(?,?,?,?,?,?,?)",
            (task.task_id, event, previous.value if previous else None, task.status.value, task.trace_id,
             datetime.now(timezone.utc).isoformat(), json.dumps(metadata, separators=(",", ":"))),
        )

    def create_task(self, task: Task, acceptance: AcceptanceContract | None = None) -> Task:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO tasks(task_id,payload) VALUES(?,?)", (task.task_id, self._dump(task)))
            connection.execute("INSERT INTO task_acceptance(task_id,contract,strict,locked) VALUES(?,?,?,?)",
                               (task.task_id, (acceptance or AcceptanceContract()).model_dump_json(),
                                int(acceptance is not None), int(acceptance is not None)))
            self._event(connection, task, "task.created", None, {})
        if self.audit:
            self.audit.write("task.created", status=task.status.value)
        return task

    def get_task(self, task_id: str) -> Task:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT payload FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(task_id)
        return Task.model_validate_json(row["payload"])

    def transition(self, task_id: str, target: TaskStatus, *, current_step: str | None = None, metadata: dict[str, Any] | None = None) -> Task:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            from dante.task_queue import assert_lease
            assert_lease(connection, task_id, before_effect=False)
            if target == TaskStatus.COMPLETED:
                assert_lease(connection, task_id, before_effect=True)
            row = connection.execute("SELECT payload,revision FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError(task_id)
            task = Task.model_validate_json(row["payload"])
            previous = task.status
            if previous == target:
                return task
            if target not in TRANSITIONS[previous]:
                raise InvalidTransition(f"{previous.value} -> {target.value} is not allowed")
            if target == TaskStatus.COMPLETED:
                if not self._acceptance_result(connection, task_id).acceptance_complete:
                    raise InvalidTransition("Persistent acceptance or step evidence is incomplete")
            task.status = target
            task.updated_at = datetime.now(timezone.utc)
            if current_step is not None:
                task.current_step = current_step
            if target == TaskStatus.RUNNING:
                task.attempts += 1
            updated = connection.execute(
                "UPDATE tasks SET payload=?,revision=revision+1 WHERE task_id=? AND revision=?",
                (self._dump(task), task_id, row["revision"]),
            )
            if updated.rowcount != 1:
                raise RuntimeError("Concurrent task update rejected")
            self._event(connection, task, "task.transition", previous, metadata or {})
        if self.audit:
            self.audit.write("task.transition", from_status=previous.value, to_status=target.value)
        return task

    def checkpoint(self, task_id: str, step: str, state: dict[str, Any]) -> Task:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            from dante.task_queue import assert_lease
            assert_lease(connection, task_id, before_effect=False)
            row = connection.execute("SELECT payload,revision FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError(task_id)
            task = Task.model_validate_json(row["payload"])
            task.current_step = step
            task.checkpoint = dict(state)
            task.updated_at = datetime.now(timezone.utc)
            connection.execute("UPDATE tasks SET payload=?,revision=revision+1 WHERE task_id=? AND revision=?", (self._dump(task), task_id, row["revision"]))
            self._event(connection, task, "task.checkpoint", task.status, {"step": step})
        return task

    def set_route(self, decision: RouteDecision) -> Task:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            from dante.task_queue import assert_lease
            assert_lease(connection, decision.task_id, before_effect=True)
            connection.execute("INSERT OR IGNORE INTO routes(route_id,task_id,payload) VALUES(?,?,?)", (decision.route_id, decision.task_id, self._dump(decision)))
            row = connection.execute("SELECT payload FROM tasks WHERE task_id=?", (decision.task_id,)).fetchone()
            if not row:
                raise KeyError(decision.task_id)
            task = Task.model_validate_json(row["payload"])
            task.selected_route = decision.route_id
            task.updated_at = datetime.now(timezone.utc)
            connection.execute("UPDATE tasks SET payload=?,revision=revision+1 WHERE task_id=?", (self._dump(task), task.task_id))
            self._event(connection, task, "route.selected", task.status, {"route_id": decision.route_id, "model_id": decision.selected_model.model_id})
        return task

    def get_route(self, route_id: str) -> RouteDecision:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT payload FROM routes WHERE route_id=?", (route_id,)).fetchone()
        if not row:
            raise KeyError(route_id)
        return RouteDecision.model_validate_json(row["payload"])

    def get_artifact(self, artifact_id: str) -> Artifact:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT payload FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if not row:
            raise KeyError(artifact_id)
        return Artifact.model_validate_json(row["payload"])

    def add_artifact(self, artifact: Artifact) -> Task:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._active_task(connection, artifact.task_id)
            connection.execute("INSERT INTO artifacts(artifact_id,task_id,payload) VALUES(?,?,?)", (artifact.artifact_id, artifact.task_id, self._dump(artifact)))
            row = connection.execute("SELECT payload FROM tasks WHERE task_id=?", (artifact.task_id,)).fetchone()
            if not row:
                raise KeyError(artifact.task_id)
            task = Task.model_validate_json(row["payload"])
            if artifact.artifact_id not in task.artifacts:
                task.artifacts.append(artifact.artifact_id)
            task.updated_at = datetime.now(timezone.utc)
            connection.execute("UPDATE tasks SET payload=?,revision=revision+1 WHERE task_id=?", (self._dump(task), task.task_id))
            self._event(connection, task, "artifact.created", task.status, {"artifact_id": artifact.artifact_id})
        return task

    def recoverable_tasks(self) -> list[Task]:
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED_TERMINAL, TaskStatus.CANCELLED}
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT payload FROM tasks").fetchall()
        return [task for row in rows if (task := Task.model_validate_json(row["payload"])).status not in terminal]

    def events(self, task_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT * FROM task_events WHERE task_id=? ORDER BY event_id", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_acceptance(self, task_id: str) -> AcceptanceContract:
        with closing(self._connect()) as connection:
            row = connection.execute('SELECT contract FROM task_acceptance WHERE task_id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return AcceptanceContract.model_validate_json(row['contract'])

    def bind_acceptance(self, task_id: str, contract: AcceptanceContract) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            self._active_task(connection, task_id)
            row = connection.execute('SELECT * FROM task_acceptance WHERE task_id=?', (task_id,)).fetchone()
            if row is None:
                raise RuntimeError('Missing persistent acceptance')
            if row['locked'] and AcceptanceContract.model_validate_json(row['contract']) != contract:
                raise ValueError('Persistent acceptance cannot be replaced')
            connection.execute('UPDATE task_acceptance SET contract=?,locked=1 WHERE task_id=?',
                               (contract.model_dump_json(), task_id))

    @staticmethod
    def _active_task(connection, task_id: str) -> Task:
        row = connection.execute('SELECT payload FROM tasks WHERE task_id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        task = Task.model_validate_json(row['payload'])
        if task.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED_TERMINAL}:
            raise InvalidTransition('Terminal tasks cannot acquire new execution/evidence')
        return task

    def record_observations(self, task_id: str, observed: ObservedAcceptance) -> None:
        """Compatibility ingress for trusted callers of pre-P2 acceptance APIs.

        Explicit operational contracts never accept these as evidence. For legacy
        tasks, attestations are persisted and journaled tool outcomes take priority.
        """
        from dante.privacy import PrivacyGate
        from dante.contracts import PrivacyClass
        if PrivacyGate().classify(observed.model_dump_json()).classification == PrivacyClass.SECRET:
            raise ValueError('Secret-bearing observations must not be persisted')
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            task = self._active_task(connection, task_id)
            row = connection.execute('SELECT * FROM task_acceptance WHERE task_id=?', (task_id,)).fetchone()
            if row['strict']:
                return
            prior = ObservedAcceptance.model_validate_json(row['attestations'])
            connection.execute('UPDATE task_acceptance SET attestations=? WHERE task_id=?',
                               (prior.merged(observed).model_dump_json(), task_id))
            self._event(connection, task, 'acceptance.caller_attestation', task.status, {'source': 'legacy_trusted_caller'})

    @staticmethod
    def _checked_artifact(connection, task_id: str, artifact_id: str) -> Artifact:
        row = connection.execute('SELECT payload FROM artifacts WHERE artifact_id=? AND task_id=?',
                                 (artifact_id, task_id)).fetchone()
        if row is None:
            raise ValueError('Artifact does not belong to task')
        artifact = Artifact.model_validate_json(row['payload'])
        if file_digest(Path(artifact.path)) != artifact.checksum_sha256:
            raise ValueError('Artifact integrity failed')
        return artifact

    def record_final_evidence(self, task_id: str, artifact_id: str) -> None:
        """Reference a verified JSON object whose string values are final fields."""
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            task = self._active_task(connection, task_id)
            artifact = self._checked_artifact(connection, task_id, artifact_id)
            values = json.loads(Path(artifact.path).read_text(encoding='utf-8'))
            if not isinstance(values, dict) or not all(isinstance(v, str) for v in values.values()):
                raise ValueError('Final field evidence must be a JSON string mapping')
            row = connection.execute('SELECT final_evidence FROM task_acceptance WHERE task_id=?', (task_id,)).fetchone()
            refs = json.loads(row['final_evidence'])
            refs.update({key: artifact_id for key in values})
            connection.execute('UPDATE task_acceptance SET final_evidence=? WHERE task_id=?', (json.dumps(refs), task_id))
            self._event(connection, task, 'acceptance.final_evidence', task.status, {'artifact_id': artifact_id})

    def _acceptance_result(self, connection, task_id: str) -> AcceptanceResult:
        row = connection.execute('SELECT * FROM task_acceptance WHERE task_id=?', (task_id,)).fetchone()
        if row is None:
            return AcceptanceResult(acceptance_complete=False, missing_checks=('acceptance_contract',),
                                    missing_artifacts=(), missing_final_fields=())
        contract = AcceptanceContract.model_validate_json(row['contract'])
        observed = ObservedAcceptance() if row['strict'] else ObservedAcceptance.model_validate_json(row['attestations'])
        steps = [StepRecord.model_validate(dict(item)) for item in connection.execute(
            'SELECT * FROM execution_steps WHERE task_id=?', (task_id,))]
        tools = set(observed.tools) - {step.tool_id for step in steps}
        pending = []
        for step in steps:
            try:
                verified_receipt(step)
                if step.evidence_ref:
                    self._checked_artifact(connection, task_id, step.evidence_ref)
                tools.add(step.tool_id)
            except (ReconciliationRequired, ValueError, OSError):
                pending.append('step:' + step.step_id)
        artifacts = set(observed.artifacts)
        for item in connection.execute('SELECT artifact_id FROM artifacts WHERE task_id=?', (task_id,)):
            try:
                artifact = self._checked_artifact(connection, task_id, item['artifact_id'])
                artifacts.add(artifact.artifact_id)
                artifacts.add(artifact.provenance.get('logical_name', Path(artifact.path).name))
            except (OSError, ValueError):
                pending.append('artifact:' + item['artifact_id'])
        final_fields = dict(observed.final_fields)
        for field, artifact_id in json.loads(row['final_evidence']).items():
            try:
                artifact = self._checked_artifact(connection, task_id, artifact_id)
                value = json.loads(Path(artifact.path).read_text(encoding='utf-8'))[field]
                if isinstance(value, str):
                    final_fields[field] = value
            except (OSError, ValueError, KeyError, TypeError):
                final_fields.pop(field, None)
        result = DeterministicAcceptanceVerifier().verify(contract, ObservedAcceptance(
            tools=tools, artifacts=artifacts, final_fields=final_fields))
        if pending:
            result.acceptance_complete = False
            result.missing_checks += tuple(sorted(pending))
        return result

    def verify_persistent_acceptance(self, task_id: str) -> AcceptanceResult:
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            result = self._acceptance_result(connection, task_id)
            task = self._active_task(connection, task_id)
            self._event(connection, task, 'acceptance.checked', task.status, result.model_dump(mode='json'))
            return result

    def record_tool_outcome(self, task_id: str, tool_digest: str, status: str, step_id: str | None) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            task = self.get_task(task_id)
            self._event(connection, task, 'tool.outcome', task.status,
                        {'tool_digest': tool_digest, 'status': status, 'step_id': step_id})

    def lookup_step(self, task_id: str, tool_id: str, tool_version: str, arguments: dict[str, Any],
                    idempotency_key: str | None = None) -> StepRecord | None:
        args_digest = digest(arguments)
        key = digest(idempotency_key) if idempotency_key is not None else digest([tool_id, tool_version, args_digest])
        with closing(self._connect()) as connection:
            row = connection.execute('SELECT * FROM execution_steps WHERE task_id=? AND idempotency_key=?',
                                     (task_id, key)).fetchone()
        if row is None:
            return None
        step = StepRecord.model_validate(dict(row))
        if (step.tool_id, step.tool_version, step.arguments_digest) != (tool_id, tool_version, args_digest):
            raise ValueError('Idempotency key reused for a different operation')
        return step

    def prepare_step(self, task_id: str, tool_id: str, tool_version: str, arguments: dict[str, Any],
                     idempotency_key: str | None = None) -> StepRecord:
        args_digest = digest(arguments)
        key = digest(idempotency_key) if idempotency_key is not None else digest([tool_id, tool_version, args_digest])
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            from dante.task_queue import assert_lease
            assert_lease(connection, task_id, before_effect=True)
            task = self._active_task(connection, task_id)
            row = connection.execute('SELECT * FROM execution_steps WHERE task_id=? AND idempotency_key=?',
                                     (task_id, key)).fetchone()
            if row:
                step = StepRecord.model_validate(dict(row))
                if (step.tool_id, step.tool_version, step.arguments_digest) != (tool_id, tool_version, args_digest):
                    raise ValueError('Idempotency key reused for a different operation')
                return step
            acceptance_row = connection.execute('SELECT * FROM task_acceptance WHERE task_id=?', (task_id,)).fetchone()
            if acceptance_row is None:
                raise RuntimeError('Missing persistent acceptance')
            contract = AcceptanceContract.model_validate_json(acceptance_row['contract'])
            if not acceptance_row['locked']:
                contract.required_tools = contract.required_tools | {tool_id}
            connection.execute("UPDATE task_acceptance SET strict=1,contract=?,attestations='{}' WHERE task_id=?",
                               (contract.model_dump_json(), task_id))
            step_id = new_id('step')
            connection.execute('''INSERT INTO execution_steps
                (step_id,task_id,tool_id,tool_version,arguments_digest,idempotency_key,state,created_at)
                VALUES(?,?,?,?,?,?,?,?)''',
                (step_id, task_id, tool_id, tool_version, args_digest, key, 'intent', datetime.now(timezone.utc).isoformat()))
            self._event(connection, task, 'step.intent', task.status, {'step_id': step_id, 'arguments_digest': args_digest})
        return self.get_step(task_id, step_id)

    def get_step(self, task_id: str, step_id: str) -> StepRecord:
        with closing(self._connect()) as connection:
            row = connection.execute('SELECT * FROM execution_steps WHERE task_id=? AND step_id=?', (task_id, step_id)).fetchone()
        if row is None:
            raise KeyError(step_id)
        return StepRecord.model_validate(dict(row))

    def steps(self, task_id: str) -> list[StepRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute('SELECT * FROM execution_steps WHERE task_id=? ORDER BY created_at,step_id', (task_id,)).fetchall()
        return [StepRecord.model_validate(dict(row)) for row in rows]

    def verified_step_result(self, task_id: str, step_id: str) -> dict[str, Any]:
        step = self.get_step(task_id, step_id)
        result = verified_receipt(step)
        if step.evidence_ref:
            try:
                with closing(self._connect()) as connection:
                    self._checked_artifact(connection, task_id, step.evidence_ref)
            except (OSError, ValueError):
                raise ReconciliationRequired('Reconciliation evidence is unavailable or altered') from None
        return result

    def claim_step(self, task_id: str, step_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            from dante.task_queue import assert_lease
            assert_lease(connection, task_id, before_effect=True)
            task = self._active_task(connection, task_id)
            row = connection.execute("SELECT evidence_ref FROM execution_steps WHERE task_id=? AND step_id=? AND state='intent'",
                                     (task_id, step_id)).fetchone()
            if row and row['evidence_ref']:
                try:
                    self._checked_artifact(connection, task_id, row['evidence_ref'])
                except (OSError, ValueError):
                    raise ReconciliationRequired('Evidence permitting retry is unavailable') from None
            changed = connection.execute('''UPDATE execution_steps SET state='started',attempt=attempt+1,started_at=?,error=NULL
                WHERE task_id=? AND step_id=? AND state='intent' ''', (datetime.now(timezone.utc).isoformat(), task_id, step_id))
            if changed.rowcount != 1:
                raise ReconciliationRequired('Step already claimed; do not replay')
            self._event(connection, task, 'step.started', task.status, {'step_id': step_id})

    def finish_step(self, task_id: str, step_id: str, result: dict[str, Any]) -> StepRecord:
        safe = receipt(result, self.get_task(task_id).workspace)
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            from dante.task_queue import assert_lease
            assert_lease(connection, task_id, before_effect=False)
            task = self._active_task(connection, task_id)
            changed = connection.execute('''UPDATE execution_steps SET state='succeeded',completed_at=?,result=?,result_digest=?,error=NULL
                WHERE task_id=? AND step_id=? AND state='started' ''',
                (datetime.now(timezone.utc).isoformat(), json.dumps(safe), digest(safe), task_id, step_id))
            if changed.rowcount != 1:
                raise ReconciliationRequired('Step outcome requires reconciliation')
            self._event(connection, task, 'step.succeeded', task.status, {'step_id': step_id, 'result_digest': digest(safe)})
        return self.get_step(task_id, step_id)

    def mark_uncertain(self, task_id: str, step_id: str, error_type: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            from dante.task_queue import assert_lease
            assert_lease(connection, task_id, before_effect=False)
            task = self._active_task(connection, task_id)
            connection.execute("UPDATE execution_steps SET state='uncertain',error=? WHERE task_id=? AND step_id=? AND state='started'",
                               (error_type, task_id, step_id))
            self._event(connection, task, 'step.uncertain', task.status, {'step_id': step_id, 'error_type': error_type})

    def reconcile_step(self, task_id: str, step_id: str, artifact_id: str, *, actor: str, executor_stopped: bool) -> StepRecord:
        """Trusted reconciliation only, after the prior executor has been stopped.

        The JSON evidence must bind step_id, arguments_digest and an outcome of
        applied/not_applied. No automatic guess about arbitrary external effects.
        """
        if not executor_stopped or not actor.strip():
            raise ReconciliationRequired('Confirm executor quiescence and reconciliation actor')
        with closing(self._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            task = self._active_task(connection, task_id)
            row = connection.execute('SELECT * FROM execution_steps WHERE task_id=? AND step_id=?', (task_id, step_id)).fetchone()
            if row is None or row['state'] == 'intent':
                raise ReconciliationRequired('Step is not awaiting reconciliation')
            if row['state'] == 'succeeded':
                try:
                    verified_receipt(StepRecord.model_validate(dict(row)))
                except ReconciliationRequired:
                    pass
                else:
                    raise ReconciliationRequired('A verified outcome cannot be rewritten')
            artifact = self._checked_artifact(connection, task_id, artifact_id)
            evidence = json.loads(Path(artifact.path).read_text(encoding='utf-8'))
            if evidence.get('step_id') != step_id or evidence.get('arguments_digest') != row['arguments_digest']:
                raise ValueError('Reconciliation evidence does not match operation')
            outcome = evidence.get('outcome')
            if outcome == 'applied':
                safe = receipt(evidence['result'], task.workspace)
                connection.execute("""UPDATE execution_steps SET state='succeeded',completed_at=?,result=?,result_digest=?,evidence_ref=?,error=NULL
                    WHERE step_id=?""", (datetime.now(timezone.utc).isoformat(), json.dumps(safe), digest(safe), artifact_id, step_id))
            elif outcome == 'not_applied':
                connection.execute("UPDATE execution_steps SET state='intent',evidence_ref=?,error=NULL WHERE step_id=?", (artifact_id, step_id))
            else:
                raise ReconciliationRequired('Effect remains uncertain')
            self._event(connection, task, 'step.reconciled', task.status,
                        {'step_id': step_id, 'artifact_id': artifact_id, 'actor': actor, 'outcome': outcome})
        return self.get_step(task_id, step_id)
