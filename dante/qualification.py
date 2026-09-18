"""Durable node qualification, independent of health and model approval.

Probes are trusted local code. This is evidence accounting, not remote attestation.
No production runtime is probed or promoted automatically.
"""
from collections.abc import Callable
from contextlib import closing
from typing import Protocol, Literal
from uuid import UUID, uuid4

from dante.contracts import ModelRef, utc_now
from dante.contracts.qualification import (
    Check, CheckEvidence, MachineProfile, ModelQualification, PerformanceEvidence,
    QualificationAssessment, QualificationIdentity, QualificationState as State,
    REQUIRED_CHECKS, SUITE_VERSION,
)
from dante.ledger import TaskLedger
from dante.recovery import digest


def changed_inputs(before: QualificationIdentity, current: QualificationIdentity) -> tuple[str, ...]:
    """Compare semantic inputs only; timestamps, discovery ordering and free disk are irrelevant."""
    def compare(left, right, prefix=''):
        changes = []
        for key in sorted(left.keys() | right.keys()):
            path = prefix + key
            a, b = left.get(key), right.get(key)
            if isinstance(a, dict) and isinstance(b, dict):
                changes.extend(compare(a, b, path + '.'))
            elif a != b:
                changes.append(path)
        return changes
    return tuple(compare(before.identity_data(), current.identity_data()))


def assess(record: ModelQualification, current: QualificationIdentity) -> QualificationAssessment:
    changed = changed_inputs(record.identity, current)
    if changed:
        return QualificationAssessment(state=State.STALE, reasons=tuple('changed:' + key for key in changed))
    failures = tuple('failed:' + item.check.value for item in record.checks if item.outcome == 'failed')
    if record.interrupted:
        failures += ('probe_interrupted',)
    if failures:
        return QualificationAssessment(state=State.FAILED, reasons=failures)
    passed = {item.check for item in record.checks if item.outcome == 'passed'}
    missing = tuple('missing:' + check.value for check in sorted(REQUIRED_CHECKS - passed))
    reasons = missing + tuple('identity_unknown:' + name for name in current.missing_facts())
    if record.completed_at is None:
        reasons += ('run_incomplete',)
    if current.suite_version != SUITE_VERSION:
        reasons += ('unsupported_suite',)
    if record.source != 'hardware':
        reasons += ('synthetic_evidence_only',)
    return QualificationAssessment(state=State.UNKNOWN if reasons else State.QUALIFIED,
        reasons=reasons, measured_capabilities=tuple(sorted(passed)))


class QualificationStore:
    """Append-only snapshots in the existing ledger; no second database or connection policy."""
    def __init__(self, ledger: TaskLedger):
        self.ledger = ledger

    def record_machine(self, machine: MachineProfile) -> str:
        machine = MachineProfile.model_validate_json(machine.model_dump_json())
        payload = machine.model_dump_json()
        key = digest(machine.model_dump(mode='json'))
        with closing(self.ledger._connect()) as db, db:
            db.execute('INSERT OR IGNORE INTO node_profiles VALUES(?,?,?)', (key, str(machine.node_id), payload))
        return key

    def machines(self, node_id: UUID) -> list[MachineProfile]:
        with closing(self.ledger._connect()) as db:
            rows = db.execute('SELECT snapshot_id,payload FROM node_profiles WHERE node_id=? ORDER BY rowid',
                              (str(node_id),)).fetchall()
        profiles = []
        for row in rows:
            profile = MachineProfile.model_validate_json(row['payload'])
            if digest(profile.model_dump(mode='json')) != row['snapshot_id']:
                raise ValueError('Machine evidence integrity failed')
            profiles.append(profile)
        return profiles

    def append(self, record: ModelQualification) -> None:
        record = ModelQualification.model_validate_json(record.model_dump_json())
        payload = record.model_dump_json()
        with closing(self.ledger._connect()) as db, db:
            db.execute('''INSERT INTO model_qualifications
                (qualification_id,node_id,scope,payload,payload_digest) VALUES(?,?,?,?,?)''',
                (str(record.qualification_id), str(record.identity.machine.node_id), record.identity.scope,
                 payload, digest(record.model_dump(mode='json'))))

    @staticmethod
    def _read(row) -> ModelQualification:
        if row is None:
            raise KeyError('Qualification not found')
        record = ModelQualification.model_validate_json(row['payload'])
        if (digest(record.model_dump(mode='json')) != row['payload_digest']
                or str(record.qualification_id) != row['qualification_id']
                or str(record.identity.machine.node_id) != row['node_id']
                or record.identity.scope != row['scope']):
            raise ValueError('Qualification evidence integrity failed')
        return record

    def get(self, qualification_id: UUID) -> ModelQualification:
        with closing(self.ledger._connect()) as db:
            return self._read(db.execute('SELECT * FROM model_qualifications WHERE qualification_id=?',
                                        (str(qualification_id),)).fetchone())

    def history(self, node_id: UUID) -> list[ModelQualification]:
        with closing(self.ledger._connect()) as db:
            rows = db.execute('SELECT * FROM model_qualifications WHERE node_id=? ORDER BY sequence',
                              (str(node_id),)).fetchall()
        return [self._read(row) for row in rows]

    def latest(self, current: QualificationIdentity) -> ModelQualification | None:
        # Arrival order, not caller-supplied timestamps; a failed rerun cannot resurrect an old pass.
        with closing(self.ledger._connect()) as db:
            row = db.execute('SELECT * FROM model_qualifications WHERE scope=? ORDER BY sequence DESC LIMIT 1',
                             (current.scope,)).fetchone()
        return self._read(row) if row else None

    def status(self, current: QualificationIdentity) -> QualificationAssessment:
        latest = self.latest(current)
        return assess(latest, current) if latest else QualificationAssessment(state=State.UNKNOWN, reasons=('no_evidence',))

    def require_qualified(self, current: QualificationIdentity, *, tool_use=False) -> None:
        result = self.status(current)
        if result.state != State.QUALIFIED or (tool_use and Check.TOOL_CALLING not in result.measured_capabilities):
            raise ValueError('Current node/model tuple is not qualified')


class QualificationProbe(Protocol):
    """Future bounded local harness: owns timeouts, cancellation, evidence files and measurements.

    Each check must test its named behavior, not just endpoint reachability. No raw
    prompts, responses, environment dumps or exception text cross this interface.
    """
    source: Literal['synthetic', 'hardware']

    def observe(self) -> QualificationIdentity: ...
    def run(self, check: Check) -> CheckEvidence: ...
    def performance(self) -> PerformanceEvidence: ...


class QualificationRunner:
    def __init__(self, store: QualificationStore):
        self.store = store

    def run(self, probe: QualificationProbe) -> ModelQualification:
        identity = QualificationIdentity.model_validate_json(probe.observe().model_dump_json())
        started = utc_now()
        # Crash during a new qualification leaves UNKNOWN, never an old qualified result.
        intent = ModelQualification(qualification_id=uuid4(), identity=identity,
                                    source=probe.source, started_at=started)
        self.store.append(intent)
        checks, interrupted, performance = [], False, PerformanceEvidence()
        try:
            for check in Check:
                evidence = CheckEvidence.model_validate_json(probe.run(check).model_dump_json())
                if evidence.check != check:
                    raise ValueError('Mismatched check evidence')
                checks.append(evidence)
            performance = PerformanceEvidence.model_validate_json(probe.performance().model_dump_json())
            if changed_inputs(identity, probe.observe()):
                interrupted = True  # Identity changed while measurements were in flight.
        except Exception:
            interrupted = True  # Never persist arbitrary exception messages.
        record = ModelQualification(qualification_id=uuid4(), identity=identity, source=intent.source,
            checks=tuple(checks), performance=performance, started_at=started,
            completed_at=utc_now(), interrupted=interrupted)
        self.store.append(record)
        return record


class NodeQualificationGate:
    """Optional additional registry gate; never grants lifecycle, license or privacy approval."""
    def __init__(self, store: QualificationStore, current: Callable[[ModelRef], QualificationIdentity]):
        self.store, self.current = store, current

    def __call__(self, model: ModelRef, *, tool_use=False) -> None:
        identity = self.current(model)
        metadata = model.local_metadata
        artifact = None if metadata is None else (
            metadata.sha256 if identity.artifact_kind == 'file' else metadata.runtime_digest)
        if (metadata is None or identity.model_id != model.model_id
                or identity.runtime.runtime_id != model.provider_id or identity.runtime.runtime != model.runtime
                or artifact is None or identity.artifact_sha256 != artifact
                or identity.quantization != metadata.quantization or identity.context_tokens != metadata.context_tokens):
            raise ValueError('Qualification identity does not match model')
        self.store.require_qualified(identity, tool_use=tool_use)
