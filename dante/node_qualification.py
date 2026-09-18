"""Minimal orchestration of bootstrap, P7 probes, lifecycle and persisted verdicts."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from dante.contracts.qualification import (
    Check, CheckEvidence, EvidenceModel, MachineProfile, PerformanceEvidence,
    QualificationAssessment, QualificationIdentity, QualificationState,
)
from dante.node_bootstrap import inspect_node, persistent_node_id
from dante.qualification import QualificationProbe, QualificationRunner, QualificationStore, assess


class QualificationProbeFactory(Protocol):
    """Composition seam for a future reviewed runtime-specific probe."""
    def __call__(self, identity: QualificationIdentity) -> QualificationProbe: ...


class ObservationOnlyProbe:
    """Default safe probe: records an attempt but claims no hardware/runtime check."""
    source: Literal['synthetic'] = 'synthetic'

    def __init__(self, identity: QualificationIdentity):
        self.identity = identity

    def observe(self) -> QualificationIdentity:
        return self.identity

    def run(self, check: Check) -> CheckEvidence:
        return CheckEvidence(check=check, outcome='unknown')

    def performance(self) -> PerformanceEvidence:
        return PerformanceEvidence()


class QualificationAttemptReport(EvidenceModel):
    node_id: UUID
    node_id_created: bool
    machine_snapshot_id: str
    attempt_id: UUID
    qualification_id: UUID
    attempt_phase: Literal['completed'] = 'completed'
    previous_state: QualificationState
    qualification_state: QualificationState
    qualification_status: Literal['UNKNOWN', 'QUALIFIED', 'FAILED', 'STALE']
    reasons: tuple[str, ...]


class NodeQualificationHarness:
    def __init__(self, store: QualificationStore, node_id_file: Path, *,
                 machine_probe: Callable[[UUID], MachineProfile] | None = None,
                 probe_factory: QualificationProbeFactory | None = None):
        self.store = store
        self.node_id_file = node_id_file
        self.machine_probe = machine_probe
        self.probe_factory = probe_factory

    def qualify(self, requested: QualificationIdentity) -> QualificationAttemptReport:
        node_id, created = persistent_node_id(self.node_id_file)
        if requested.machine.node_id != node_id:
            raise ValueError('Qualification identity belongs to a different node')
        snapshot_id, machine = inspect_node(self.store, node_id, self.machine_probe)
        current = requested.model_copy(update={'machine': machine})
        previous = self.store.status(current)
        probe = self.probe_factory(current) if self.probe_factory else ObservationOnlyProbe(current)
        record = QualificationRunner(self.store).run(probe)
        result = assess(record, current)
        if record.attempt_id is None:
            raise RuntimeError('Qualification runner omitted attempt identity')
        return QualificationAttemptReport(node_id=node_id, node_id_created=created,
            machine_snapshot_id=snapshot_id, attempt_id=record.attempt_id,
            qualification_id=record.qualification_id, previous_state=previous.state,
            qualification_state=result.state, qualification_status=result.state.value.upper(),
            reasons=result.reasons)
