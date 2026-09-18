"""Minimal Node 0 bootstrap: persistent identity plus conservative P7 inspection."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from dante.contracts.qualification import EvidenceModel, MachineProfile, QualificationState
from dante.node_probe import probe_machine
from dante.qualification import QualificationStore


NOT_YET_QUALIFIED = 'NOT_YET_QUALIFIED'


class NodeBootstrapReport(EvidenceModel):
    node_id: UUID
    node_id_created: bool
    snapshot_id: str
    machine: MachineProfile
    qualification_state: Literal[QualificationState.UNKNOWN] = QualificationState.UNKNOWN
    qualification_status: Literal['NOT_YET_QUALIFIED'] = NOT_YET_QUALIFIED
    reasons: tuple[Literal['hardware_evidence_missing', 'runtime_evidence_missing'], ...] = (
        'hardware_evidence_missing', 'runtime_evidence_missing')


def _read_node_id(path: Path) -> UUID:
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise ValueError('Node identity must not be a link')
    value = path.read_text(encoding='ascii').strip()
    node_id = UUID(value)
    if node_id.version != 4 or str(node_id) != value:
        raise ValueError('Node identity must be a canonical UUID4')
    return node_id


def persistent_node_id(path: Path) -> tuple[UUID, bool]:
    """Create once with publish-if-absent semantics; never replace an existing identity."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return _read_node_id(path), False

    node_id = uuid4()
    temporary: Path | None = None
    created = False
    try:
        with tempfile.NamedTemporaryFile('w', encoding='ascii', dir=path.parent,
                                         prefix='.dante-node-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(str(node_id) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    persisted = _read_node_id(path)
    if created and persisted != node_id:
        raise RuntimeError('Persisted node identity differs from generated identity')
    return persisted, created


def inspect_node(store: QualificationStore, node_id: UUID,
                 probe: Callable[[UUID], MachineProfile] | None = None) -> tuple[str, MachineProfile]:
    """Run the existing P7 machine probe and persist its immutable snapshot."""
    machine = MachineProfile.model_validate_json((probe or probe_machine)(node_id).model_dump_json())
    if machine.node_id != node_id:
        raise ValueError('Machine profile node identity mismatch')
    return store.record_machine(machine), machine


def bootstrap_node(store: QualificationStore, node_id_file: Path, *,
                   probe: Callable[[UUID], MachineProfile] | None = None) -> NodeBootstrapReport:
    node_id, created = persistent_node_id(node_id_file)
    snapshot_id, machine = inspect_node(store, node_id, probe)
    # Machine inventory alone cannot form a runtime/model qualification identity.
    return NodeBootstrapReport(node_id=node_id, node_id_created=created,
        snapshot_id=snapshot_id, machine=machine)
