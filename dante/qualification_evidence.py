"""Content-addressed, bounded evidence for real node qualification.

The SQLite record holds only a digest. A production gate verifies the corresponding
document before treating a completed hardware attempt as routable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from dante.contracts.qualification import Check, ModelQualification, QualificationIdentity


MAX_EVIDENCE_BYTES = 64 * 1024


class EvidenceDocument(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    identity_fingerprint: str = Field(pattern=r'^[a-f0-9]{64}$')
    node_id: str
    check: Check
    outcome: str
    started_at: AwareDatetime
    completed_at: AwareDatetime
    facts: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    failure: str | None = None


class EvidenceStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _payload(document: EvidenceDocument) -> bytes:
        payload = json.dumps(document.model_dump(mode='json'), sort_keys=True,
                             separators=(',', ':'), ensure_ascii=True).encode('ascii')
        if len(payload) > MAX_EVIDENCE_BYTES:
            raise ValueError('Qualification evidence exceeds size limit')
        return payload

    def put(self, document: EvidenceDocument) -> str:
        payload = self._payload(document)
        key = hashlib.sha256(payload).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / (key + '.json')
        fd, temporary = tempfile.mkstemp(prefix='.evidence-', suffix='.tmp', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if not self.verify(key, document.identity_fingerprint, document.node_id,
                           document.check, document.outcome):
            raise ValueError('Qualification evidence failed read-back verification')
        return key

    def verify(self, key: str, identity_fingerprint: str, node_id: str,
               check: Check, outcome: str) -> bool:
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            return False
        try:
            path = self.root / (key + '.json')
            if path.stat().st_size > MAX_EVIDENCE_BYTES:
                return False
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != key:
                return False
            document = EvidenceDocument.model_validate_json(payload)
            return (document.identity_fingerprint == identity_fingerprint
                    and document.node_id == node_id and document.check == check
                    and document.outcome == outcome
                    and document.completed_at >= document.started_at)
        except (OSError, ValueError):
            return False

    def verify_record(self, record: ModelQualification) -> bool:
        identity: QualificationIdentity = record.identity
        return all(item.evidence_sha256 is not None
                   and self.verify(item.evidence_sha256, identity.fingerprint,
                                   str(identity.machine.node_id), item.check, item.outcome)
                   for item in record.checks if item.outcome == 'passed')
