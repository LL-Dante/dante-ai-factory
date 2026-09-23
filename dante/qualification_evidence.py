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
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from dante.contracts.qualification import Check, ModelQualification, QualificationIdentity


MAX_EVIDENCE_BYTES = 64 * 1024


class EvidenceDocument(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    identity_fingerprint: str = Field(pattern=r'^[a-f0-9]{64}$')
    node_id: str
    attempt_id: UUID | None = None
    check: Check
    outcome: str
    started_at: AwareDatetime
    completed_at: AwareDatetime
    deadline_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    gpu_uuid: str | None = None
    gpu_name: str | None = None
    gpu_driver: str | None = None
    runtime_id: str | None = None
    runtime_version: str | None = None
    runtime_configuration_sha256: str | None = None
    model_reference: str | None = None
    model_digest: str | None = None
    quantization: str | None = None
    context_tokens: int | None = None
    qualification_configuration_sha256: str | None = None
    facts: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    failure: str | None = None


class EvidenceStore:
    def __init__(self, root: Path, *, strict: bool = False,
                 model_reference: str | None = None):
        self.root = Path(root)
        self.strict = strict
        self.model_reference = model_reference

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
        document = self.read(key)
        return (document is not None and document.identity_fingerprint == identity_fingerprint
                and document.node_id == node_id and document.check == check
                and document.outcome == outcome and document.completed_at >= document.started_at)

    def read(self, key: str) -> EvidenceDocument | None:
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            return None
        try:
            path = self.root / (key + '.json')
            if path.stat().st_size > MAX_EVIDENCE_BYTES:
                return None
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != key:
                return None
            return EvidenceDocument.model_validate_json(payload)
        except (OSError, ValueError):
            return None

    def verify_record(self, record: ModelQualification) -> bool:
        identity: QualificationIdentity = record.identity
        for item in record.checks:
            if item.outcome != 'passed':
                continue
            if item.evidence_sha256 is None:
                return False
            document = self.read(item.evidence_sha256)
            if (document is None or document.identity_fingerprint != identity.fingerprint
                    or document.node_id != str(identity.machine.node_id)
                    or document.check != item.check or document.outcome != item.outcome
                    or document.completed_at < document.started_at):
                return False
            if self.strict:
                gpu = next((gpu for gpu in identity.machine.gpus or ()
                            if gpu.uuid == document.gpu_uuid), None)
                if (record.attempt_id is None or document.attempt_id != record.attempt_id
                        or document.deadline_s is None or gpu is None
                        or document.gpu_name != gpu.name or document.gpu_driver != gpu.driver_version
                        or document.runtime_id != identity.runtime.runtime_id
                        or document.runtime_version != identity.runtime.version
                        or document.runtime_configuration_sha256 != identity.runtime.configuration_sha256
                        or not document.model_reference
                        or (self.model_reference is not None
                            and document.model_reference != self.model_reference)
                        or document.model_digest != identity.artifact_sha256
                        or document.quantization != identity.quantization
                        or document.context_tokens != identity.context_tokens
                        or document.qualification_configuration_sha256 != identity.configuration_sha256):
                    return False
        return True
