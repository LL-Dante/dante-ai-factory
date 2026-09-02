from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from dante.contracts import Artifact
from dante.ledger import TaskLedger
from dante.recovery import file_digest


class ArtifactStore:
    def __init__(self, root: Path, ledger: TaskLedger) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger

    def create(self, task_id: str, filename: str, data: bytes, *, media_type: str, producer: str,
               provenance: dict[str, Any], step_id: str | None = None) -> Artifact:
        if Path(filename).name != filename or filename in {'', '.', '..'}:
            raise ValueError('Artifact filename must be a simple relative name')
        self.ledger.get_task(task_id)
        step_id = step_id or provenance.get('step_id')
        if step_id is not None:
            self.ledger.get_step(task_id, step_id)
        checksum = hashlib.sha256(data).hexdigest()
        directory = (self.root / 'blobs' / 'sha256' / checksum[:2]).resolve()
        directory.relative_to(self.root)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / checksum
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                temporary = handle.name
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Atomic publish-if-absent. Never replace a committed blob.
                os.link(temporary, target)
            except FileExistsError:
                pass
            if file_digest(target) != checksum:
                raise ValueError('Existing immutable blob has failed integrity verification')
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        lineage = {**provenance, 'logical_name': filename}
        if step_id is not None:
            lineage['step_id'] = step_id
        artifact = Artifact(task_id=task_id, media_type=media_type, path=str(target), checksum_sha256=checksum,
                            producer=producer, provenance=lineage)
        self.ledger.add_artifact(artifact)
        return artifact
