"""Durable receipts deliberately exclude arbitrary tool output and arguments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from dante.contracts import StrictModel


class ReconciliationRequired(RuntimeError):
    pass


class StepRecord(StrictModel):
    step_id: str
    task_id: str
    tool_id: str
    tool_version: str
    arguments_digest: str
    idempotency_key: str
    attempt: int
    state: Literal['intent', 'started', 'succeeded', 'uncertain']
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: str | None = None
    result_digest: str | None = None
    evidence_ref: str | None = None
    error: str | None = None


def digest(value: Any) -> str:
    def check_keys(item):
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError('JSON object keys must be strings')
            for nested in item.values():
                check_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                check_keys(nested)
    check_keys(value)
    normalized = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def receipt(result: dict[str, Any], workspace: str) -> dict[str, Any]:
    if result.get('ok') is not True:
        raise ValueError('A successful result is required')
    safe: dict[str, Any] = {'ok': True}
    if isinstance(result.get('path'), str):
        path = Path(result['path']).resolve()
        try:
            path.relative_to(Path(workspace).resolve())
        except ValueError:
            pass
        else:
            if path.is_file():
                safe.update(path=str(path), checksum_sha256=file_digest(path))
    return safe


def verified_receipt(step: StepRecord) -> dict[str, Any]:
    if step.state != 'succeeded' or step.result is None:
        raise ReconciliationRequired('Step has no verified durable outcome')
    result = json.loads(step.result)
    if result.get('ok') is not True or digest(result) != step.result_digest:
        raise ReconciliationRequired('Step receipt integrity failed')
    if 'path' in result:
        try:
            valid = file_digest(Path(result['path'])) == result['checksum_sha256']
        except (OSError, KeyError):
            valid = False
        if not valid:
            raise ReconciliationRequired('Step file evidence changed or is unavailable')
    return result
