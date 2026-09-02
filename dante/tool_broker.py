from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from dante.contracts import Task, ToolManifest
from dante.contracts.tools import ToolResult, ToolStatus, check_schema, validate_schema
from dante.recovery import digest
from dante.telemetry import JsonlAudit


class ToolDenied(RuntimeError):
    def __init__(self, result: ToolResult | str):
        self.result = result if isinstance(result, ToolResult) else ToolResult(status=ToolStatus.POLICY_DENIED)
        super().__init__(self.result.status.value)


@dataclass(frozen=True)
class WorkspaceMethod:
    handler: Callable
    root: Path
    permission: str
    schema: dict
    paths: dict

    def __call__(self, **arguments):
        return self.handler(**arguments)


def contained_path(root: Path, value: str) -> Path:
    windows = PureWindowsPath(value)
    if '..' in windows.parts or '..' in Path(value).parts:
        raise ValueError('Traversal denied')
    candidate = Path(value)
    if windows.drive and not candidate.is_absolute():
        raise ValueError('Ambiguous path')
    candidate = candidate if candidate.is_absolute() else root / candidate
    # Reject existing links/junctions, including roots, before resolving them.
    for part in (candidate, *candidate.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('Reparse path denied')
    resolved = candidate.resolve()
    resolved.relative_to(root.resolve())
    return resolved


class ToolBroker:
    """Enforcement for trusted registered Python tools, not an OS sandbox.

    Deadline expiry abandons the outcome, not the thread. Effects may continue;
    P2 must reconcile only after executor quiescence. No shell/network tools.
    """
    def __init__(self, audit: JsonlAudit | None = None, *, ledger=None) -> None:
        self.audit, self.ledger = audit, ledger
        self._tools: dict[str, tuple[ToolManifest, Callable, frozenset | None]] = {}

    def register(self, manifest: ToolManifest, handler: Callable[..., Any], *,
                 required_permissions: frozenset[str] | set[str] | None = None) -> None:
        manifest = manifest.model_copy(deep=True)
        if isinstance(handler, WorkspaceMethod):
            if manifest.arguments_schema is None:
                manifest.arguments_schema = handler.schema
                manifest.path_permissions = handler.paths
            required_permissions = {handler.permission}
        self._tools[manifest.tool_id] = (manifest, handler, None if required_permissions is None else frozenset(required_permissions))

    def manifest(self, tool_id: str) -> ToolManifest:
        if tool_id not in self._tools:
            raise ToolDenied('Tool is not registered')
        return self._tools[tool_id][0].model_copy(deep=True)

    def preflight(self, task: Task, tool_id: str, arguments: dict[str, Any]) -> ToolResult | None:
        if tool_id not in self._tools:
            return ToolResult(status=ToolStatus.POLICY_DENIED)
        manifest, handler, required = self._tools[tool_id]
        supported = {'read_workspace', 'write_workspace'}
        if (required is None or required != manifest.permissions or not required <= supported
                or manifest.risk not in {'R0', 'R1'} or manifest.secret_access != 'none'
                or manifest.network_scope != 'none' or manifest.approval_policy not in {'never', 'always'}
                or (manifest.risk == 'R0' and 'write_workspace' in required)
                or manifest.arguments_schema is None or not manifest.version.strip() or not manifest.tool_id.strip()):
            return ToolResult(status=ToolStatus.POLICY_DENIED)
        try:
            check_schema(manifest.arguments_schema)
        except (ValueError, TypeError, RecursionError):
            return ToolResult(status=ToolStatus.POLICY_DENIED)
        try:
            digest(arguments)
            if not isinstance(arguments, dict) or manifest.arguments_schema.get('type') != 'object':
                raise ValueError('Object required')
            validate_schema(manifest.arguments_schema, arguments)
            if isinstance(handler, WorkspaceMethod):
                validate_schema(handler.schema, arguments)
        except (ValueError, TypeError, KeyError, RecursionError):
            return ToolResult(status=ToolStatus.VALIDATION_DENIED)
        try:
            if required:
                root = Path(manifest.filesystem_scope)
                if not root.is_absolute() or not Path(task.workspace).is_absolute():
                    raise ValueError('Absolute scope required')
                contained_path(Path(task.workspace), str(root))
                if isinstance(handler, WorkspaceMethod):
                    if handler.root != root.resolve() or manifest.path_permissions != handler.paths:
                        raise ValueError('Wrapper scope mismatch')
                for name, permission in manifest.path_permissions.items():
                    if permission not in required:
                        raise ValueError('Undeclared path permission')
                    if name in arguments:
                        if not isinstance(arguments[name], str):
                            raise ValueError('Path must be a string')
                        contained_path(root, arguments[name])
            elif manifest.filesystem_scope != 'none' or manifest.path_permissions:
                raise ValueError('Undeclared filesystem scope')
        except (ValueError, OSError, RuntimeError):
            return ToolResult(status=ToolStatus.POLICY_DENIED)
        return None

    def action_hash(self, task: Task, tool_id: str, arguments: dict, plan_hash: str = '') -> str:
        manifest = self.manifest(tool_id)
        metadata = manifest.model_dump(mode='json')
        metadata['permissions'] = sorted(manifest.permissions)
        return digest({'task': task.task_id, 'workspace': task.workspace,
                       'manifest': metadata, 'arguments': arguments, 'plan': plan_hash})

    def authorize(self, task: Task, tool_id: str, arguments: dict, *, approval_id: str | None = None,
                  plan_hash: str = '') -> ToolResult | None:
        if self.manifest(tool_id).approval_policy == 'never':
            return None
        if approval_id is None:
            return ToolResult(status=ToolStatus.APPROVAL_REQUIRED)
        if self.ledger is None:
            return ToolResult(status=ToolStatus.APPROVAL_INVALID)
        from dante.approvals import ApprovalManager
        try:
            ApprovalManager(self.ledger).consume_action(approval_id, task.task_id,
                self.action_hash(task, tool_id, arguments, plan_hash))
        except PermissionError:
            return ToolResult(status=ToolStatus.APPROVAL_INVALID)
        return None

    def _run(self, task: Task, tool_id: str, arguments: dict) -> ToolResult:
        manifest, handler, _ = self._tools[tool_id]
        outcome = queue.Queue(maxsize=1)

        def work():
            try:
                raw = handler(**arguments)
                if isinstance(raw, str):
                    if len(raw.encode('utf-8')) > manifest.output_limit_bytes:
                        outcome.put(ToolResult(status=ToolStatus.OUTPUT_LIMIT, effect_uncertain=True))
                        return
                    raw = json.loads(raw)
                # Stream the encoding to avoid constructing a second unbounded payload.
                size = 0
                for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(raw):
                    size += len(chunk.encode('utf-8'))
                    if size > manifest.output_limit_bytes:
                        outcome.put(ToolResult(status=ToolStatus.OUTPUT_LIMIT, effect_uncertain=True))
                        return
                if not isinstance(raw, dict) or raw.get('ok') is not True:
                    raise RuntimeError('Tool failed')
                outcome.put(ToolResult(status=ToolStatus.SUCCESS, data=json.loads(json.dumps(raw, allow_nan=False))))
            except Exception as exc:
                outcome.put(ToolResult(status=ToolStatus.EXECUTION_FAILURE, effect_uncertain=True,
                                       error_type=type(exc).__name__[:80]))

        threading.Thread(target=work, daemon=True, name='dante-tool').start()
        try:
            return outcome.get(timeout=manifest.timeout_s)
        except queue.Empty:
            return ToolResult(status=ToolStatus.TIMEOUT, effect_uncertain=True)

    def record(self, task: Task, tool_id: str, result: ToolResult, step_id: str | None = None) -> None:
        # Never persist rejected arguments, arbitrary output or exception messages.
        if self.ledger:
            self.ledger.record_tool_outcome(task.task_id, digest(tool_id), result.status.value, step_id)
        if self.audit:
            self.audit.write('tool.outcome', tool_digest=digest(tool_id), status=result.status.value)

    def invoke(self, task: Task, tool_id: str, arguments: dict[str, Any], *,
               approval_id: str | None = None, plan_hash: str = '',
               idempotency_key: str | None = None) -> ToolResult:
        if self.ledger is not None:
            task = self.ledger.get_task(task.task_id)
        # Own the argument snapshot; caller mutations cannot change the approved action.
        try:
            digest(arguments)
            arguments = json.loads(json.dumps(arguments, allow_nan=False))
        except (ValueError, TypeError, RecursionError):
            result = ToolResult(status=ToolStatus.VALIDATION_DENIED)
            self.record(task, tool_id, result)
            return result
        result = self.preflight(task, tool_id, arguments)
        step = None
        if result is None:
            manifest = self.manifest(tool_id)
            if self.ledger is None:
                result = ToolResult(status=ToolStatus.POLICY_DENIED)
            else:
                from dante.recovery import ReconciliationRequired
                try:
                    step = self.ledger.lookup_step(task.task_id, tool_id, manifest.version, arguments, idempotency_key)
                except ValueError:
                    result = ToolResult(status=ToolStatus.VALIDATION_DENIED)
                    self.record(task, tool_id, result)
                    return result
                try:
                    if step is not None and step.state == 'succeeded':
                        result = ToolResult(status=ToolStatus.SUCCESS,
                                            data=self.ledger.verified_step_result(task.task_id, step.step_id))
                    elif step is not None and step.state in {'started', 'uncertain'}:
                        result = ToolResult(status=ToolStatus.UNCERTAIN, effect_uncertain=True)
                    else:
                        result = self.authorize(task, tool_id, arguments, approval_id=approval_id, plan_hash=plan_hash)
                        if result is None:
                            step = self.ledger.prepare_step(task.task_id, tool_id, manifest.version, arguments, idempotency_key)
                            self.ledger.claim_step(task.task_id, step.step_id)
                            result = self._run(task, tool_id, arguments)
                            if result.status == ToolStatus.SUCCESS:
                                self.ledger.finish_step(task.task_id, step.step_id, result.data)
                            else:
                                self.ledger.mark_uncertain(task.task_id, step.step_id,
                                    result.error_type or result.status.value)
                except ReconciliationRequired:
                    result = ToolResult(status=ToolStatus.UNCERTAIN, effect_uncertain=True)
                except Exception:
                    if step is not None:
                        self.ledger.mark_uncertain(task.task_id, step.step_id, 'execution_failure')
                    raise
        result.step_id = step.step_id if step else None
        self.record(task, tool_id, result, result.step_id)
        return result

    def execute(self, task: Task, tool_id: str, arguments: dict[str, Any], *, approved: bool = False,
                approval_id: str | None = None, plan_hash: str = '', idempotency_key: str | None = None) -> dict[str, Any]:
        # Legacy boolean never grants permission. The structured API is invoke().
        result = self.invoke(task, tool_id, arguments, approval_id=approval_id,
                             plan_hash=plan_hash, idempotency_key=idempotency_key)
        if result.status != ToolStatus.SUCCESS:
            raise ToolDenied(result)
        return result.data


def workspace_method(tools: Any, method: str) -> WorkspaceMethod:
    permissions = {'ensure_workspace': 'write_workspace', 'create_directory': 'write_workspace',
                   'write_text_file': 'write_workspace', 'file_exists': 'read_workspace',
                   'list_directory': 'read_workspace'}
    if method not in permissions:
        raise ToolDenied('Workspace method is not allowlisted')
    properties = {} if method == 'ensure_workspace' else {'relative_path': {'type': 'string'}}
    required = [] if method in {'ensure_workspace', 'list_directory'} else ['relative_path']
    if method == 'write_text_file':
        properties['content'] = {'type': 'string'}
        required.append('content')
    schema = {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}
    paths = {'relative_path': permissions[method]} if properties else {}
    return WorkspaceMethod(getattr(tools, method), tools._root, permissions[method], schema, paths)
