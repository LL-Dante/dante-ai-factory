"""Authenticated, bounded local operator control for the live Node 0 process.

Windows named-pipe ACLs restrict access to the supervisor's user, SYSTEM, and
Administrators. The wire format is length-prefixed JSON; no pickle or command
execution is available through this interface.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import threading
import time
from ctypes import wintypes

from dante.contracts.inference import DEFAULT_MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS, ThinkingPolicy


PIPE_NAME = r"\\.\pipe\DanteNode0.Operator.v1"
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 128 * 1024
MAX_INFER_PROMPT_CHARS = 8192
MAX_INFER_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
MAX_DIAGNOSTIC_BYTES = 8192
PIPE_IO_TIMEOUT_S = 75.0
PIPE_ACK_TIMEOUT_S = 5.0
_ALLOWED_OPERATIONS = {
    "status", "health", "infer", "workload_submit", "workload_status",
    "workload_result", "workload_cancel", "workload_events", "workload_list",
    "orchestrator_status", "agent_list", "agent_describe", "agent_submit",
    "agent_job", "agent_job_cancel", "agent_result",
}

_DIAGNOSTIC_KEYS = {
    'http_status', 'http_content_type', 'response_bytes', 'json_decoded',
    'top_level_type', 'top_level_keys', 'done_present', 'done_type', 'done_value',
    'model_present', 'model_type', 'model_value', 'message_present', 'message_type',
    'message_keys', 'role_present', 'role_type', 'role_value', 'content_present',
    'content_type', 'content_length', 'thinking_present', 'thinking_type',
    'thinking_length', 'tool_calls_present', 'tool_calls_type', 'tool_calls_count',
    'generation_completed', 'done_reason', 'failed_validation_rule', 'failed_field',
    'expected', 'actual',
}
_SAFE_DIAGNOSTIC_TEXT = re.compile(r'^[A-Za-z0-9_.$:/\[\] <>!=-]{0,192}$')
_SAFE_DIAGNOSTIC_KEY = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')


def _bounded_diagnostic(value):
    if not isinstance(value, dict):
        return None
    result = {}
    for key, item in value.items():
        if key not in _DIAGNOSTIC_KEYS:
            continue
        if key.endswith('_keys') and isinstance(item, list):
            result[key] = [name if isinstance(name, str) and _SAFE_DIAGNOSTIC_KEY.fullmatch(name)
                           else '<nonstandard-key>' for name in item[:24]]
        elif key in {'http_status', 'response_bytes', 'content_length', 'thinking_length', 'tool_calls_count'}:
            if item is None or (type(item) is int and 0 <= item <= 64 * 1024 * 1024):
                result[key] = item
        elif key.endswith('_present') or key == 'json_decoded' or key == 'generation_completed':
            if type(item) is bool:
                result[key] = item
        elif key == 'done_value':
            if item is None or type(item) in (bool, int, float) or (
                    isinstance(item, str) and _SAFE_DIAGNOSTIC_TEXT.fullmatch(item)):
                result[key] = item
        elif item is None or (isinstance(item, str) and _SAFE_DIAGNOSTIC_TEXT.fullmatch(item)):
            result[key] = item
    try:
        if len(json.dumps(result, ensure_ascii=True, separators=(',', ':')).encode('utf-8')) > MAX_DIAGNOSTIC_BYTES:
            return {'failed_validation_rule': 'diagnostic_size_limit'}
    except (TypeError, ValueError):
        return None
    return result or None


class ControlError(Exception):
    def __init__(self, code: str, *, status: int = 400, diagnostic=None):
        self.code, self.status = code, status
        self.diagnostic = _bounded_diagnostic(diagnostic)
        super().__init__(code)


class ControlServerStartupError(RuntimeError):
    """The named-pipe listener did not reach its ready state."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.startup_message = message
        super().__init__(f"Node 0 control server startup failed ({error_type}): {message}")


class Node0ControlService:
    """Typed operations over the already-running production supervisor."""

    def __init__(self, supervisor, *, workload_store=None, orchestrator=None,
                 agent_registry=None):
        self.supervisor = supervisor
        self.workload_store = workload_store
        self.orchestrator = orchestrator
        self.agent_registry = agent_registry

    def dispatch(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("op"), str):
            raise ControlError("malformed_request")
        operation = request["op"]
        if operation not in _ALLOWED_OPERATIONS:
            raise ControlError("operation_not_supported", status=404)
        handlers = {
            "status": self._status,
            "health": self._health,
            "infer": self._infer,
            "workload_submit": self._workload_submit,
            "workload_status": self._workload_status,
            "workload_result": self._workload_result,
            "workload_cancel": self._workload_cancel,
            "workload_events": self._workload_events,
            "workload_list": self._workload_list,
            "orchestrator_status": self._orchestrator_status,
            "agent_list": self._agent_list,
            "agent_describe": self._agent_describe,
            "agent_submit": self._agent_submit,
            "agent_job": self._agent_job,
            "agent_job_cancel": self._agent_job_cancel,
            "agent_result": self._agent_result,
        }
        return {"ok": True, "result": handlers[operation](request)}

    @staticmethod
    def _only(request, keys):
        if set(request) != set(keys) | {"op"}:
            raise ControlError("invalid_request_fields")

    def _status(self, request):
        self._only(request, set())
        return self.supervisor.operator_status()

    def _health(self, request):
        self._only(request, set())
        return self.supervisor.operator_health()

    def _infer(self, request):
        allowed = {"op", "model", "prompt", "max_output_tokens", "thinking"}
        if not {"op", "model", "prompt"}.issubset(request) or set(request) - allowed:
            raise ControlError("invalid_request_fields")
        model, prompt = request["model"], request["prompt"]
        tokens = request.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        thinking = request.get("thinking", ThinkingPolicy.OFF.value)
        configured = self.supervisor.config.qualification.model_reference
        if model != configured:
            raise ControlError("model_not_configured")
        if (not isinstance(prompt, str) or not prompt.strip()
                or len(prompt) > MAX_INFER_PROMPT_CHARS
                or len(prompt.encode("utf-8")) > MAX_REQUEST_BYTES // 2):
            raise ControlError("invalid_prompt")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= MAX_INFER_OUTPUT_TOKENS:
            raise ControlError("invalid_output_limit")
        if not isinstance(thinking, str) or thinking not in {policy.value for policy in ThinkingPolicy}:
            raise ControlError("invalid_thinking_policy")
        from dante.contracts.continuity import ExecutionPolicy
        policy = getattr(getattr(self.supervisor.node, "continuity", None), "policy", None)
        if policy is None or policy.mode != ExecutionPolicy.LOCAL_ONLY:
            raise ControlError("local_only_policy_required", status=403)
        if not self.supervisor.operator_status().get("ready_for_local_routing"):
            raise ControlError("qualification_or_runtime_not_ready", status=503)
        try:
            response = self.supervisor.infer(prompt, model_reference=model,
                                             max_output_tokens=tokens, thinking=thinking)
            health = self.supervisor.operator_health()
        except ControlError:
            raise
        except Exception as exc:
            from dante.continuity import ContinuitySignal
            from dante.inference import InferenceTimeout, QualificationDenied
            from dante.node0_supervisor import Node0InferenceBusy
            if isinstance(exc, Node0InferenceBusy):
                raise ControlError("inference_busy", status=409) from None
            if isinstance(exc, ContinuitySignal):
                code = exc.outcome.reason
                raise ControlError(code if code.isidentifier() else "route_unavailable", status=503,
                                   diagnostic=exc.outcome.diagnostic) from None
            if isinstance(exc, InferenceTimeout):
                raise ControlError("inference_timeout", status=504) from None
            if isinstance(exc, QualificationDenied):
                raise ControlError("qualification_denied", status=403) from None
            raise ControlError("inference_failed", status=502) from None
        text = response.content
        if not text.strip():
            raise ControlError("empty_inference_response", status=502)
        if (response.fallback or response.provider_id != "ollama"
                or response.model.local_metadata is None
                or response.model.local_metadata.runtime_reference != model):
            raise ControlError("local_route_not_verified", status=502)
        if not health.get("healthy") or not health.get("gpu_execution_verified"):
            raise ControlError("gpu_execution_not_verified", status=502)
        return {
            "provider": response.provider_id,
            "model": model,
            "model_id": response.model.model_id,
            "response": text,
            "fallback": False,
            "thinking": thinking,
            "max_output_tokens": tokens,
            "done": True,
            "done_reason": getattr(response, "finish_reason", "unknown"),
            "output_tokens": getattr(getattr(response, "usage", None), "output_tokens", None),
            "qualification_id": health.get("qualification_id"),
            "gpu_uuid": health.get("gpu_uuid"),
            "gpu_vram_bytes": health.get("gpu_vram_bytes"),
            "runtime_version": health.get("runtime_version"),
        }

    def _require_workload(self):
        if self.workload_store is None:
            raise ControlError("workload_service_unavailable", status=503)
        return self.workload_store

    def _workload_submit(self, request):
        self._only(request, {"spec", "idempotency_key"})
        from dante.workload import ModelRequirement, WorkloadSpec
        store = self._require_workload()
        try:
            spec = WorkloadSpec.model_validate(request["spec"])
        except Exception:
            raise ControlError("invalid_workload_spec") from None
        if spec.job_type != 'LOCAL_INFERENCE':
            raise ControlError('agent_task_requires_agent_submit', status=403)
        cfg = self.supervisor.config
        metadata = cfg.model.local_metadata
        if (not metadata or spec.model.model_id != cfg.model.model_id
                or spec.model.runtime_reference != cfg.qualification.model_reference
                or spec.model.digest_sha256 != cfg.qualification.model_digest
                or spec.model.context_tokens > cfg.qualification.context_tokens
                or spec.model.local_only is not True):
            raise ControlError("workload_model_not_qualified", status=403)
        key = request["idempotency_key"]
        if key is not None and (not isinstance(key, str) or not key or len(key) > 200):
            raise ControlError("invalid_idempotency_key")
        record = store.submit(spec, idempotency_key=key)
        return record.model_dump(mode="json")

    def _workload_status(self, request):
        self._only(request, {"job_id"})
        return self._require_workload().get(request["job_id"]).model_dump(mode="json")

    def _workload_result(self, request):
        self._only(request, {"job_id"})
        record = self._require_workload().get(request["job_id"])
        if record.result is None:
            raise ControlError("workload_result_unavailable", status=409)
        return {"job_id": request["job_id"], "state": record.state.value, "result": record.result}

    def _workload_cancel(self, request):
        self._only(request, {"job_id"})
        return self._require_workload().request_cancel(request["job_id"]).model_dump(mode="json")

    def _workload_events(self, request):
        self._only(request, {"job_id"})
        return self._require_workload().events(request["job_id"])

    def _workload_list(self, request):
        self._only(request, {"limit"})
        limit = request["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ControlError("invalid_limit")
        return [item.model_dump(mode="json") for item in self._require_workload().list(limit=limit)]

    def _orchestrator_status(self, request):
        self._only(request, set())
        if self.orchestrator is None:
            return {"state": "STOPPED"}
        return {"state": self.orchestrator.state, "worker_id": self.orchestrator.worker_id}

    def _require_agents(self):
        if self.agent_registry is None:
            raise ControlError('agent_service_unavailable', status=503)
        self._require_workload()
        return self.agent_registry

    def _agent_list(self, request):
        self._only(request, set())
        return [definition.model_dump(mode='json') for definition in self._require_agents().list()]

    def _agent_describe(self, request):
        self._only(request, {'agent_id'})
        try:
            definition = self._require_agents().describe(request['agent_id'])
        except (KeyError, LookupError):
            raise ControlError('unknown_agent', status=404) from None
        return definition.model_dump(mode='json')

    def _agent_submit(self, request):
        required = {'agent_id', 'objective'}
        allowed = required | {'op', 'context', 'requested_output_tokens', 'idempotency_key'}
        if not required.issubset(request) or set(request) - allowed:
            raise ControlError('invalid_request_fields')
        from dante.agent_runner import AgentUnavailable, definition_digest
        from dante.contracts.agents import AgentTaskPayload
        from dante.workload import ModelRequirement, WorkloadSpec

        registry = self._require_agents()
        agent_id = request['agent_id']
        if not isinstance(agent_id, str):
            raise ControlError('unknown_agent', status=404)
        try:
            registration = registry.require(agent_id)
        except AgentUnavailable:
            raise ControlError('unknown_agent', status=404) from None
        definition = registration.definition
        snapshot = self.supervisor.operator_status()
        target = definition.model_target
        cfg = self.supervisor.config
        if (not snapshot.get('ready_for_local_routing') or not snapshot.get('gate_accepted')
                or target.model_id != cfg.model.model_id
                or target.runtime_reference != cfg.qualification.model_reference
                or target.digest_sha256 != cfg.qualification.model_digest
                or target.context_tokens > cfg.qualification.context_tokens):
            raise ControlError('qualification_or_runtime_not_ready', status=503)
        requested = request.get('requested_output_tokens')
        if requested is None:
            output_tokens = definition.output_token_budget
        elif (isinstance(requested, bool) or not isinstance(requested, int)
                or not 64 <= requested <= definition.output_token_budget):
            raise ControlError('invalid_output_budget')
        else:
            output_tokens = requested
        try:
            payload = AgentTaskPayload(
                agent_id=agent_id,
                objective=request['objective'],
                context=request.get('context'),
                requested_output_tokens=output_tokens,
                definition_version=definition.version,
                definition_digest=definition_digest(definition),
            )
            model = ModelRequirement(**target.model_dump())
            spec = WorkloadSpec(
                job_type='AGENT_TASK',
                model=model,
                prompt='registered-agent-task:' + agent_id,
                agent_task=payload,
                max_output_tokens=output_tokens,
                thinking=definition.thinking,
                timeout_s=definition.timeout_s,
                maximum_attempts=definition.retry_policy.maximum_attempts,
                retry_base_s=definition.retry_policy.retry_base_s,
                retry_max_s=definition.retry_policy.retry_max_s,
            )
        except Exception:
            raise ControlError('invalid_agent_request') from None
        key = request.get('idempotency_key')
        if key is not None and (not isinstance(key, str) or not key.strip() or len(key) > 200):
            raise ControlError('invalid_idempotency_key')
        record = self._require_workload().submit(spec, idempotency_key=key)
        return record.model_dump(mode='json')

    def _agent_job(self, request):
        self._only(request, {'job_id'})
        store = self._require_workload()
        try:
            spec = store.spec(request['job_id'])
        except KeyError:
            raise ControlError('unknown_job', status=404) from None
        if spec.job_type != 'AGENT_TASK':
            raise ControlError('not_an_agent_job', status=404)
        return store.get(request['job_id']).model_dump(mode='json')

    def _agent_job_cancel(self, request):
        self._only(request, {'job_id'})
        store = self._require_workload()
        try:
            spec = store.spec(request['job_id'])
        except KeyError:
            raise ControlError('unknown_job', status=404) from None
        if spec.job_type != 'AGENT_TASK':
            raise ControlError('not_an_agent_job', status=404)
        record = (self.orchestrator.cancel(request['job_id']) if self.orchestrator is not None
                  else store.request_cancel(request['job_id']))
        store.record_event(request['job_id'], 'agent.job.cancel_requested',
            agent_id=spec.agent_task.agent_id, state=record.state.value)
        return record.model_dump(mode='json')

    def _agent_result(self, request):
        self._only(request, {'job_id'})
        store = self._require_workload()
        try:
            spec = store.spec(request['job_id'])
        except KeyError:
            raise ControlError('unknown_job', status=404) from None
        if spec.job_type != 'AGENT_TASK':
            raise ControlError('not_an_agent_job', status=404)
        record = store.get(request['job_id'])
        if record.result is None or not isinstance(record.result.get('structured_result'), dict):
            raise ControlError('agent_result_unavailable', status=409)
        return {'job_id': str(record.job_id), 'state': record.state.value,
                'result': record.result['structured_result'],
                'execution': {key: record.result.get(key) for key in
                    ('model_id', 'provider_id', 'runtime', 'fallback', 'metadata')}}


def encode_frame(value, *, maximum=MAX_REQUEST_BYTES):
    try:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ControlError("invalid_json_value") from None
    if not body or len(body) > maximum:
        raise ControlError("message_too_large", status=413)
    return struct.pack("!I", len(body)) + body


def decode_payload(raw):
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise ControlError("message_too_large", status=413)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, TypeError):
        raise ControlError("malformed_json") from None
    if not isinstance(value, dict):
        raise ControlError("malformed_request")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _win_api():
    if os.name != "nt":
        raise OSError("Node 0 operator control requires Windows named pipes")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.restype = wintypes.HANDLE
    kernel.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel.CreateNamedPipeW.restype = wintypes.HANDLE
    kernel.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel.ConnectNamedPipe.restype = wintypes.BOOL
    kernel.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
    kernel.DisconnectNamedPipe.restype = wintypes.BOOL
    kernel.SetNamedPipeHandleState.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD)]
    kernel.SetNamedPipeHandleState.restype = wintypes.BOOL
    kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.WriteFile.restype = wintypes.BOOL
    kernel.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel.CancelIoEx.restype = wintypes.BOOL
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel.WaitNamedPipeW.restype = wintypes.BOOL
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [wintypes.LPCWSTR,
        wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.DWORD)]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    return kernel, advapi


def _current_user_sid():
    kernel, advapi = _win_api()
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise OSError(ctypes.get_last_error(), "OpenProcessToken failed")
    try:
        needed = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        if not needed.value or needed.value > 65536:
            raise OSError("Invalid current-user token size")
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi.GetTokenInformation(token, 1, buffer, needed, ctypes.byref(needed)):
            raise OSError(ctypes.get_last_error(), "GetTokenInformation failed")

        class SidAndAttributes(ctypes.Structure):
            _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

        class TokenUser(ctypes.Structure):
            _fields_ = [("User", SidAndAttributes)]

        sid = TokenUser.from_buffer(buffer).User.Sid
        text = wintypes.LPWSTR()
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(text)):
            raise OSError(ctypes.get_last_error(), "ConvertSidToStringSidW failed")
        try:
            return text.value
        finally:
            kernel.LocalFree(text)
    finally:
        kernel.CloseHandle(token)


def current_user_pipe_sddl():
    sid = _current_user_sid()
    return f"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;{sid})"


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL)]


def _pipe_security_attributes():
    kernel, advapi = _win_api()
    descriptor = wintypes.LPVOID()
    sddl = current_user_pipe_sddl()
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(descriptor), None):
        raise OSError(ctypes.get_last_error(), "Pipe security descriptor creation failed")
    attrs = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False)
    return kernel, descriptor, attrs


def _close(handle):
    if handle and handle != wintypes.HANDLE(-1).value:
        kernel, _ = _win_api()
        kernel.CloseHandle(handle)


def _set_nowait(handle):
    kernel, _ = _win_api()
    mode = wintypes.DWORD(0x00000001)  # PIPE_READMODE_BYTE | PIPE_NOWAIT
    if not kernel.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None):
        raise OSError(ctypes.get_last_error(), "SetNamedPipeHandleState failed")


def _read_exact(handle, size: int, deadline: float) -> bytes:
    kernel, _ = _win_api()
    chunks = bytearray()
    while len(chunks) < size:
        buffer = ctypes.create_string_buffer(size - len(chunks))
        count = wintypes.DWORD()
        ok = kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None)
        if ok and count.value:
            chunks.extend(buffer.raw[:count.value])
            continue
        error = ctypes.get_last_error()
        if error in (232, 997):  # no data yet / operation pending
            if time.monotonic() >= deadline:
                raise TimeoutError("pipe read timed out")
            time.sleep(0.01)
            continue
        if error == 0:
            raise EOFError("pipe closed")
        raise OSError(error, "pipe read failed")
    return bytes(chunks)


def _write_all(handle, data: bytes, deadline: float):
    kernel, _ = _win_api()
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + 16384]
        buffer = ctypes.create_string_buffer(chunk)
        count = wintypes.DWORD()
        ok = kernel.WriteFile(handle, buffer, len(chunk), ctypes.byref(count), None)
        if ok and count.value:
            offset += count.value
            continue
        error = ctypes.get_last_error()
        if error in (232, 997):
            if time.monotonic() >= deadline:
                raise TimeoutError("pipe write timed out")
            time.sleep(0.01)
            continue
        raise OSError(error, "pipe write failed")


def _read_frame(handle, *, maximum: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    length = struct.unpack("!I", _read_exact(handle, 4, deadline))[0]
    if not 1 <= length <= maximum:
        raise ControlError("message_too_large", status=413)
    return _read_exact(handle, length, deadline)


class Node0ControlServer:
    """Single-owner Windows named-pipe server running beside the supervisor."""

    def __init__(self, service: Node0ControlService, *, pipe_name=PIPE_NAME):
        self.service, self.pipe_name = service, pipe_name
        self._stop = threading.Event()
        self._thread = None
        self._active = set()
        self._active_lock = threading.Lock()
        self._listen_handle = None
        self._instances = threading.BoundedSemaphore(4)
        self._startup_event = threading.Event()
        self.started = False
        self.ready = False
        self.startup_error: ControlServerStartupError | None = None

    def start(self):
        if self._thread and self._thread.is_alive() and self.ready:
            return
        if self._thread and self._thread.is_alive():
            raise ControlServerStartupError("already_starting", "listener thread is not ready")
        self._stop.clear()
        self._startup_event.clear()
        self.started = True
        self.ready = False
        self.startup_error = None
        self._thread = threading.Thread(target=self._accept, name="dante-node0-control", daemon=True)
        self._thread.start()
        if not self._startup_event.wait(5.0):
            error = ControlServerStartupError("TimeoutError", "listener creation exceeded 5 seconds")
            self.startup_error = error
            self.stop(timeout_s=2.0)
            raise error
        if self.startup_error is not None:
            raise self.startup_error
        if not self.ready:
            raise ControlServerStartupError("listener_not_ready", "startup completed without a listener")

    def _accept(self):
        first = True
        self._ready = False
        try:
            kernel, _ = _win_api()
            while not self._stop.is_set():
                descriptor = wintypes.LPVOID()
                try:
                    kernel, descriptor, attrs = _pipe_security_attributes()
                    open_mode = 0x00000003 | (0x00080000 if first else 0)
                    pipe_mode = 0x00000008  # PIPE_REJECT_REMOTE_CLIENTS; byte mode, blocking I/O
                    handle = kernel.CreateNamedPipeW(self.pipe_name, open_mode, pipe_mode, 4,
                        MAX_RESPONSE_BYTES, MAX_REQUEST_BYTES, 1000, ctypes.byref(attrs))
                finally:
                    if descriptor:
                        kernel.LocalFree(descriptor)
                if handle == wintypes.HANDLE(-1).value:
                    code = ctypes.get_last_error()
                    raise OSError(code, "CreateNamedPipeW failed")
                self._listen_handle = handle
                first = False
                if not self.ready:
                    self.ready = True
                    self._ready = True
                    self._startup_event.set()
                try:
                    connected = kernel.ConnectNamedPipe(handle, None)
                    error = 0 if connected else ctypes.get_last_error()
                    if not connected and error != 535:  # ERROR_PIPE_CONNECTED
                        if self._stop.is_set() or error in (995, 6):
                            break
                        raise OSError(error, "ConnectNamedPipe failed")
                    if not self._instances.acquire(blocking=False):
                        continue
                    thread = threading.Thread(target=self._serve_client, args=(handle,),
                                              name="dante-node0-control-client", daemon=True)
                    with self._active_lock:
                        self._active.add(thread)
                    try:
                        thread.start()
                    except Exception:
                        with self._active_lock:
                            self._active.discard(thread)
                        self._instances.release()
                        raise
                    handle = None
                finally:
                    self._listen_handle = None
                    if handle:
                        try:
                            kernel.DisconnectNamedPipe(handle)
                        except Exception:
                            pass
                        _close(handle)
        except Exception as exc:
            error = ControlServerStartupError(type(exc).__name__, str(exc))
            self.startup_error = error
            if not self._startup_event.is_set():
                self._startup_event.set()
            self._stop.set()
        finally:
            self._ready = False
            self.ready = False

    def _serve_client(self, handle):
        try:
            _set_nowait(handle)
            raw = _read_frame(handle, maximum=MAX_REQUEST_BYTES, timeout_s=5.0)
            request = decode_payload(raw)
            try:
                response = self.service.dispatch(request)
            except ControlError as exc:
                error = {"code": exc.code}
                if exc.diagnostic is not None:
                    error['diagnostic'] = exc.diagnostic
                response = {"ok": False, "error": error}
            except Exception:
                response = {"ok": False, "error": {"code": "internal_error"}}
            try:
                frame = encode_frame(response, maximum=MAX_RESPONSE_BYTES)
            except ControlError:
                frame = encode_frame({"ok": False, "error": {"code": "response_too_large"}},
                                     maximum=MAX_RESPONSE_BYTES)
            _write_all(handle, frame, time.monotonic() + PIPE_IO_TIMEOUT_S)
            # A server-side disconnect can discard a response still buffered in
            # the pipe. Keep the instance connected until the client confirms
            # that it consumed the complete response frame.
            ack = decode_payload(_read_frame(handle, maximum=1024, timeout_s=PIPE_ACK_TIMEOUT_S))
            if ack != {"ack": True}:
                raise ControlError("invalid_response_ack")
        except Exception:
            pass
        finally:
            try:
                kernel, _ = _win_api()
                kernel.DisconnectNamedPipe(handle)
            except Exception:
                pass
            _close(handle)
            self._instances.release()
            with self._active_lock:
                self._active.discard(threading.current_thread())

    def stop(self, *, timeout_s=35):
        self._stop.set()
        try:
            kernel, _ = _win_api()
            if self._listen_handle:
                try:
                    kernel.CancelIoEx(self._listen_handle, None)
                except Exception:
                    pass
            # A local connection wakes blocking ConnectNamedPipe without sending a request.
            handle = _open_pipe(self.pipe_name, timeout_s=1.0, allow_missing=True)
            if handle:
                _close(handle)
        except OSError:
            pass
        if self._thread:
            self._thread.join(timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._active_lock:
                active = list(self._active)
            if not any(thread.is_alive() for thread in active):
                break
            time.sleep(0.05)
        self.started = bool(self._thread and self._thread.is_alive())
        self.ready = bool(self.started and self._ready)


def _open_pipe(pipe_name: str, *, timeout_s: float, allow_missing=False):
    kernel, _ = _win_api()
    deadline = time.monotonic() + timeout_s
    while True:
        handle = kernel.CreateFileW(pipe_name, 0xC0000000, 0, None, 3, 0, None)
        if handle != wintypes.HANDLE(-1).value:
            _set_nowait(handle)
            return handle
        error = ctypes.get_last_error()
        if allow_missing and error in (2, 231):
            return None
        if time.monotonic() >= deadline:
            raise TimeoutError("Node 0 control pipe is unavailable")
        wait_ms = max(1, min(250, int((deadline - time.monotonic()) * 1000)))
        if not kernel.WaitNamedPipeW(pipe_name, wait_ms):
            continue


class Node0ControlClient:
    def __init__(self, *, pipe_name=PIPE_NAME, timeout_s=PIPE_IO_TIMEOUT_S):
        self.pipe_name, self.timeout_s = pipe_name, timeout_s

    def request(self, payload):
        frame = encode_frame(payload)
        handle = _open_pipe(self.pipe_name, timeout_s=min(5.0, self.timeout_s))
        try:
            deadline = time.monotonic() + self.timeout_s
            _write_all(handle, frame, deadline)
            raw = _read_frame(handle, maximum=MAX_RESPONSE_BYTES, timeout_s=self.timeout_s)
            _write_all(handle, encode_frame({"ack": True}, maximum=1024),
                       time.monotonic() + PIPE_ACK_TIMEOUT_S)
            response = decode_payload(raw)
            if response.get("ok") is not True:
                error = response.get("error")
                code = error.get("code") if isinstance(error, dict) else "control_request_failed"
                diagnostic = error.get('diagnostic') if isinstance(error, dict) else None
                raise ControlError(code if isinstance(code, str) else "control_request_failed", status=502,
                                   diagnostic=diagnostic)
            return response["result"]
        finally:
            _close(handle)


__all__ = ["ControlError", "Node0ControlClient", "Node0ControlServer", "Node0ControlService",
           "MAX_REQUEST_BYTES", "MAX_RESPONSE_BYTES", "MAX_INFER_PROMPT_CHARS",
           "MAX_INFER_OUTPUT_TOKENS", "PIPE_NAME", "current_user_pipe_sddl", "decode_payload", "encode_frame"]
