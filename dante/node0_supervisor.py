"""Long-lived fail-closed Node 0 runtime supervisor.

The supervisor reuses the production Node 0 graph and its single qualification
store. Runtime liveness and qualification trust remain separate decisions.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from dante.contracts import ModelRef, PrivacyClass
from dante.contracts.inference import DEFAULT_MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS, ThinkingPolicy
from dante.contracts.continuity import ContinuityPolicy, ExecutionPolicy
from dante.contracts.qualification import (
    QualificationIdentity, QualificationState, RuntimeObservation,
)
from dante.contracts.runtime import RuntimeProfile
from dante.node0 import Node0Runtime, build_node0
from dante.node_probe import probe_machine
from dante.nvidia_probe import discover_gpu_uuids
from dante.ollama_qualification import OllamaQualificationConfig, ProbeFailure
from dante.inference import inference_cancellation
from dante.recovery import digest
from dante.telemetry import JsonlAudit


class SupervisorConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    node_uuid: UUID
    machine_profile: str = Field(min_length=1)
    ledger_path: Path
    evidence_path: Path
    audit_path: Path
    workload_path: Path | None = None
    state_path: Path
    config_path: Path
    model: ModelRef
    qualification: OllamaQualificationConfig
    health_interval_s: float = Field(default=30, ge=5, le=300)
    startup_timeout_s: float = Field(default=60, ge=5, le=180)
    shutdown_timeout_s: float = Field(default=15, ge=2, le=60)
    restart_window_s: float = Field(default=900, ge=60, le=86400)
    max_restarts: int = Field(default=3, ge=1, le=10)
    restart_backoff_s: float = Field(default=2, ge=1, le=60)
    max_backoff_s: float = Field(default=60, ge=2, le=300)
    stable_reset_s: float = Field(default=300, ge=30, le=3600)
    audit_max_bytes: int = Field(default=5 * 1024 * 1024, ge=65536, le=100 * 1024 * 1024)

    @classmethod
    def load(cls, path: Path, *, state_path: Path | None = None) -> 'SupervisorConfig':
        raw = json.loads(path.read_text(encoding='utf-8'))
        node_id = UUID(str(raw['node_uuid']))
        if node_id.version != 4 or str(node_id) != str(raw['node_uuid']).lower():
            raise ValueError('Node identity must be a canonical UUID4')
        runtime = raw['runtime']
        supervisor_settings = raw.get('supervisor', {})
        if not isinstance(supervisor_settings, dict):
            raise ValueError('Invalid supervisor settings')
        profile = RuntimeProfile(runtime='ollama', base_url=runtime['endpoint'])
        qualification = OllamaQualificationConfig(profile=profile,
            model_reference=runtime['model_reference'], model_digest=runtime['model_digest'],
            quantization=runtime['quantization'], context_tokens=runtime['context_tokens'],
            gpu_uuid=runtime['gpu_uuid'], executable=Path(runtime['executable']),
            model_store=Path(runtime['model_store']),
            startup_timeout_s=supervisor_settings.get('startup_timeout_s', 60),
            shutdown_timeout_s=supervisor_settings.get('shutdown_timeout_s', 15))
        model = ModelRef.model_validate(raw['model'])
        if (profile.base_url != 'http://127.0.0.1:11435'
                or not model.local or model.runtime != 'ollama' or model.provider_id != 'ollama'
                or model.local_metadata is None
                or not model.available
                or model.lifecycle.value not in {'approved', 'production'}
                or model.local_metadata.license_status != 'verified'
                or model.local_metadata.runtime_reference != qualification.model_reference
                or model.local_metadata.runtime_digest != qualification.model_digest
                or model.local_metadata.quantization != qualification.quantization
                or model.local_metadata.context_tokens != qualification.context_tokens
                or model.local_metadata.tool_use is not False):
            raise ValueError('Supervisor requires an exact local Ollama model and loopback endpoint')
        if not qualification.executable.is_file() or qualification.executable.name.lower() != 'ollama.exe':
            raise ValueError('Configured Ollama executable is unavailable')
        if not qualification.model_store.is_dir():
            raise ValueError('Configured model store is unavailable')
        paths = raw
        config_file = Path(os.path.abspath(os.fspath(path)))
        ledger_path = _absolute_from_config(paths['ledger_path'], config_file)
        evidence_path = _absolute_from_config(paths['evidence_path'], config_file)
        audit_path = _absolute_from_config(paths['audit_path'], config_file)
        configured_state = state_path or paths.get('state_path')
        snapshot_path = (_absolute_from_config(configured_state, config_file)
            if configured_state else ledger_path.parent / 'supervisor-state.json')
        data = {'node_uuid': node_id, 'machine_profile': paths['machine_profile'],
            'ledger_path': ledger_path, 'evidence_path': evidence_path,
            'workload_path': ledger_path.parent / 'workloads.db',
            'audit_path': audit_path, 'state_path': snapshot_path,
            'config_path': config_file, 'model': model, 'qualification': qualification}
        allowed_settings = {'health_interval_s', 'startup_timeout_s', 'shutdown_timeout_s',
            'restart_window_s', 'max_restarts', 'restart_backoff_s', 'max_backoff_s',
            'stable_reset_s', 'audit_max_bytes'}
        supervisor_settings = raw.get('supervisor', {})
        if not isinstance(supervisor_settings, dict) or set(supervisor_settings) - allowed_settings:
            raise ValueError('Invalid supervisor settings')
        data.update(supervisor_settings)
        return cls.model_validate(data)


def _absolute_from_config(value: str | Path, config_file: Path) -> Path:
    """Normalize configured paths lexically, preserving the selected filesystem alias."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_file.parent / candidate
    return Path(os.path.abspath(os.fspath(candidate)))

class ProcessHandle(Protocol):
    pid: int
    def poll(self) -> int | None: ...


class Runtime(Protocol):
    owned: bool
    process: ProcessHandle | None
    def start(self) -> None: ...
    def stop(self) -> bool: ...
    def pids(self) -> frozenset[int]: ...


class InstanceLock(Protocol):
    def acquire(self) -> bool: ...
    def release(self) -> None: ...


class Node0InferenceBusy(RuntimeError):
    """The single GPU inference slot is already occupied."""


def _failure_code(error: Exception) -> str:
    if isinstance(error, ProbeFailure):
        return error.reason
    return type(error).__name__


class WindowsMutex:
    """Kernel mutex is released by Windows if the owning process exits."""
    def __init__(self, name: str):
        self.name = name
        self.handle = None

    def acquire(self) -> bool:
        if os.name != 'nt':
            return True
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        handle = kernel.CreateMutexW(None, False, self.name)
        if not handle:
            return False
        self.handle = handle
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            self.release()
            return False
        return True

    def release(self) -> None:
        if self.handle and os.name == 'nt':
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle(self.handle)
        self.handle = None


class RotatingAudit:
    """Small bounded wrapper: current log plus one previous file."""
    def __init__(self, path: Path, max_bytes: int):
        self.path, self.max_bytes = path, max_bytes
        self._lock = threading.Lock()
        self._audit = JsonlAudit(path)

    def write(self, event: str, **metadata) -> None:
        with self._lock:
            try:
                if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                    previous = self.path.with_suffix(self.path.suffix + '.1')
                    previous.unlink(missing_ok=True)
                    os.replace(self.path, previous)
                    self._audit = JsonlAudit(self.path)
            except OSError:
                # Do not report readiness if audit cannot be written.
                raise
            self._audit.write(event, **metadata)


@dataclass(frozen=True)
class SupervisorSnapshot:
    state: str
    node_uuid: str
    runtime_state: str
    runtime_pid: int | None
    model_reference: str
    expected_model_digest: str
    observed_model_digest: str | None
    runtime_version: str | None
    qualification_state: str
    qualification_id: str | None
    evidence_identity: str | None
    gpu_identity: str | None
    driver_version: str | None
    last_health_check: str | None
    last_qualification_verification: str | None
    restart_count: int
    last_failure: str | None
    ready_for_local_routing: bool
    updated_at: str


@dataclass
class Node0Supervisor:
    config: SupervisorConfig
    node: Node0Runtime
    runtime: Runtime
    observer: object
    http: object
    audit: RotatingAudit
    lock: InstanceLock = field(default_factory=lambda: WindowsMutex('Local\\DanteNode0Supervisor'))
    clock: Callable[[], float] = time.monotonic
    stop_event: threading.Event = field(default_factory=threading.Event)
    state: str = 'STOPPED'
    runtime_healthy: bool = False
    qualification_state: str = 'unknown'
    qualification_id: str | None = None
    observed_identity: QualificationIdentity | None = None
    last_health_check: str | None = None
    last_qualification_verification: str | None = None
    last_failure: str | None = None
    restart_times: deque = field(default_factory=deque)
    _ready: bool = False
    _lock_acquired: bool = False
    _last_audited_ready: bool | None = None
    _started_at: float | None = None
    _snapshot_lock: threading.RLock = field(default_factory=threading.RLock)
    _inference_lock: threading.Lock = field(default_factory=threading.Lock)
    control_server: object | None = None
    workload_store: object | None = None
    orchestrator: object | None = None
    _orchestrator_thread: threading.Thread | None = None

    def __post_init__(self):
        # The existing shared gate is the routing authority. Its identity source
        # now also checks the supervisor's live readiness state.
        self.node.gate.current = self._routing_identity

    @classmethod
    def from_file(cls, path: Path) -> 'Node0Supervisor':
        config = SupervisorConfig.load(path)
        raw = json.loads(path.read_text(encoding='utf-8'))
        machine = probe_machine(config.node_uuid, uuid_probe=discover_gpu_uuids)
        seed = QualificationIdentity(
            machine=machine,
            model_id=config.model.model_id,
            runtime=RuntimeObservation(runtime_id='ollama', runtime='ollama', backend='cuda',
                probe_version='node0-supervisor-intent-v1'),
            artifact_kind='runtime_manifest', artifact_sha256=config.qualification.model_digest,
            quantization=config.qualification.quantization, context_tokens=config.qualification.context_tokens,
            configuration_sha256=digest(raw['runtime']))
        audit = RotatingAudit(config.audit_path, config.audit_max_bytes)
        node = build_node0(ledger_path=config.ledger_path, evidence_path=config.evidence_path,
            audit_path=config.audit_path, seed_identity=seed, qualification=config.qualification,
            models=[config.model], machine_profile=config.machine_profile,
            policy=ContinuityPolicy(mode=ExecutionPolicy.LOCAL_ONLY), audit=audit)
        # A control-plane inference has a hard runtime I/O ceiling. Keep the
        # local adapter's response-size bound while avoiding any unbounded call.
        node.adapter.profile = node.adapter.profile.model_copy(update={
            'timeout_s': min(node.adapter.profile.timeout_s, 30.0)})
        probe = node.probe
        return cls(config, node, probe.runtime, probe.observer, probe.http,
                   audit)

    def _routing_identity(self, _model: ModelRef) -> QualificationIdentity:
        with self._snapshot_lock:
            if not self._ready or self.state != 'QUALIFIED' or not self.runtime_healthy:
                raise ValueError('Node 0 supervisor is not ready')
        try:
            identity = self.observer()
            assessment = self.node.store.status(identity)
        except Exception as exc:
            with self._snapshot_lock:
                self._ready = False
                self.last_failure = _failure_code(exc)
                self.qualification_state = ('stale' if isinstance(exc, ProbeFailure)
                    and exc.reason == 'assertion_failed' else 'unknown')
                self._transition('DEGRADED', reason=self.last_failure)
                self._persist_snapshot()
            raise ValueError('Current Node 0 identity or evidence is unavailable') from None
        with self._snapshot_lock:
            if not self._ready or self.state != 'QUALIFIED' or not self.runtime_healthy:
                raise ValueError('Node 0 readiness changed during identity verification')
            if assessment.state != QualificationState.QUALIFIED:
                self._ready = False
                self.qualification_state = assessment.state.value
                self.last_failure = 'qualification_identity_or_evidence_changed'
                self._transition('DEGRADED', reason=self.last_failure)
                self._persist_snapshot()
                raise ValueError('Node 0 qualification is no longer current')
            self.observed_identity = identity
            self.qualification_state = assessment.state.value
            self.last_qualification_verification = self._now()
        return identity

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _transition(self, state: str, *, reason: str | None = None) -> None:
        previous = self.state
        self.state = state
        if previous != state:
            self.audit.write('node0.supervisor.state', previous=previous, state=state,
                             reason_type=reason)
        ready = bool(self._ready and state == 'QUALIFIED' and self.runtime_healthy)
        if ready != self._last_audited_ready:
            self._last_audited_ready = ready
            self.audit.write('node0.routing.readiness', ready=ready, supervisor_state=state)

    def _persist_snapshot(self) -> None:
        snapshot = self.snapshot()
        target = self.config.state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.supervisor-', suffix='.tmp', dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(json.dumps(snapshot.__dict__, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def snapshot(self) -> SupervisorSnapshot:
        identity = self.observed_identity
        record = self.node.store.latest(identity) if identity is not None else None
        pid = getattr(self.runtime.process, 'pid', None)
        return SupervisorSnapshot(state=self.state, node_uuid=str(self.config.node_uuid),
            runtime_state='healthy' if self.runtime_healthy else 'unhealthy', runtime_pid=pid,
            model_reference=self.config.qualification.model_reference,
            expected_model_digest=self.config.qualification.model_digest,
            observed_model_digest=identity.artifact_sha256 if identity else None,
            runtime_version=identity.runtime.version if identity else None,
            qualification_state=self.qualification_state,
            qualification_id=str(record.qualification_id) if record else self.qualification_id,
            evidence_identity=identity.fingerprint if identity else None,
            gpu_identity=next((g.uuid for g in identity.machine.gpus or ()
                               if g.uuid == self.config.qualification.gpu_uuid), None) if identity else None,
            driver_version=next((g.driver_version for g in identity.machine.gpus or ()
                                 if g.uuid == self.config.qualification.gpu_uuid), None) if identity else None,
            last_health_check=self.last_health_check,
            last_qualification_verification=self.last_qualification_verification,
            restart_count=len(self.restart_times), last_failure=self.last_failure,
            ready_for_local_routing=self._ready, updated_at=self._now())

    def _ensure_exact_runtime(self) -> QualificationIdentity:
        # Empty generation loads exact local weights without generating a response.
        self.runtime.start()
        self._transition('RUNTIME_HEALTHY')
        payload = self.http.post('/api/generate', {'model': self.config.qualification.model_reference,
            'prompt': '', 'stream': False, 'keep_alive': '10m',
            'options': {'num_ctx': self.config.qualification.context_tokens}})
        if payload.get('done') is not True:
            raise ProbeFailure('invalid_response')
        models = self.http.get('/api/tags').get('models')
        exact = [m for m in models or () if isinstance(m, dict)
                 and m.get('name') == self.config.qualification.model_reference
                 and str(m.get('digest', '')).removeprefix('sha256:').lower()
                     == self.config.qualification.model_digest
                 and isinstance(m.get('details'), dict)
                 and m['details'].get('quantization_level') == self.config.qualification.quantization]
        if len(exact) != 1:
            raise ProbeFailure('assertion_failed')
        identity = self.observer()
        if (identity.artifact_sha256 != self.config.qualification.model_digest
                or identity.quantization != self.config.qualification.quantization
                or identity.context_tokens != self.config.qualification.context_tokens
                or identity.runtime.backend != 'cuda'):
            raise ProbeFailure('assertion_failed')
        return identity

    def _trust_or_requalify(self, identity: QualificationIdentity) -> bool:
        self._ready = False
        self._transition('VERIFYING')
        assessment = self.node.store.status(identity)
        self.qualification_state = assessment.state.value
        self.last_qualification_verification = self._now()
        if assessment.state == QualificationState.QUALIFIED:
            latest = self.node.store.latest(identity)
            self.qualification_id = str(latest.qualification_id) if latest else None
            self.observed_identity = identity
            self.qualification_state = 'qualified'
            self.runtime_healthy = True
            self._ready = True
            self._transition('QUALIFIED')
            self.audit.write('node0.qualification.verified', qualification_id=self.qualification_id,
                             identity=identity.fingerprint, reused=True)
            return True
        self._transition('REQUALIFYING', reason=assessment.state.value)
        self.audit.write('node0.qualification.requalification.started', reason=assessment.state.value)
        record = self.node.qualify()
        try:
            current = self.observer()
            after = self.node.store.status(current)
        except Exception:
            current, after = None, None
        if current is None or after is None or after.state != QualificationState.QUALIFIED:
            self.qualification_state = (after.state.value if after else 'unknown')
            self.last_failure = 'requalification_not_qualified'
            self._transition('FAILED', reason=self.last_failure)
            self._ready = False
            self.runtime_healthy = False
            self.audit.write('node0.qualification.requalification.completed', state=self.qualification_state)
            return False
        self.observed_identity = current
        self.qualification_state = 'qualified'
        self.qualification_id = str(record.qualification_id)
        self.last_qualification_verification = self._now()
        self.runtime_healthy = True
        self._ready = True
        self.last_failure = None
        self._transition('QUALIFIED')
        self.audit.write('node0.qualification.requalification.completed', state='qualified',
                         qualification_id=self.qualification_id)
        return True

    def start(self) -> bool:
        if self.state == 'QUALIFIED' and self._ready:
            return True
        self._transition('STARTING')
        self._ready = False
        if not self.lock.acquire():
            self.last_failure = 'duplicate_supervisor'
            self._transition('STOPPED', reason=self.last_failure)
            return False
        self._lock_acquired = True
        self.audit.write('node0.supervisor.start', node_uuid=str(self.config.node_uuid))
        try:
            self._transition('VERIFYING')
            identity = self._ensure_exact_runtime()
            self.runtime_healthy = True
            ready = self._trust_or_requalify(identity)
            self.last_health_check = self._now()
            if not ready:
                self.runtime.stop()
                self.runtime_healthy = False
            self._started_at = self.clock()
            self._persist_snapshot()
            return ready
        except Exception as exc:
            self.last_failure = _failure_code(exc)
            self.runtime_healthy = False
            self.qualification_state = 'unknown'
            self._ready = False
            self._transition('FAILED', reason=self.last_failure)
            self.audit.write('node0.supervisor.failed', error_type=type(exc).__name__)
            self._persist_snapshot()
            try:
                if self.runtime.process is not None:
                    self.runtime.stop()
            except Exception:
                self.audit.write('node0.runtime.stop.failed', error_type='cleanup_error')
            return False

    def health_check(self) -> bool:
        if not self.runtime.process or self.runtime.process.poll() is not None:
            raise ProbeFailure('unavailable')
        identity = self._ensure_exact_runtime()
        if not self._trust_or_requalify(identity):
            raise ProbeFailure('assertion_failed')
        return True

    def _restart_allowed(self) -> bool:
        now = self.clock()
        while self.restart_times and now - self.restart_times[0] > self.config.restart_window_s:
            self.restart_times.popleft()
        if self._started_at is not None and now - self._started_at >= self.config.stable_reset_s:
            self.restart_times.clear()
        return len(self.restart_times) < self.config.max_restarts

    def recover_once(self) -> bool:
        self._ready = False
        self.runtime_healthy = False
        self._transition('DEGRADED', reason=self.last_failure or 'runtime_health_failed')
        self.audit.write('node0.runtime.health_failure', reason_type=self.last_failure or 'unavailable')
        if not self._restart_allowed():
            self.last_failure = 'restart_limit_exceeded'
            self._transition('FAILED', reason=self.last_failure)
            self._persist_snapshot()
            return False
        attempt = len(self.restart_times)
        delay = min(self.config.max_backoff_s,
                    self.config.restart_backoff_s * (2 ** attempt))
        if self.stop_event.wait(delay):
            return False
        self.restart_times.append(self.clock())
        self.audit.write('node0.runtime.restart.started', attempt=len(self.restart_times))
        try:
            self.runtime.stop()
            self._transition('STARTING')
            identity = self._ensure_exact_runtime()
            self.runtime_healthy = True
            if not self._trust_or_requalify(identity):
                raise ProbeFailure('assertion_failed')
            self.last_failure = None
            self._started_at = self.clock()
            self.last_health_check = self._now()
            self.audit.write('node0.runtime.restart.completed', result='healthy')
            self._persist_snapshot()
            return True
        except Exception as exc:
            self.last_failure = _failure_code(exc)
            self._ready = False
            self.runtime_healthy = False
            if isinstance(exc, ProbeFailure) and exc.reason == 'assertion_failed':
                self.qualification_state = 'stale'
                self.qualification_id = None
                self.observed_identity = None
            elif self.qualification_state != 'qualified':
                self.qualification_state = 'unknown'
            self._transition('DEGRADED', reason=self.last_failure)
            self.audit.write('node0.runtime.restart.failed', error_type=type(exc).__name__)
            self._persist_snapshot()
            return False

    def serve(self) -> int:
        if not self.start():
            self.stop()
            return 2
        self.stop_event.clear()
        try:
            self._start_operator_services()
            while not self.stop_event.wait(self.config.health_interval_s):
                try:
                    self.health_check()
                    self.last_health_check = self._now()
                    self._persist_snapshot()
                except Exception as exc:
                    self.last_failure = _failure_code(exc)
                    if not self.recover_once() and self.state == 'FAILED':
                        # Keep the supervisor alive but fail closed, allowing bounded retries
                        # only after restart-window entries age out.
                        pass
        finally:
            self.stop()
        return 0

    def _start_operator_services(self) -> None:
        from dante.node0_control import Node0ControlServer, Node0ControlService
        from dante.workload import Node0InferenceExecutor, WorkloadOrchestrator, WorkloadStore

        path = self.config.workload_path or self.config.ledger_path.with_name('workloads.db')
        self.workload_store = WorkloadStore(path, audit=self.audit)
        executor = Node0InferenceExecutor(self, expected_model_id=self.config.model.model_id,
            expected_runtime_reference=self.config.qualification.model_reference,
            expected_digest=self.config.qualification.model_digest)
        self.orchestrator = WorkloadOrchestrator(self.workload_store, executor, max_gpu_jobs=1,
            worker_id='node0-' + str(self.config.node_uuid))
        self.orchestrator.start()
        self._orchestrator_thread = threading.Thread(target=self.orchestrator.serve,
            name='dante-node0-workload-orchestrator', daemon=True)
        self._orchestrator_thread.start()
        self.control_server = Node0ControlServer(Node0ControlService(self,
            workload_store=self.workload_store, orchestrator=self.orchestrator))
        try:
            self.control_server.start()
        except Exception:
            self.orchestrator.request_stop()
            self._orchestrator_thread.join(timeout=2)
            self.control_server = None
            self.orchestrator = None
            self._orchestrator_thread = None
            self.workload_store = None
            raise
        self.audit.write('node0.control.started', transport='windows_named_pipe',
                         workload_state='ready')

    def operator_status(self) -> dict:
        gate_accepted = False
        if self._ready and self.state == 'QUALIFIED' and self.runtime_healthy:
            try:
                self.node.registry.require_automatic(self.config.model)
                gate_accepted = True
            except (ValueError, ProbeFailure):
                gate_accepted = False
        snapshot = self.snapshot()
        continuity = self.node.continuity
        continuity_status = (continuity.read_only_status(
            context_tokens=self.config.qualification.context_tokens)
            if hasattr(continuity, 'read_only_status') else [])
        if not gate_accepted:
            for provider_state in continuity_status:
                if provider_state.get('provider') == self.config.model.provider_id:
                    provider_state['route_admissible'] = False
                    provider_state['denial_reason'] = 'qualification_or_machine'
        return {
            'state': snapshot.state,
            'runtime_state': snapshot.runtime_state,
            'runtime_version': snapshot.runtime_version,
            'runtime_pid': snapshot.runtime_pid,
            'model_reference': snapshot.model_reference,
            'model_digest': snapshot.expected_model_digest,
            'observed_model_digest': snapshot.observed_model_digest,
            'qualification_state': snapshot.qualification_state,
            'qualification_id': snapshot.qualification_id,
            'qualification_current': gate_accepted,
            'evidence_valid': gate_accepted,
            'gate_accepted': gate_accepted,
            'gpu_uuid': snapshot.gpu_identity,
            'driver_version': snapshot.driver_version,
            'ready_for_local_routing': bool(snapshot.ready_for_local_routing and gate_accepted),
            'last_health_check': snapshot.last_health_check,
            'updated_at': snapshot.updated_at,
            'last_failure': snapshot.last_failure,
            'continuity': continuity_status,
        }

    def operator_health(self) -> dict:
        status = self.operator_status()
        process = self.runtime.process
        alive = bool(process is not None and process.poll() is None)
        result = {
            'healthy': False,
            'qualification_id': status.get('qualification_id'),
            'qualification_current': status.get('qualification_current', False),
            'ready_for_local_routing': status.get('ready_for_local_routing', False),
            'runtime': 'ollama', 'runtime_version': None,
            'runtime_reachable': False, 'model_present': False,
            'model_reference': self.config.qualification.model_reference,
            'model_digest': self.config.qualification.model_digest,
            'gpu_uuid': status.get('gpu_uuid'), 'gpu_vram_bytes': 0,
            'gpu_execution_verified': False,
        }
        if not alive or not status.get('gate_accepted'):
            return result
        try:
            version = self.http.get('/api/version').get('version')
            tags = self.http.get('/api/tags').get('models')
            running = self.http.get('/api/ps').get('models')
            if not isinstance(tags, list) or not isinstance(running, list):
                return result
            ref = self.config.qualification.model_reference
            digest = self.config.qualification.model_digest
            exact = [item for item in tags if isinstance(item, dict)
                and item.get('name') == ref
                and str(item.get('digest', '')).removeprefix('sha256:').lower() == digest
                and isinstance(item.get('details'), dict)
                and item['details'].get('quantization_level') == self.config.qualification.quantization]
            resident = [item for item in running if isinstance(item, dict)
                and item.get('name') == ref and isinstance(item.get('size_vram'), int)]
            vram = max((item['size_vram'] for item in resident), default=0)
            result.update(runtime_version=version, runtime_reachable=version == status.get('runtime_version'),
                model_present=len(exact) == 1, gpu_vram_bytes=vram,
                gpu_execution_verified=bool(vram > 0 and status.get('gpu_uuid') == self.config.qualification.gpu_uuid))
            result['healthy'] = bool(alive and result['runtime_reachable'] and result['model_present']
                and result['qualification_current'] and result['ready_for_local_routing'])
        except Exception:
            return result
        return result

    def stop(self) -> None:
        if self.state == 'STOPPED' and not self._lock_acquired:
            return
        self._ready = False
        self._transition('STOPPING')
        self.stop_event.set()
        try:
            if self.control_server is not None:
                try:
                    self.control_server.stop(timeout_s=min(35, self.config.shutdown_timeout_s + 20))
                except Exception:
                    self.audit.write('node0.control.stop.failed')
                self.control_server = None
            if self.orchestrator is not None:
                try:
                    self.orchestrator.stop(drain_timeout_s=min(35, self.config.shutdown_timeout_s + 20))
                except Exception:
                    self.audit.write('node0.workload.stop.failed')
                if self._orchestrator_thread is not None:
                    self._orchestrator_thread.join(timeout=min(35, self.config.shutdown_timeout_s + 20))
                self.orchestrator = None
                self._orchestrator_thread = None
            stopped = self.runtime.stop()
            self.runtime_healthy = False
            self.qualification_state = 'unknown' if not stopped else self.qualification_state
            self._transition('STOPPED')
            self.audit.write('node0.supervisor.stop', runtime_stopped=stopped)
            self._persist_snapshot()
        finally:
            self.lock.release()
            self._lock_acquired = False

    def infer(self, prompt: str, *, model_reference: str | None = None,
              context_tokens: int | None = None, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
              thinking: ThinkingPolicy = ThinkingPolicy.OFF,
              cancellation=None):
        if model_reference is not None and model_reference != self.config.qualification.model_reference:
            raise ValueError('Requested model differs from the configured Node 0 model')
        if (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000
                or isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int)
                or not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS):
            raise ValueError('Invalid bounded inference request')
        try:
            thinking = ThinkingPolicy(thinking)
        except (ValueError, TypeError):
            raise ValueError('Invalid thinking policy') from None
        if context_tokens is None:
            context_tokens = self.config.qualification.context_tokens
        if (isinstance(context_tokens, bool) or not isinstance(context_tokens, int)
                or not 1 <= context_tokens <= self.config.qualification.context_tokens):
            raise ValueError('Requested context exceeds Node 0 qualification')
        if not self._inference_lock.acquire(blocking=False):
            raise Node0InferenceBusy('Node 0 inference is busy')
        task = None
        try:
            if not self._ready or self.state != 'QUALIFIED' or not self.runtime_healthy:
                raise ValueError('Node 0 is not ready for local routing')
            process = self.runtime.process
            if process is None or process.poll() is not None:
                raise ProbeFailure('unavailable')
            identity = self._routing_identity(self.config.model)
            if (identity.artifact_sha256 != self.config.qualification.model_digest
                    or identity.runtime.version != self.observed_identity.runtime.version
                    or not any(gpu.uuid == self.config.qualification.gpu_uuid for gpu in identity.machine.gpus or ())):
                raise ProbeFailure('assertion_failed')
            self.node.gate.current = self._routing_identity
            task = self.node.host.start('Node 0 supervised inference', '.', PrivacyClass.INTERNAL)
            context = inference_cancellation(cancellation) if cancellation is not None else nullcontext()
            with context:
                response = self.node.host.infer(task.task_id, prompt, set(),
                    required_context_tokens=context_tokens, max_output_tokens=max_output_tokens,
                    thinking=thinking)
            if not response.content.strip():
                raise ValueError('Local inference returned an empty response')
            self.node.host.complete(task.task_id)
            return response
        except Exception as exc:
            if task is not None:
                try:
                    from dante.contracts import TaskStatus
                    from dante.continuity import ContinuitySignal
                    current = self.node.ledger.get_task(task.task_id)
                    if current.status in {TaskStatus.RUNNING, TaskStatus.VERIFYING}:
                        terminal = isinstance(exc, ContinuitySignal) and exc.outcome.disposition.value == 'terminal'
                        self.node.ledger.transition(task.task_id,
                            TaskStatus.FAILED_TERMINAL if terminal else TaskStatus.FAILED_RETRYABLE,
                            current_step='inference-failed', metadata={'error_type': type(exc).__name__})
                except Exception:
                    pass
            raise
        finally:
            self._inference_lock.release()


def install_user_startup(config_path: Path, *, repository: Path) -> str:
    """Register a current-user logon task; it needs no elevation and is reversible."""
    if os.name != 'nt':
        raise OSError('Windows startup integration is only available on Windows')
    import subprocess
    pythonw = repository / '.venv' / 'Scripts' / 'pythonw.exe'
    entry = repository / 'scripts' / 'node0_supervisor.py'
    if not pythonw.is_file() or not entry.is_file() or not config_path.is_file():
        raise FileNotFoundError('Supervisor startup target is incomplete')
    task_name = 'Dante Node0 Supervisor'
    ps_quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    arguments = f'"{entry}" --config "{config_path}"'
    action = (f'New-ScheduledTaskAction -Execute {ps_quote(pythonw)} '
              f'-Argument {ps_quote(arguments)} -WorkingDirectory {ps_quote(repository)}')
    trigger = r'New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"'
    settings = ('New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew '
        '-RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) '
        '-ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable '
        '-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -DontStopOnIdleEnd')
    principal = r'New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited'
    script = f"$ErrorActionPreference='Stop'; Register-ScheduledTask -TaskName {ps_quote(task_name)} -Action ({action}) -Trigger ({trigger}) -Settings ({settings}) -Principal ({principal}) -Force | Out-Null"
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
        check=True, capture_output=True, timeout=30,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return task_name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Dante Node 0 fail-closed supervisor')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true', help='Startup simulation: verify then stop')
    parser.add_argument('--install-startup', action='store_true')
    parser.add_argument('--remove-startup', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.install_startup:
            SupervisorConfig.load(args.config)
            repo = Path(__file__).resolve().parent.parent
            print(json.dumps({'startup_task': install_user_startup(args.config.resolve(), repository=repo)}))
            return 0
        if args.remove_startup:
            if os.name != 'nt':
                raise OSError('Windows startup integration is only available on Windows')
            import subprocess
            subprocess.run(['schtasks.exe', '/Delete', '/TN', 'Dante Node0 Supervisor', '/F'],
                check=True, capture_output=True, timeout=20,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return 0
        supervisor = Node0Supervisor.from_file(args.config)
        if args.once:
            ready = supervisor.start()
            print(json.dumps(supervisor.snapshot().__dict__, sort_keys=True))
            supervisor.stop()
            return 0 if ready else 2
        return supervisor.serve()
    except Exception as exc:
        # Avoid leaking config values, environment, prompts or runtime response data.
        print(json.dumps({'state': 'FAILED', 'error_type': type(exc).__name__}))
        try:
            root = Path(os.environ['LOCALAPPDATA']) / 'DanteNode0'
            root.mkdir(parents=True, exist_ok=True)
            error_log = root / 'supervisor-startup-errors.jsonl'
            if error_log.exists() and error_log.stat().st_size > 256 * 1024:
                previous = root / 'supervisor-startup-errors.jsonl.1'
                previous.unlink(missing_ok=True)
                os.replace(error_log, previous)
            with error_log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'event': 'startup.failed', 'error_type': type(exc).__name__}) + '\n')
        except Exception:
            pass
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
