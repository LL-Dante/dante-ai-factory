"""Opt-in native Ollama qualification. Inventory alone never proves execution.

Injected dependencies always mark a probe synthetic. The production constructor
uses only the bounded HTTP client, an owned Ollama process and NVIDIA process
telemetry; ambiguous attribution fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Callable, Protocol
from urllib.parse import urlsplit

import httpx

from dante.contracts import utc_now
from dante.contracts.qualification import Check, CheckEvidence, PerformanceEvidence, QualificationIdentity
from dante.contracts.runtime import RuntimeProfile
from dante.qualification_evidence import EvidenceDocument, EvidenceStore


class ProbeFailure(Exception):
    def __init__(self, reason: str):
        self.reason = reason


@dataclass(frozen=True)
class OllamaQualificationConfig:
    profile: RuntimeProfile
    model_reference: str
    model_digest: str
    quantization: str
    context_tokens: int
    gpu_uuid: str
    executable: Path | None = None
    model_store: Path | None = None
    stability_attempts: int = 3
    request_timeout_s: float = 20.0

    def __post_init__(self):
        if (self.profile.runtime != 'ollama' or self.profile.allow_remote
                or not self.profile.base_url.startswith('http://')):
            raise ValueError('Qualification requires a local HTTP Ollama endpoint')
        if (not self.model_reference or 'cloud' in self.model_reference.lower()
                or len(self.model_digest) != 64
                or any(c not in '0123456789abcdef' for c in self.model_digest)
                or not self.quantization or self.context_tokens < 64
                or not self.gpu_uuid.startswith('GPU-')
                or not 1 <= self.stability_attempts <= 10
                or not 0 < self.request_timeout_s <= 60):
            raise ValueError('Incomplete or unsafe qualification configuration')
        if (self.executable is None) != (self.model_store is None):
            raise ValueError('Owned runtime requires both executable and model store')


class RuntimeControl(Protocol):
    owned: bool
    def start(self) -> None: ...
    def stop(self) -> bool: ...
    def pids(self) -> frozenset[int]: ...


class ExternalOllamaRuntime:
    owned = False
    def start(self) -> None:
        pass
    def stop(self) -> bool:
        return False
    def pids(self) -> frozenset[int]:
        return frozenset()


class ManagedOllamaRuntime:
    """Owns only the Popen handle it created; never kills a discovered PID."""
    owned = True

    def __init__(self, config: OllamaQualificationConfig):
        if config.executable is None or config.model_store is None:
            raise ValueError('Managed runtime requires explicit executable and model store')
        self.config = config
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                return  # The same owned Popen handle is already running.
            self.process = None
        executable = self.config.executable.resolve(strict=True)
        if executable.name.lower() != 'ollama.exe':
            raise ProbeFailure('probe_error')
        store = self.config.model_store.resolve(strict=True)
        parsed = urlsplit(self.config.profile.base_url)
        try:
            with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=0.2):
                raise ProbeFailure('unavailable')  # An existing server is not ours.
        except (ConnectionRefusedError, TimeoutError, OSError):
            pass
        env = os.environ.copy()
        env.update({'OLLAMA_HOST': self.config.profile.base_url.removeprefix('http://'),
                    'OLLAMA_MODELS': str(store), 'OLLAMA_NO_CLOUD': '1',
                    'OLLAMA_NUM_PARALLEL': '1'})
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        self.process = subprocess.Popen([str(executable), 'serve'], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False,
            env=env, creationflags=flags)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.process = None
                raise ProbeFailure('unavailable')
            try:
                with httpx.Client(trust_env=False, follow_redirects=False,
                                  timeout=httpx.Timeout(0.5)) as client:
                    response = client.get(self.config.profile.base_url + '/api/version')
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        self.stop()
        raise ProbeFailure('timeout')

    def stop(self) -> bool:
        process = self.process
        if process is None:
            return False
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.process = None
        return True

    def pids(self) -> frozenset[int]:
        process = self.process
        if process is None or process.poll() is not None:
            return frozenset()
        pids = {process.pid}
        if os.name != 'nt':
            return frozenset(pids)
        # Ollama may execute its GPU runner as a child process on Windows.
        script = ('Get-CimInstance Win32_Process | '
                  'Select-Object ProcessId,ParentProcessId,ExecutablePath | ConvertTo-Json -Compress')
        try:
            result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                capture_output=True, timeout=5, check=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if len(result.stdout) > 512 * 1024:
                return frozenset(pids)
            rows = json.loads(result.stdout)
            rows = rows if isinstance(rows, list) else [rows]
            expected = str(self.config.executable.resolve()).casefold()
            changed = True
            while changed:
                before = len(pids)
                for row in rows:
                    if (row.get('ParentProcessId') in pids and isinstance(row.get('ProcessId'), int)
                            and isinstance(row.get('ExecutablePath'), str)
                            and row['ExecutablePath'].casefold() == expected):
                        pids.add(row['ProcessId'])
                changed = len(pids) != before
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            pass
        return frozenset(pids)


class OllamaQualificationHTTP:
    def __init__(self, config: OllamaQualificationConfig, *, transport=None):
        self.config, self.transport = config, transport

    def _client(self, timeout_s: float):
        return httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(connect=min(2.0, timeout_s), read=timeout_s,
                                  write=min(2.0, timeout_s), pool=min(2.0, timeout_s)))

    def _request(self, method: str, path: str, body: dict | None, timeout_s: float) -> dict:
        try:
            with self._client(timeout_s) as client:
                with client.stream(method, self.config.profile.base_url + path,
                                   **({} if body is None else {'json': body})) as response:
                    if response.status_code != 200:
                        raise ProbeFailure('unavailable' if response.status_code >= 500 else 'invalid_response')
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > self.config.profile.max_response_bytes:
                            raise ProbeFailure('invalid_response')
            payload = json.loads(raw)
            if not isinstance(payload, dict) or 'error' in payload:
                raise ProbeFailure('invalid_response')
            return payload
        except httpx.TimeoutException:
            raise ProbeFailure('timeout') from None
        except httpx.TransportError:
            raise ProbeFailure('unavailable') from None
        except (ValueError, UnicodeError):
            raise ProbeFailure('invalid_response') from None

    def get(self, path: str) -> dict:
        return self._request('GET', path, None, self.config.request_timeout_s)

    def post(self, path: str, body: dict, *, timeout_s: float | None = None) -> dict:
        return self._request('POST', path, body, timeout_s or self.config.request_timeout_s)

    def cancel_stream(self, path: str, body: dict) -> bool:
        try:
            with self._client(self.config.request_timeout_s) as client:
                with client.stream('POST', self.config.profile.base_url + path, json=body) as response:
                    if response.status_code != 200:
                        raise ProbeFailure('invalid_response')
                    chunks = response.iter_bytes()
                    first = next(chunks, b'')
                    if not first or len(first) > self.config.profile.max_response_bytes:
                        raise ProbeFailure('invalid_response')
                return response.is_closed
        except httpx.TimeoutException:
            raise ProbeFailure('timeout') from None
        except httpx.TransportError:
            raise ProbeFailure('unavailable') from None

    def expect_timeout(self, path: str, body: dict) -> bool:
        try:
            self.post(path, body, timeout_s=0.001)
        except ProbeFailure as failure:
            if failure.reason == 'timeout':
                return True
            raise
        return False


class NvidiaProcessEvidence:
    """Require an exact NVIDIA GPU UUID and an owned Ollama process PID."""
    def __init__(self, config: OllamaQualificationConfig):
        self.config = config

    def correlated(self, pids: frozenset[int]) -> bool:
        if not pids:
            return False
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                '--format=csv,noheader,nounits'], capture_output=True, timeout=5, check=False,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode != 0 or len(result.stdout) > 256 * 1024:
                return False
            for line in result.stdout.decode('utf-8', errors='replace').splitlines():
                fields = [part.strip() for part in line.split(',')]
                if (len(fields) == 2 and fields[0].isascii() and fields[0].isdecimal()
                        and int(fields[0]) in pids and fields[1] == self.config.gpu_uuid):
                    return True
        except (OSError, subprocess.SubprocessError):
            pass
        return False


class OllamaQualificationProbe:
    def __init__(self, identity: QualificationIdentity, config: OllamaQualificationConfig,
                 evidence: EvidenceStore, runtime: RuntimeControl, http: OllamaQualificationHTTP,
                 nvidia: NvidiaProcessEvidence, *, synthetic: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = utc_now,
                 observer: Callable[[], QualificationIdentity] | None = None):
        self.identity, self.config, self.evidence = identity, config, evidence
        self.runtime, self.http, self.nvidia = runtime, http, nvidia
        self.source = 'synthetic' if synthetic else 'hardware'
        self.clock, self.now = clock, now
        self.observer = observer
        self.attempt_id = None
        self.deadline_s = None
        self._load_s: float | None = None
        self._peak_vram: int | None = None
        self._last_observation = identity

    def observe(self) -> QualificationIdentity:
        return self.observer() if self.observer is not None else self._last_observation

    def begin_attempt(self, attempt_id, deadline_s: float) -> None:
        self.attempt_id, self.deadline_s = attempt_id, deadline_s

    def _exact_inventory(self) -> None:
        version = self.http.get('/api/version').get('version')
        if version != self.identity.runtime.version:
            raise ProbeFailure('assertion_failed')
        models = self.http.get('/api/tags').get('models')
        if not isinstance(models, list):
            raise ProbeFailure('invalid_response')
        matches = [item for item in models if isinstance(item, dict)
                   and item.get('name') == self.config.model_reference]
        if len(matches) != 1:
            raise ProbeFailure('unavailable')
        entry = matches[0]
        details = entry.get('details')
        if (not isinstance(details, dict) or details.get('format') != 'gguf'
                or entry.get('digest', '').removeprefix('sha256:') != self.config.model_digest
                or details.get('quantization_level') != self.config.quantization
                or entry.get('remote_host') or entry.get('remote_model')):
            raise ProbeFailure('assertion_failed')

    def _resident(self, *, context: bool = False) -> int:
        models = self.http.get('/api/ps').get('models')
        if not isinstance(models, list):
            raise ProbeFailure('invalid_response')
        matches = [item for item in models if isinstance(item, dict)
                   and item.get('name') == self.config.model_reference
                   and isinstance(item.get('digest'), str)
                   and item['digest'].removeprefix('sha256:') == self.config.model_digest]
        if (len(matches) != 1 or not isinstance(matches[0].get('size_vram'), int)
                or matches[0]['size_vram'] <= 0):
            raise ProbeFailure('assertion_failed')
        if context and (not isinstance(matches[0].get('context_length'), int)
                        or matches[0]['context_length'] < self.config.context_tokens):
            raise ProbeFailure('assertion_failed')
        if not self.nvidia.correlated(self.runtime.pids()):
            raise ProbeFailure('assertion_failed')
        self._peak_vram = max(self._peak_vram or 0, matches[0]['size_vram'])
        return matches[0]['size_vram']

    def _body(self, prompt: str, *, stream: bool = False, predict: int = 32) -> dict:
        return {'model': self.config.model_reference, 'stream': stream,
                'messages': [{'role': 'user', 'content': prompt}], 'keep_alive': '10m',
                'options': {'num_ctx': self.config.context_tokens, 'num_predict': predict,
                            'temperature': 0}}

    def _generate(self, prompt: str = 'Reply with one short word.') -> dict:
        payload = self.http.post('/api/chat', self._body(prompt))
        message = payload.get('message')
        if (payload.get('done') is not True or payload.get('model') != self.config.model_reference
                or not isinstance(message, dict) or message.get('role') != 'assistant'
                or not isinstance(message.get('content'), str) or not message['content'].strip()):
            raise ProbeFailure('invalid_response')
        return payload

    def _health(self) -> None:
        if self.http.get('/api/version').get('version') != self.identity.runtime.version:
            raise ProbeFailure('assertion_failed')

    def _check(self, check: Check) -> dict[str, str | int | float | bool | None]:
        if check == Check.LOAD:
            self.runtime.start()
            self._exact_inventory()
            loaded = self.http.post('/api/generate', {'model': self.config.model_reference,
                'prompt': '', 'stream': False, 'keep_alive': '10m',
                'options': {'num_ctx': self.config.context_tokens}})
            if loaded.get('done') is not True:
                raise ProbeFailure('invalid_response')
            vram = self._resident()
            return {'vram_bytes': vram, 'gpu_correlated': True}
        if check == Check.GENERATION:
            payload = self._generate()
            self._resident()
            return {'output_tokens': payload.get('eval_count') if isinstance(payload.get('eval_count'), int) else None}
        if check == Check.CONTEXT:
            payload = self._generate('Context qualification: ' + 'context ' * 64)
            count = payload.get('prompt_eval_count')
            if not isinstance(count, int) or count < 16:
                raise ProbeFailure('assertion_failed')
            self._resident(context=True)
            return {'context_tokens': self.config.context_tokens, 'prompt_eval_count': count}
        if check == Check.CANCELLATION:
            if not self.http.cancel_stream('/api/chat', self._body('Count upward indefinitely.', stream=True, predict=4096)):
                raise ProbeFailure('assertion_failed')
            self._generate()
            return {'stream_cancelled': True, 'inference_responsive': True}
        if check == Check.TIMEOUT:
            if not self.http.expect_timeout('/api/chat', self._body('Count upward.', predict=4096)):
                raise ProbeFailure('assertion_failed')
            self._generate()
            return {'client_timeout_observed': True, 'inference_responsive': True}
        if check == Check.RUNTIME_FAILURE:
            if not self.runtime.owned or not self.runtime.stop():
                raise ProbeFailure('unavailable')
            try:
                self._health()
            except ProbeFailure as failure:
                if failure.reason == 'unavailable':
                    return {'owned_process_stopped': True, 'typed_failure': 'unavailable'}
                raise
            raise ProbeFailure('assertion_failed')
        if check == Check.RECOVERY:
            if not self.runtime.owned:
                raise ProbeFailure('unavailable')
            self.runtime.start()
            self._exact_inventory()
            self._generate()
            self._resident()
            return {'runtime_restarted': True, 'gpu_correlated': True}
        if check == Check.STABILITY:
            began = self.clock()
            worst = 0.0
            for _ in range(self.config.stability_attempts):
                call_started = self.clock()
                self._generate()
                latency = self.clock() - call_started
                worst = max(worst, latency)
                if latency > self.config.request_timeout_s or self.clock() - began > 90:
                    raise ProbeFailure('timeout')
            self._resident()
            return {'attempts': self.config.stability_attempts, 'failures': 0,
                    'max_latency_s': worst, 'budget_s': 90}
        raise ProbeFailure('unavailable')

    def run(self, check: Check) -> CheckEvidence:
        started, began = self.now(), self.clock()
        facts: dict[str, str | int | float | bool | None] = {}
        failure = None
        outcome = 'passed'
        if check in {Check.STRUCTURED_OUTPUT, Check.TOOL_CALLING}:
            outcome = 'unsupported'
        else:
            try:
                facts = self._check(check)
            except ProbeFailure as exc:
                outcome, failure = 'failed', exc.reason
            except Exception:
                outcome, failure = 'failed', 'probe_error'
        duration = max(0.0, self.clock() - began)
        if check == Check.LOAD and outcome == 'passed':
            self._load_s = duration
        document = EvidenceDocument(identity_fingerprint=self.identity.fingerprint,
            node_id=str(self.identity.machine.node_id), check=check, outcome=outcome,
            attempt_id=self.attempt_id, started_at=started, completed_at=self.now(),
            deadline_s=self.deadline_s, gpu_uuid=self.config.gpu_uuid,
            gpu_name=next((gpu.name for gpu in self.identity.machine.gpus or ()
                           if gpu.uuid == self.config.gpu_uuid), None),
            gpu_driver=next((gpu.driver_version for gpu in self.identity.machine.gpus or ()
                             if gpu.uuid == self.config.gpu_uuid), None),
            runtime_id=self.identity.runtime.runtime_id,
            runtime_version=self.identity.runtime.version,
            runtime_configuration_sha256=self.identity.runtime.configuration_sha256,
            model_reference=self.config.model_reference,
            model_digest=self.config.model_digest,
            quantization=self.config.quantization, context_tokens=self.config.context_tokens,
            qualification_configuration_sha256=self.identity.configuration_sha256,
            facts=facts, failure=failure)
        key = self.evidence.put(document)
        return CheckEvidence(check=check, outcome=outcome, evidence_sha256=key,
                             failure=failure, duration_s=duration)

    def performance(self) -> PerformanceEvidence:
        return PerformanceEvidence(peak_vram_bytes=self._peak_vram, load_time_s=self._load_s)


class OllamaQualificationProbeFactory:
    """Production defaults are real; any injected test seam makes evidence synthetic."""
    def __init__(self, config: OllamaQualificationConfig, evidence: EvidenceStore, *,
                 runtime_factory: Callable[[OllamaQualificationConfig], RuntimeControl] | None = None,
                 http_factory: Callable[[OllamaQualificationConfig], OllamaQualificationHTTP] | None = None,
                 nvidia_factory: Callable[[OllamaQualificationConfig], NvidiaProcessEvidence] | None = None,
                 observer_factory: Callable[..., Callable[[], QualificationIdentity]] | None = None,
                 clock: Callable[[], float] | None = None,
                 now: Callable[[], datetime] | None = None):
        self.config, self.evidence = config, evidence
        self.runtime_factory = runtime_factory or (lambda cfg: ManagedOllamaRuntime(cfg)
            if cfg.executable is not None else ExternalOllamaRuntime())
        self.http_factory = http_factory or OllamaQualificationHTTP
        self.nvidia_factory = nvidia_factory or NvidiaProcessEvidence
        self.observer_factory = observer_factory
        self.clock, self.now = clock, now
        self.synthetic = any(item is not None for item in
            (runtime_factory, http_factory, nvidia_factory, clock, now))

    def __call__(self, identity: QualificationIdentity) -> OllamaQualificationProbe:
        if (identity.runtime.runtime != 'ollama'
                or identity.artifact_sha256 != self.config.model_digest
                or identity.context_tokens != self.config.context_tokens
                or identity.quantization != self.config.quantization):
            raise ValueError('Requested identity differs from exact Ollama configuration')
        options = {}
        if self.clock is not None:
            options['clock'] = self.clock
        if self.now is not None:
            options['now'] = self.now
        runtime = self.runtime_factory(self.config)
        http = self.http_factory(self.config)
        nvidia = self.nvidia_factory(self.config)
        if self.observer_factory is not None:
            options['observer'] = self.observer_factory(identity, self.config, runtime, http, nvidia)
        elif not self.synthetic:
            from dante.ollama_identity import OllamaIdentityObserver
            options['observer'] = OllamaIdentityObserver(identity, self.config, runtime, http, nvidia)
        return OllamaQualificationProbe(identity, self.config, self.evidence,
            runtime, http, nvidia, synthetic=self.synthetic, **options)

    def prepare(self, requested: QualificationIdentity) -> tuple[QualificationIdentity, OllamaQualificationProbe]:
        probe = self(requested)
        prepare = getattr(probe.observer, 'prepare', None)
        if prepare is None:
            raise ValueError('A fresh production observer is required')
        current = prepare()
        probe.identity = current
        probe.observer.requested = current
        return current, probe
