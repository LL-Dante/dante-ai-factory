"""Fresh, canonical Node 0 identity for exact native Ollama execution."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from dante.contracts.qualification import MachineProfile, QualificationIdentity, RuntimeObservation
from dante.nvidia_probe import discover_gpu_uuids
from dante.node_probe import probe_machine
from dante.ollama_qualification import (
    NvidiaProcessEvidence, OllamaQualificationConfig, OllamaQualificationHTTP,
    ProbeFailure, RuntimeControl,
)
from dante.recovery import digest


def _binary_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class OllamaIdentityObserver:
    def __init__(self, requested: QualificationIdentity, config: OllamaQualificationConfig,
                 runtime: RuntimeControl, http: OllamaQualificationHTTP,
                 nvidia: NvidiaProcessEvidence, *,
                 machine_probe: Callable | None = None):
        self.requested, self.config = requested, config
        self.runtime, self.http, self.nvidia = runtime, http, nvidia
        self.machine_probe = machine_probe or (lambda node_id: probe_machine(
            node_id, uuid_probe=discover_gpu_uuids))

    def _model(self) -> tuple[str | None, str | None]:
        models = self.http.get('/api/tags').get('models')
        if not isinstance(models, list):
            raise ProbeFailure('invalid_response')
        matches = [item for item in models if isinstance(item, dict)
                   and item.get('name') == self.config.model_reference]
        if len(matches) != 1:
            raise ProbeFailure('unavailable')
        item = matches[0]
        details = item.get('details')
        if not isinstance(details, dict) or details.get('format') != 'gguf':
            raise ProbeFailure('invalid_response')
        value = item.get('digest')
        model_digest = value.removeprefix('sha256:').lower() if isinstance(value, str) else None
        return model_digest, details.get('quantization_level')

    def _backend(self, model_digest: str | None, machine: MachineProfile) -> str | None:
        if (not any(gpu.uuid == self.config.gpu_uuid for gpu in machine.gpus or ())
                or not self.nvidia.correlated(self.runtime.pids())):
            return None
        models = self.http.get('/api/ps').get('models')
        if not isinstance(models, list):
            return None
        matches = [item for item in models if isinstance(item, dict)
                   and item.get('name') == self.config.model_reference
                   and isinstance(item.get('digest'), str)
                   and item['digest'].removeprefix('sha256:').lower() == model_digest
                   and isinstance(item.get('size_vram'), int) and item['size_vram'] > 0]
        return 'cuda' if len(matches) == 1 else None

    def __call__(self) -> QualificationIdentity:
        machine = MachineProfile.model_validate(self.machine_probe(self.requested.machine.node_id))
        if machine.node_id != self.requested.machine.node_id:
            raise ProbeFailure('assertion_failed')
        version = self.http.get('/api/version').get('version')
        if not isinstance(version, str) or not version:
            raise ProbeFailure('invalid_response')
        model_digest, quantization = self._model()
        binary_digest = _binary_digest(self.config.executable)
        runtime_config = digest({'profile': self.config.profile.model_dump(mode='json'),
            'owned': self.runtime.owned, 'binary': binary_digest,
            'model_store': str(self.config.model_store.resolve()) if self.config.model_store else None,
            'gpu_uuid': self.config.gpu_uuid, 'parallel': 1, 'no_cloud': True})
        configuration = digest({'suite': self.requested.suite_version,
            'model_reference': self.config.model_reference, 'model_digest': self.config.model_digest,
            'quantization': self.config.quantization, 'context_tokens': self.config.context_tokens,
            'stability_attempts': self.config.stability_attempts,
            'request_timeout_s': self.config.request_timeout_s})
        runtime = RuntimeObservation(runtime_id=self.requested.runtime.runtime_id,
            runtime='ollama', version=version,
            backend=self._backend(model_digest, machine), binary_sha256=binary_digest,
            configuration_sha256=runtime_config, probe_version='ollama-execution-v2')
        return QualificationIdentity(machine=machine, runtime=runtime,
            model_id=self.requested.model_id, artifact_kind='runtime_manifest',
            artifact_sha256=model_digest, quantization=quantization,
            context_tokens=self.config.context_tokens, configuration_sha256=configuration)

    def prepare(self) -> QualificationIdentity:
        """Bounded preflight to establish an observed CUDA identity before intent.

        The runner still repeats LOAD and retains its evidence. Failure during a
        later attempt uses the supplied requested identity to supersede old PASS.
        """
        self.runtime.start()
        digest_seen, quantization = self._model()
        if digest_seen != self.config.model_digest or quantization != self.config.quantization:
            raise ProbeFailure('assertion_failed')
        response = self.http.post('/api/generate', {'model': self.config.model_reference,
            'prompt': '', 'stream': False, 'keep_alive': '10m',
            'options': {'num_ctx': self.config.context_tokens}})
        if response.get('done') is not True:
            raise ProbeFailure('invalid_response')
        current = self()
        if current.runtime.backend != 'cuda':
            raise ProbeFailure('assertion_failed')
        return current
