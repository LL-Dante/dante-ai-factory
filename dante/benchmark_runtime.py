"""Bounded access to the isolated benchmark runtime.

This module is the only route to the benchmark runtime. Metadata is read through
the ordinary read-only adapter path; generation adds only what measurement needs:
explicit options, `think=False`, and streaming so time-to-first-token is observed
rather than guessed. Nothing here can reach the qualified production runtime.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from dante.contracts.benchmarks import BenchmarkParameters
from dante.contracts.runtime import RuntimeProfile
from dante.local_runtime import OllamaAdapter

RUNTIME = 'ollama'
_SAFE_MODEL = re.compile(r'^[A-Za-z0-9._:/-]{1,128}$')
_SAFE_ENDPOINT = re.compile(r'^https?://[A-Za-z0-9._:-]{1,180}$')
_MAX_RESPONSE = 8 * 1024 * 1024
_MAX_CHUNKS = 20000


class BenchmarkRuntimeError(RuntimeError):
    """A benchmark runtime call failed. Carries a stable, reportable reason."""


@dataclass(frozen=True)
class RawGeneration:
    """Exactly what the runtime reported, with nothing inferred."""

    response: str
    thinking: str | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    prompt_eval_duration_ns: int | None
    eval_count: int | None
    eval_duration_ns: int | None
    total_duration_ns: int | None
    done_reason: str | None
    time_to_first_token_s: float | None
    wall_clock_s: float
    chunk_count: int
    source: str = 'ollama-generate'


def _as_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _as_str(value) -> str | None:
    return value if isinstance(value, str) and value else None


class BenchmarkRuntime:
    """One benchmark endpoint. Rejects a malformed endpoint and a bare port."""

    def __init__(self, endpoint: str, *, timeout_s: float = 300.0, transport=None,
                 adapter_factory=OllamaAdapter, sleep: Callable[[float], None] = time.sleep):
        if not _SAFE_ENDPOINT.fullmatch(endpoint or ''):
            raise BenchmarkRuntimeError('Benchmark endpoint is not an http(s) URL')
        self.endpoint = endpoint.rstrip('/')
        self.timeout_s = timeout_s
        self.transport = transport
        self._adapter_factory = adapter_factory
        self._sleep = sleep
        self._adapter = adapter_factory(RuntimeProfile(runtime=RUNTIME, base_url=self.endpoint))

    def version(self) -> str | None:
        try:
            return _as_str(self._adapter.version())
        except Exception as error:
            raise BenchmarkRuntimeError(f'{type(error).__name__}: runtime version unavailable') from None

    def residency(self) -> tuple[dict, ...]:
        """Normalize resident models to name, vram bytes and context."""
        try:
            entries = list(self._adapter.loaded())
        except Exception as error:
            raise BenchmarkRuntimeError(f'{type(error).__name__}: residency unavailable') from None
        normalized = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = _as_str(entry.get('name') or entry.get('id'))
            if not name:
                continue
            details = entry.get('details') if isinstance(entry.get('details'), dict) else {}
            normalized.append({
                'name': name,
                'digest': _as_str(entry.get('digest')) or '',
                'size_vram': _as_int(entry.get('size_vram')) or 0,
                'context_length': _as_int(entry.get('context_length')) or 0,
                'quantization': _as_str(details.get('quantization_level')),
                'parameter_size': _as_str(details.get('parameter_size')),
            })
        return tuple(normalized)

    def installed(self) -> tuple[dict, ...]:
        """Every model the store already holds. Proves presence without loading."""
        try:
            entries = list(self._adapter.inventory())
        except Exception as error:
            raise BenchmarkRuntimeError(f'{type(error).__name__}: installed model list unavailable') from None
        normalized = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = _as_str(entry.get('name') or entry.get('model') or entry.get('id'))
            digest = _as_str(entry.get('digest'))
            size = _as_int(entry.get('size'))
            if not name or not digest or len(digest) != 64 or size is None:
                continue
            details = entry.get('details') if isinstance(entry.get('details'), dict) else {}
            normalized.append({
                'name': name,
                'digest': digest.lower(),
                'size': size,
                'family': _as_str(details.get('family')),
                'quantization': _as_str(details.get('quantization_level')),
                'parameter_size': _as_str(details.get('parameter_size')),
                'parent_model': _as_str(details.get('parent_model')),
                'context_length': _as_int(details.get('context_length')),
                'capabilities': tuple(sorted(str(item) for item in (details.get('capabilities') or ())
                                             if isinstance(item, str))),
            })
        return tuple(sorted(normalized, key=lambda item: item['name']))

    def block_count(self, model_id: str) -> int | None:
        """Transformer block count, used to size a partial GPU offload."""
        shown = self.show(model_id)
        model_info = shown.get('model_info') if isinstance(shown.get('model_info'), dict) else {}
        for key, value in model_info.items():
            if key.endswith('.block_count'):
                return _as_int(value)
        return None

    def show(self, model_id: str) -> dict:
        if not _SAFE_MODEL.fullmatch(model_id or ''):
            raise BenchmarkRuntimeError('Model reference contains unsupported characters')
        try:
            shown = self._adapter.show(model_id)
        except Exception as error:
            raise BenchmarkRuntimeError(f'{type(error).__name__}: model metadata unavailable') from None
        return dict(shown) if isinstance(shown, dict) else {}

    def _request_body(self, model_id: str, *, prompt: str | None,
                      parameters: BenchmarkParameters | None, keep_alive) -> dict:
        if not _SAFE_MODEL.fullmatch(model_id or ''):
            raise BenchmarkRuntimeError('Model reference contains unsupported characters')
        body: dict = {'model': model_id, 'stream': True, 'think': False}
        if prompt is not None:
            body['prompt'] = prompt
        if keep_alive is not None:
            body['keep_alive'] = keep_alive
        if parameters is not None:
            options: dict = {
                'temperature': parameters.temperature,
                'seed': parameters.seed,
                'num_predict': parameters.output_token_budget,
                'num_ctx': parameters.context_token_budget,
            }
            if parameters.num_gpu_layers is not None:
                options['num_gpu'] = parameters.num_gpu_layers
            body['options'] = options
            if parameters.structured_output:
                body['format'] = 'json'
        return body

    def generate(self, model_id: str, *, prompt: str, parameters: BenchmarkParameters,
                 keep_alive=None) -> RawGeneration:
        """Stream one completion, recording first-token latency and raw counters."""
        body = self._request_body(model_id, prompt=prompt, parameters=parameters, keep_alive=keep_alive)
        started = time.perf_counter()
        pieces: list[str] = []
        thinking: str | None = None
        ttft: float | None = None
        final: dict = {}
        chunks = 0
        try:
            with httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
                              timeout=self.timeout_s) as client:
                with client.stream('POST', f'{self.endpoint}/api/generate', json=body) as response:
                    if response.status_code != 200:
                        raise BenchmarkRuntimeError(
                            f'Benchmark runtime rejected the request with status {response.status_code}')
                    raw = bytearray()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        raw.extend(line)
                        if len(raw) > _MAX_RESPONSE or chunks >= _MAX_CHUNKS:
                            raise BenchmarkRuntimeError('Benchmark response exceeded the streaming limit')
                        chunks += 1
                        try:
                            message = json.loads(line)
                        except ValueError:
                            raise BenchmarkRuntimeError('Benchmark runtime emitted malformed JSON') from None
                        if not isinstance(message, dict):
                            raise BenchmarkRuntimeError('Benchmark runtime emitted a non-object chunk')
                        if isinstance(message.get('error'), str) and message['error']:
                            raise BenchmarkRuntimeError('Benchmark runtime returned an error envelope')
                        text = message.get('response')
                        if isinstance(text, str) and text:
                            if ttft is None:
                                ttft = time.perf_counter() - started
                            pieces.append(text)
                        thought = message.get('thinking')
                        if isinstance(thought, str) and thought:
                            thinking = thought
                        if message.get('done') is True:
                            final = message
        except httpx.TimeoutException:
            raise BenchmarkRuntimeError('Benchmark runtime timed out') from None
        except httpx.TransportError as error:
            raise BenchmarkRuntimeError(f'{type(error).__name__}: benchmark runtime unavailable') from None
        wall = time.perf_counter() - started
        if not final:
            raise BenchmarkRuntimeError('Benchmark stream ended without a final message')
        return RawGeneration(
            response=''.join(pieces), thinking=thinking,
            load_duration_ns=_as_int(final.get('load_duration')),
            prompt_eval_count=_as_int(final.get('prompt_eval_count')),
            prompt_eval_duration_ns=_as_int(final.get('prompt_eval_duration')),
            eval_count=_as_int(final.get('eval_count')),
            eval_duration_ns=_as_int(final.get('eval_duration')),
            total_duration_ns=_as_int(final.get('total_duration')),
            done_reason=_as_str(final.get('done_reason')),
            time_to_first_token_s=ttft, wall_clock_s=wall, chunk_count=chunks)

    def unload(self, model_id: str) -> None:
        """Release residency on the benchmark runtime, then confirm it is gone."""
        self._request_body(model_id, prompt=None, parameters=None, keep_alive=0)
        try:
            with httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
                              timeout=self.timeout_s) as client:
                with client.stream('POST', f'{self.endpoint}/api/generate', json={
                    'model': model_id, 'keep_alive': 0, 'stream': False}) as response:
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > _MAX_RESPONSE:
                            raise BenchmarkRuntimeError('Unload response exceeded the size limit')
                    if response.status_code != 200:
                        raise BenchmarkRuntimeError(
                            f'Benchmark runtime refused unload with status {response.status_code}')
        except httpx.TimeoutException:
            raise BenchmarkRuntimeError('Benchmark runtime timed out during unload') from None
        except httpx.TransportError as error:
            raise BenchmarkRuntimeError(f'{type(error).__name__}: unload failed') from None
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            if not any(entry['name'] == model_id for entry in self.residency()):
                return
            self._sleep(0.5)
        raise BenchmarkRuntimeError('Model is still resident after an unload request')


@dataclass
class SampledGpu:
    """Sensor readings taken while a generation was in flight."""

    samples: int = 0
    util_peak: float | None = None
    util_mean: float | None = None
    temperature_peak: float | None = None
    power_peak: float | None = None
    free_vram_min_bytes: int | None = None
    used_vram_max_bytes: int | None = None
    source: str = field(default='nvidia-smi-sampled')


__all__ = ['BenchmarkRuntime', 'BenchmarkRuntimeError', 'RawGeneration', 'SampledGpu']
