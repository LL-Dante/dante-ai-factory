"""Read-only model discovery.

Enumerates models that already exist on every known local Ollama runtime using
metadata endpoints only (`/api/tags`, `/api/ps`, `/api/show`). Nothing here pulls,
copies, deletes, loads, quantizes or executes a model, and no cloud or network
destination outside the configured local endpoints is contacted.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from dante.contracts.hardware import (Fact, HardwareSnapshot, VOLATILE_FIELDS,
                                      VOLATILE_GROUP_FIELDS, _without_timestamps)
from dante.contracts.models import (BenchmarkClass, BenchmarkPlan, ModelFitFacts, ModelRecord,
                                    ModelRegistry, RoleCategory, ScoutSnapshot)
from dante.contracts.runtime import RuntimeProfile
from dante.local_runtime import OllamaAdapter

RUNTIME = 'ollama'
PROVIDER = 'ollama'
TAGS = 'ollama-tags'
PS = 'ollama-ps'
SHOW = 'ollama-show'
HARDWARE = 'hardware-snapshot'
FIT = 'derived-bytes-comparison'


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and len(value) <= 200 and all(ord(c) >= 32 for c in value) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdecimal():
            return int(text)
    return None


def _model_info(payload: dict, *suffixes: str) -> int | None:
    """Read a scalar out of Ollama's architecture-prefixed model_info map."""
    info = payload.get('model_info')
    if not isinstance(info, dict):
        return None
    for key, value in info.items():
        if isinstance(key, str) and key.endswith(suffixes) and not key.startswith('tokenizer'):
            number = _int(value)
            if number is not None:
                return number
    return None


def _capabilities(payload: dict, entry: dict) -> tuple[str, ...]:
    declared = payload.get('capabilities')
    if not isinstance(declared, list) or any(not isinstance(v, str) for v in declared):
        declared = entry.get('capabilities')
    if not isinstance(declared, list) or any(not isinstance(v, str) for v in declared):
        return ()
    return tuple(sorted({v.strip() for v in declared if v.strip()}))


def _fact(value, *, source: str, unit: str | None = None, reason: str) -> Fact:
    if value is None:
        return Fact.unknown(reason, source=source)
    if isinstance(value, (list, tuple)):
        value = ','.join(str(item) for item in value)
        if not value:
            return Fact.unknown(reason, source=source)
    return Fact.measured(value, source=source, unit=unit)


class _RuntimeView:
    """Everything one endpoint reported, already normalized."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.version: str | None = None
        self.reachable = False
        self.tags: list[dict] = []
        self.resident: list[dict] = []
        self.shown: dict[str, dict] = {}
        self.reason: str | None = None


def _observe(endpoint: str, adapter_factory=None) -> _RuntimeView:
    view = _RuntimeView(endpoint)
    adapter = (adapter_factory or OllamaAdapter)(RuntimeProfile(runtime=RUNTIME, base_url=endpoint))
    try:
        view.version = _text(adapter.version())
        view.tags = list(adapter.inventory())
        view.reachable = True
    except Exception as error:
        view.reason = f'{type(error).__name__}: runtime metadata is unavailable'
        return view
    try:
        view.resident = list(adapter.loaded())
    except Exception as error:
        view.reason = f'{type(error).__name__}: resident model view is unavailable'
    by_name = {}
    for entry in view.resident:
        name = _text(entry.get('name') or entry.get('id'))
        if name:
            by_name[name] = entry
    for entry in view.tags:
        name = _text(entry.get('name') or entry.get('id'))
        if not name or name in view.shown:
            continue
        try:
            view.shown[name] = dict(adapter.show(name))
        except Exception as error:
            view.shown[name] = {'__unavailable__': f'{type(error).__name__}'}
    return view


def _record(model_id: str, observations: Sequence[tuple[_RuntimeView, dict]],
            qualified_endpoint: str | None, qualified_model: str | None,
            qualified_id: str | None) -> ModelRecord:
    """Merge every runtime's view of one model into a single normalized record."""
    first_view, first_entry = observations[0]
    shown = first_view.shown.get(model_id) or {}
    entry = first_entry
    for candidate_view, candidate_entry in observations:
        if candidate_view.shown.get(model_id) and not shown:
            shown, entry = candidate_view.shown[model_id], candidate_entry
            break
    if '__unavailable__' in shown:
        shown = {}

    details = entry.get('details') if isinstance(entry.get('details'), dict) else {}
    if not details and isinstance(shown.get('details'), dict):
        details = shown['details']
    resident = None
    for candidate_view, _ in observations:
        resident = next((item for item in candidate_view.resident
                         if _text(item.get('name') or item.get('id')) == model_id), None)
        if resident is not None:
            break

    digest = _text(entry.get('digest'))
    architecture = _text(shown.get('model_info', {}).get('general.architecture')
                         if isinstance(shown.get('model_info'), dict) else None) or _text(details.get('family'))
    quantization = _text(details.get('quantization_level'))
    file_format = _text(details.get('format'))
    size = _int(entry.get('size'))
    parameters = _model_info(shown, 'general.parameter_count')
    label = _text(details.get('parameter_size'))
    native_context = _int(details.get('context_length')) or _model_info(shown, '.context_length')
    capabilities = _capabilities(shown, entry)
    parent = _text(details.get('parent_model'))
    modified = _text(entry.get('modified_at'))
    loaded_vram = _int(resident.get('size_vram')) if resident else None
    expires = _text(resident.get('expires_at')) if resident else None
    configured_context = _int(resident.get('context_length')) if resident else None

    endpoints = tuple(sorted({item.endpoint for item, _ in observations}))
    loaded_endpoints = tuple(sorted(
        {item.endpoint for item, _ in observations
         if any(_text(entry.get('name') or entry.get('id')) == model_id for entry in item.resident)}))
    is_qualified = (qualified_model == model_id and qualified_endpoint is not None
                    and qualified_endpoint in endpoints)
    qualification = (Fact.measured(qualified_id, source='node0-qualification')
                     if is_qualified and qualified_id
                     else Fact.unknown(
                         'This model is not the qualified Node0 model on the qualified runtime',
                         source='node0-qualification'))

    presence: list[str] = ['installed']
    if loaded_endpoints:
        presence.append('loaded')
    presence.append('node0_qualified' if is_qualified else 'unqualified')

    return ModelRecord(
        model_id=model_id, runtime=RUNTIME, provider_id=PROVIDER,
        endpoints=endpoints, loaded_endpoints=loaded_endpoints,
        installed=Fact.measured(True, source=TAGS), loaded=Fact.measured(
            bool(loaded_endpoints), source=PS),
        node0_qualified=Fact.measured(is_qualified, source=HARDWARE),
        presence=tuple(presence),
        digest=_fact(digest, source=TAGS, reason='The runtime reports no digest for this model'),
        architecture=_fact(architecture, source=SHOW if shown else TAGS,
                           reason='Architecture is not reported by the runtime'),
        parameter_count=_fact(parameters, source=SHOW, unit='count',
                              reason='The artifact does not report a parameter count'),
        parameter_size_label=_fact(label, source=TAGS,
                                   reason='The runtime reports no parameter size label'),
        quantization=_fact(quantization, source=TAGS,
                           reason='The runtime reports no quantization level'),
        file_format=_fact(file_format, source=TAGS,
                          reason='The runtime reports no artifact format'),
        file_size_bytes=_fact(size, source=TAGS, unit='B',
                              reason='The runtime reports no artifact size'),
        embedding_length=_fact(_model_info(shown, '.embedding_length'), source=SHOW, unit='count',
                               reason='The artifact does not report an embedding length'),
        block_count=_fact(_model_info(shown, '.block_count'), source=SHOW, unit='count',
                          reason='The artifact does not report a block count'),
        native_context_tokens=_fact(native_context, source=SHOW, unit='tokens',
                                    reason='The artifact does not report a context length'),
        configured_context_tokens=_fact(
            configured_context, source=PS, unit='tokens',
            reason='The model is not resident, so no configured context is observable'),
        runtime_capabilities=_fact(list(capabilities) if capabilities else None, source=SHOW,
                                   reason='The runtime declares no capabilities'),
        parent_model=_fact(parent, source=TAGS,
                           reason='The runtime reports no parent model'),
        modified_at=_fact(modified, source=TAGS,
                          reason='The runtime reports no modification timestamp'),
        loaded_vram_bytes=_fact(loaded_vram, source=PS, unit='B',
                                reason='The model is not resident, so no VRAM use is observable'),
        load_expires_at=_fact(expires, source=PS,
                              reason='The model is not resident, so no expiry is observable'),
        qualification_id=qualification,
        provenance=tuple(sorted({TAGS, PS, SHOW if shown else TAGS, 'node0-qualification'})),
    )


def _hardware_values(hardware: HardwareSnapshot) -> dict[str, Fact]:
    gpu = hardware.gpus[0] if hardware.gpus else None
    volume = hardware.volumes[0] if hardware.volumes else None
    return {
        'vram_total': gpu.vram_total_bytes if gpu else Fact.unknown(
            'No GPU was detected', source=HARDWARE),
        'vram_free': gpu.vram_free_bytes if gpu else Fact.unknown(
            'No GPU was detected', source=HARDWARE),
        'storage_free': volume.free_bytes if volume else Fact.unknown(
            'No mounted volume was detected', source=HARDWARE),
        'memory_total': hardware.memory.installed_bytes,
        'memory_available': hardware.memory.available_bytes,
    }


def _compare(artifact: Fact, capacity: Fact, *, label: str) -> Fact:
    """Bytes only. False means 'did not fit in what was observed', never 'unusable'."""
    if artifact.kind == 'unknown' or capacity.kind == 'unknown':
        missing = 'artifact size' if artifact.kind == 'unknown' else label
        return Fact.unknown(f'Cannot compare against {label}: {missing} is unobserved',
                            source=FIT)
    return Fact.derived(int(artifact.value) <= int(capacity.value), source=FIT)


def _fit(record: ModelRecord, hardware: HardwareSnapshot) -> ModelFitFacts:
    values = _hardware_values(hardware)
    artifact = record.file_size_bytes
    vram_total, vram_free = values['vram_total'], values['vram_free']
    storage_free = values['storage_free']
    basis = [f'artifact={record.model_id}', f'hardware_digest={hardware_snapshot_digest(hardware)}']
    if record.loaded_vram_bytes.kind != 'unknown':
        basis.append(f'observed_loaded_vram={record.model_id}')
    if values['memory_available'].kind == 'unknown':
        basis.append(f'memory_available_unknown={values["memory_available"].reason}')

    fits_total = _compare(artifact, vram_total, label='total VRAM')
    fits_free = _compare(artifact, vram_free, label='free VRAM')
    fits_storage = _compare(artifact, storage_free, label='free storage')
    if fits_total.kind == 'unknown':
        residency = Fact.unknown(fits_total.reason, source=FIT)
    else:
        residency = Fact.derived(bool(fits_total.value) and bool(fits_free.value), source=FIT)
    if fits_total.kind == 'unknown':
        offload = Fact.unknown(
            f'Cannot determine offload: {fits_total.reason}', source=FIT)
    else:
        offload = Fact.derived(not bool(residency.value), source=FIT)
    headroom = (Fact.derived(int(storage_free.value) - int(artifact.value), source=FIT, unit='B')
                if storage_free.kind != 'unknown' and artifact.kind != 'unknown'
                else Fact.unknown('Cannot compute storage headroom', source=FIT))
    return ModelFitFacts(
        model_id=record.model_id, artifact_bytes=artifact, vram_total_bytes=vram_total,
        vram_free_bytes=vram_free, storage_free_bytes=storage_free,
        system_memory_bytes=values['memory_total'],
        system_memory_available_bytes=values['memory_available'],
        artifact_fits_total_vram=fits_total, artifact_fits_free_vram=fits_free,
        artifact_fits_free_storage=fits_storage, gpu_residency_plausible=residency,
        cpu_ram_offload_may_be_required=offload, storage_headroom_bytes=headroom,
        basis=tuple(basis))


def hardware_snapshot_digest(hardware: HardwareSnapshot) -> str:
    """Content digest of a hardware snapshot, excluding volatile readings."""
    data = _without_timestamps(hardware.model_dump(mode='json', exclude={'observed_at', 'coverage'}))
    for section, volatile in VOLATILE_FIELDS.items():
        if section in data:
            data[section] = {k: v for k, v in data[section].items() if k not in volatile}
    for group, volatile in VOLATILE_GROUP_FIELDS.items():
        for item in data.get(group) or ():
            for key in volatile:
                item.pop(key, None)
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _shared_digests(records: Sequence[ModelRecord]) -> tuple[str, ...]:
    """Digests held by more than one model id, i.e. a shared store blob."""
    by_digest: dict[str, set[str]] = {}
    for record in records:
        if record.digest.kind == 'measured':
            by_digest.setdefault(str(record.digest.value), set()).add(record.model_id)
    shared = [' + '.join(sorted(names)) for names in by_digest.values() if len(names) > 1]
    return tuple(sorted(shared))


def _benchmark_plan() -> BenchmarkPlan:
    fairness = (
        'Every model runs the same prompts, the same sampling settings and the same context budget.',
        'Each class repeats its measurement and reports median and worst case, never a single best run.',
        'A model is never compared while another model is resident, so VRAM starts from the same baseline.',
        'The qualified Node0 runtime is the only execution path; no cloud or remote fallback is allowed.',
        'Structured-output compliance is measured against the exact schema the agent declares.',
        'Timings come from the runtime, never from wall-clock guesses, and are stored with the run.',
    )
    classes = (
        BenchmarkClass(
            class_id='startup', name='Load and residency',
            purpose='Measure how long a model takes to become resident and what it holds.',
            metrics=('load_time', 'vram_peak', 'ram_peak'), repetitions=3, warmup_runs=1,
            fairness_rules=('Start from an unloaded runtime for every model.',
                            'Record VRAM and RAM peaks across the load window.')),
        BenchmarkClass(
            class_id='throughput', name='Decode throughput',
            purpose='Measure steady-state generation speed at a fixed context and output budget.',
            metrics=('time_to_first_token', 'tokens_per_second', 'gpu_utilization',
                     'gpu_temperature', 'gpu_power', 'stability'),
            repetitions=5, warmup_runs=1,
            fairness_rules=('Fix output budget and sampling parameters for every model.',
                            'Sample temperature and power throughout, not only at the end.')),
        BenchmarkClass(
            class_id='structured-json', name='Structured JSON compliance',
            purpose='Measure whether a model returns schema-valid JSON under the agent output schema.',
            metrics=('structured_json_compliance', 'repeated_run_consistency', 'stability'),
            repetitions=5, warmup_runs=1,
            fairness_rules=('Use the identical schema for every model.',
                            'Count a run compliant only if it parses without repair.')),
        BenchmarkClass(
            class_id='tool-calling', name='Tool calling',
            purpose='Measure correct tool selection and argument emission under the agent tool contract.',
            metrics=('tool_calling', 'structured_json_compliance', 'repeated_run_consistency'),
            repetitions=5, warmup_runs=1,
            fairness_rules=('Expose an identical tool catalogue to every model.',
                            'Score only observable calls, never intent.')),
        BenchmarkClass(
            class_id='long-context', name='Long context behaviour',
            purpose='Measure behaviour as the input context approaches the artifact limit.',
            metrics=('long_context_behavior', 'time_to_first_token', 'tokens_per_second',
                     'vram_peak', 'stability'),
            repetitions=3, warmup_runs=1,
            fairness_rules=('Test identical context lengths for every model.',
                            'Never exceed a model native context length.',
                            'Report the largest length that stayed stable.')),
        BenchmarkClass(
            class_id='quality-probe', name='Reasoning, coding and research probes',
            purpose='Measure observable reasoning, coding and agent-task behaviour on fixed tasks.',
            metrics=('reasoning', 'coding', 'short_context_quality', 'agent_task_performance',
                     'repeated_run_consistency'),
            repetitions=3, warmup_runs=0,
            fairness_rules=('Score against fixed rubrics, not preference.',
                            'Keep every prompt and rubric identical across models.')),
    )
    roles = (
        RoleCategory(role_id='FAST', name='Fast',
                     intent='Shortest time to first token and highest steady throughput.',
                     evidence_required=('time_to_first_token', 'tokens_per_second')),
        RoleCategory(role_id='GENERAL', name='General',
                     intent='Balanced default for mixed everyday work.',
                     evidence_required=('short_context_quality', 'tokens_per_second', 'stability')),
        RoleCategory(role_id='REASONING', name='Reasoning',
                     intent='Multi-step deduction quality.',
                     evidence_required=('reasoning', 'repeated_run_consistency')),
        RoleCategory(role_id='CODING', name='Coding',
                     intent='Code generation and editing correctness.',
                     evidence_required=('coding', 'structured_json_compliance')),
        RoleCategory(role_id='RESEARCH', name='Research',
                     intent='Long-form synthesis and analysis over supplied material only.',
                     evidence_required=('long_context_behavior', 'short_context_quality')),
        RoleCategory(role_id='TOOL_USE', name='Tool use',
                     intent='Reliable tool selection and argument emission.',
                     evidence_required=('tool_calling', 'structured_json_compliance')),
        RoleCategory(role_id='STRUCTURED_OUTPUT', name='Structured output',
                     intent='Schema compliance without manual repair.',
                     evidence_required=('structured_json_compliance', 'repeated_run_consistency')),
        RoleCategory(role_id='LONG_CONTEXT', name='Long context',
                     intent='Stable behaviour at large input sizes.',
                     evidence_required=('long_context_behavior', 'vram_peak', 'stability')),
        RoleCategory(role_id='VISION', name='Vision',
                     intent='Image understanding, if any candidate declares the capability.',
                     evidence_required=('short_context_quality',)),
    )
    metrics = ('load_time', 'time_to_first_token', 'tokens_per_second', 'vram_peak', 'ram_peak',
               'gpu_utilization', 'gpu_temperature', 'gpu_power', 'stability',
               'short_context_quality', 'long_context_behavior', 'structured_json_compliance',
               'tool_calling', 'reasoning', 'coding', 'agent_task_performance',
               'repeated_run_consistency')
    return BenchmarkPlan(metrics=metrics, classes=classes, roles=roles, fairness_rules=fairness)


def discover_models(*, hardware: HardwareSnapshot, runtime_endpoints: Sequence[str],
                    qualified_endpoint: str | None = None,
                    qualified_model: str | None = None,
                    qualified_id: str | None = None,
                    adapter_factory=None) -> ScoutSnapshot:
    """Enumerate every locally installed model and derive byte-level fit facts.

    Runtimes that share a store report the same model more than once; those views
    are merged into a single record that keeps every reporting endpoint, so the
    registry counts models rather than runtime sightings.
    """
    views: list[_RuntimeView] = []
    observed: list[str] = []
    unreachable: list[str] = []
    sightings: dict[str, list[tuple[_RuntimeView, dict]]] = {}
    for endpoint in dict.fromkeys(runtime_endpoints):
        view = _observe(endpoint, adapter_factory)
        if not view.reachable:
            unreachable.append(endpoint)
            continue
        views.append(view)
        observed.append(endpoint)
        for entry in view.tags:
            name = _text(entry.get('name') or entry.get('id'))
            if name:
                sightings.setdefault(name, []).append((view, entry))

    ordered = tuple(sorted(
        (_record(model_id, entries, qualified_endpoint, qualified_model, qualified_id)
         for model_id, entries in sorted(sightings.items())),
        key=lambda record: record.model_id))
    registry = ModelRegistry(
        records=ordered, fit=tuple(_fit(record, hardware) for record in ordered),
        runtimes_observed=tuple(sorted(observed)), runtimes_unreachable=tuple(sorted(unreachable)),
        qualified_endpoint=qualified_endpoint, qualified_model=qualified_model,
        shared_digests=_shared_digests(ordered),
        hardware_snapshot_digest=hardware_snapshot_digest(hardware))
    return ScoutSnapshot(
        registry=registry, benchmark_plan=_benchmark_plan(),
        hardware_digest=hardware_snapshot_digest(hardware),
        model_count=len(ordered),
        loaded_count=len(registry.by_state('loaded')),
        qualified_count=len(registry.by_state('node0_qualified')))
