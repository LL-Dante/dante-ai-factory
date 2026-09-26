"""Read-only model discovery contracts.

This module is NOT `dante.registry`. That module is the qualification allowlist that
gates execution. This one is a passive catalog of models that already exist on the
machine, discovered without downloading, deleting or loading anything.

Every attribute is a `Fact`, so an absent field is UNKNOWN with a reason and never a
fabricated value. Nothing here ranks models, declares a winner, or estimates speed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from dante.contracts import StrictModel, utc_now
from dante.contracts.hardware import (Fact, FactValue, HardwareSnapshot, ModelCapacityFact,
                                      Reason, Source)


PresenceState = Literal['installed', 'loaded', 'node0_qualified', 'unqualified', 'unknown']
ModelRegistryProbe = 'model-scout-v1'


class ModelRecord(StrictModel):
    """One model, exactly as observed, with every runtime that reports it."""

    model_id: str = Field(min_length=1, max_length=160)
    runtime: str = Field(min_length=1, max_length=64)
    provider_id: str = Field(default='ollama', min_length=1, max_length=64)

    endpoints: tuple[str, ...] = ()
    loaded_endpoints: tuple[str, ...] = ()
    installed: Fact
    loaded: Fact
    node0_qualified: Fact
    presence: tuple[PresenceState, ...] = ()

    digest: Fact
    architecture: Fact
    parameter_count: Fact
    parameter_size_label: Fact
    quantization: Fact
    file_format: Fact
    file_size_bytes: Fact
    embedding_length: Fact
    block_count: Fact
    native_context_tokens: Fact
    configured_context_tokens: Fact
    runtime_capabilities: Fact
    parent_model: Fact
    modified_at: Fact
    loaded_vram_bytes: Fact
    load_expires_at: Fact
    qualification_id: Fact

    provenance: tuple[Source, ...] = ()
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode='after')
    def presence_matches_observations(self):
        """Presence labels may never claim more than the underlying facts support."""
        labels = set(self.presence)
        if self.installed.value is not True and 'installed' in labels:
            raise ValueError('Presence claims installed while installation is unobserved')
        if self.loaded.value is True and not self.loaded_endpoints:
            raise ValueError('Presence claims loaded without a resident endpoint')
        if self.loaded.value is not True and 'loaded' in labels:
            raise ValueError('Presence claims loaded while the model is not resident')
        if self.node0_qualified.value is True and 'unqualified' in labels:
            raise ValueError('Presence claims unqualified while qualification is observed')
        if self.node0_qualified.value is True and self.qualification_id.kind != 'measured':
            raise ValueError('A qualified model must carry its qualification id')
        for endpoint in self.loaded_endpoints:
            if endpoint not in self.endpoints:
                raise ValueError('A resident endpoint must also report the model as installed')
        return self

    def facts(self):
        for name in type(self).model_fields:
            if name in ('model_id', 'runtime', 'provider_id', 'endpoints', 'loaded_endpoints',
                        'presence', 'provenance', 'observed_at'):
                continue
            value = getattr(self, name)
            if isinstance(value, Fact):
                yield value

    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(name for name in type(self).model_fields
                     if isinstance(getattr(self, name), Fact)
                     and getattr(self, name).kind == 'unknown')

    def declared_capabilities(self) -> tuple[str, ...]:
        """Runtime-declared capabilities, split back out of the measured string."""
        if self.runtime_capabilities.kind != 'measured':
            return ()
        return tuple(part for part in str(self.runtime_capabilities.value).split(',') if part)


class ModelFitFacts(StrictModel):
    """Deterministic feasibility arithmetic for one model on this machine.

    These are comparisons of measured bytes, never predictions of speed or quality.
    A `False` residency verdict means only that the artifact does not fit in the
    memory that was observed, not that the model is bad.
    """

    model_id: str = Field(min_length=1, max_length=160)
    artifact_bytes: Fact
    vram_total_bytes: Fact
    vram_free_bytes: Fact
    storage_free_bytes: Fact
    system_memory_bytes: Fact
    system_memory_available_bytes: Fact

    artifact_fits_total_vram: Fact
    artifact_fits_free_vram: Fact
    artifact_fits_free_storage: Fact
    gpu_residency_plausible: Fact
    cpu_ram_offload_may_be_required: Fact
    storage_headroom_bytes: Fact

    basis: tuple[str, ...] = ()
    verdicts_withheld: Literal[True] = True
    reasons_withheld: tuple[str, ...] = (
        'No throughput, latency or quality measurement exists, so no ranking is computed.',
        'Residency verdicts compare bytes only and say nothing about speed or output quality.',
    )

    def facts(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, Fact):
                yield value

    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(name for name in type(self).model_fields
                     if isinstance(getattr(self, name), Fact)
                     and getattr(self, name).kind == 'unknown')


class ModelRegistry(StrictModel):
    """Deterministic, deduplicated catalog of every discovered model."""

    observed_at: datetime = Field(default_factory=utc_now)
    probe_version: str = Field(default=ModelRegistryProbe,
                               pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    read_only: Literal[True] = True
    records: tuple[ModelRecord, ...] = ()
    fit: tuple[ModelFitFacts, ...] = ()
    runtimes_observed: tuple[str, ...] = ()
    runtimes_unreachable: tuple[str, ...] = ()
    qualified_endpoint: str | None = None
    qualified_model: str | None = None
    shared_digests: tuple[str, ...] = ()
    hardware_snapshot_digest: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    rankings_withheld: Literal[True] = True
    reasons_withheld: tuple[str, ...] = (
        'No benchmark has been executed, so no model is ranked or declared best.',
        'Parameter count and quantization describe an artifact, not its output quality.',
    )

    @model_validator(mode='after')
    def ordered_and_unique(self):
        names = [item.model_id for item in self.records]
        if names != sorted(names):
            raise ValueError('Model records must be in deterministic order')
        if len(set(names)) != len(names):
            raise ValueError('Duplicate model identity in the registry')
        if sorted(entry.model_id for entry in self.fit) != names:
            raise ValueError('Every record needs exactly one fit analysis')
        for entry in self.fit:
            if entry.verdicts_withheld is not True:
                raise ValueError('Fit verdicts must stay withheld in Stage B1')
        return self

    def identities(self) -> tuple[str, ...]:
        return tuple(item.model_id for item in self.records)

    def require(self, model_id: str) -> ModelRecord:
        for item in self.records:
            if item.model_id == model_id:
                return item
        raise KeyError(model_id)

    def by_state(self, state: str) -> tuple[ModelRecord, ...]:
        return tuple(item for item in self.records if state in item.presence)

    def fact_count(self) -> tuple[int, int, int]:
        counts = {'measured': 0, 'derived': 0, 'unknown': 0}
        for record in self.records:
            for fact in record.facts():
                counts[fact.kind] += 1
        for entry in self.fit:
            for fact in entry.facts():
                counts[fact.kind] += 1
        return counts['measured'], counts['derived'], counts['unknown']

    def unknown_fields(self) -> tuple[str, ...]:
        paths = [f'records[{item.model_id}].{name}'
                 for item in self.records for name in item.unknown_fields()]
        paths += [f'fit[{entry.model_id}].{name}'
                  for entry in self.fit for name in entry.unknown_fields()]
        return tuple(paths)

    def digest_data(self) -> dict:
        """Identity of catalog content, independent of observation timestamps."""
        from dante.contracts.hardware import _without_timestamps
        return _without_timestamps(self.model_dump(
            mode='json', exclude={'observed_at', 'records', 'fit', 'hardware_snapshot_digest'}))


BenchmarkMetric = Literal[
    'load_time', 'time_to_first_token', 'tokens_per_second', 'vram_peak', 'ram_peak',
    'gpu_utilization', 'gpu_temperature', 'gpu_power', 'stability', 'short_context_quality',
    'long_context_behavior', 'structured_json_compliance', 'tool_calling', 'reasoning',
    'coding', 'agent_task_performance', 'repeated_run_consistency',
]


class BenchmarkClass(StrictModel):
    """A comparable measurement unit. Designed in Stage B1, executed in Stage B2."""

    class_id: str = Field(min_length=1, max_length=64, pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    name: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=500)
    metrics: tuple[BenchmarkMetric, ...] = Field(min_length=1)
    repetitions: int = Field(ge=1, le=64)
    warmup_runs: int = Field(ge=0, le=16)
    fairness_rules: tuple[str, ...] = Field(min_length=1)
    execution_state: Literal['designed_not_executed'] = 'designed_not_executed'


class RoleCategory(StrictModel):
    """A future selection bucket. Stage B1 defines the axis and assigns nobody."""

    role_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Z0-9_]+$')
    name: str = Field(min_length=1, max_length=120)
    intent: str = Field(min_length=1, max_length=500)
    evidence_required: tuple[str, ...] = Field(min_length=1)
    assigned_models: tuple[str, ...] = ()
    assignment_state: Literal['unassigned'] = 'unassigned'

    @model_validator(mode='after')
    def unassigned_until_benchmarked(self):
        if self.assigned_models:
            raise ValueError('A role may not name a model before benchmarks produce evidence')
        if self.assignment_state != 'unassigned':
            raise ValueError('Stage B1 may only record unassigned roles')
        return self


class BenchmarkPlan(StrictModel):
    """The future measurement suite. Nothing here runs in Stage B1."""

    plan_version: str = Field(default='benchmark-plan-v1',
                              pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    execution_state: Literal['designed_not_executed'] = 'designed_not_executed'
    metrics: tuple[BenchmarkMetric, ...] = Field(min_length=1)
    classes: tuple[BenchmarkClass, ...] = Field(min_length=1)
    roles: tuple[RoleCategory, ...] = Field(min_length=1)
    fairness_rules: tuple[str, ...] = Field(min_length=1)
    blocked_by: tuple[str, ...] = (
        'Stage B2 has not run: no load, latency, throughput, memory or quality data exists.',
    )

    def assigned_models(self) -> tuple[str, ...]:
        """Every model bound to any role. Empty until benchmarks produce evidence."""
        return tuple(name for role in self.roles for name in role.assigned_models)


class ScoutSnapshot(StrictModel):
    """The complete read-only discovery result."""

    observed_at: datetime = Field(default_factory=utc_now)
    probe_version: str = Field(default=ModelRegistryProbe,
                               pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    read_only: Literal[True] = True
    registry: ModelRegistry
    benchmark_plan: BenchmarkPlan
    hardware_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_count: int = Field(ge=0)
    loaded_count: int = Field(ge=0)
    qualified_count: int = Field(ge=0)
    inference_performed: Literal[False] = False
    models_downloaded: Literal[0] = 0
    models_deleted: Literal[0] = 0
    winners_declared: Literal[None] = None

    @model_validator(mode='after')
    def counts_match_registry(self):
        """Headline counts must be the registry's own counts, never a separate claim."""
        if self.model_count != len(self.registry.records):
            raise ValueError('Model count does not match the registry')
        if self.loaded_count != len(self.registry.by_state('loaded')):
            raise ValueError('Loaded count does not match the registry')
        if self.qualified_count != len(self.registry.by_state('node0_qualified')):
            raise ValueError('Qualified count does not match the registry')
        if self.hardware_digest != self.registry.hardware_snapshot_digest:
            raise ValueError('Hardware digest must match the registry digest')
        if self.benchmark_plan.execution_state != 'designed_not_executed':
            raise ValueError('Stage B1 may only carry an unexecuted benchmark plan')
        if self.benchmark_plan.assigned_models():
            raise ValueError('No model may be assigned a role before benchmarks run')
        return self

    def fact_count(self) -> tuple[int, int, int]:
        return self.registry.fact_count()

    def unknown_fields(self) -> tuple[str, ...]:
        return self.registry.unknown_fields()
