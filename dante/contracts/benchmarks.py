"""Stage B2 benchmark contracts.

Every measurement keeps the identity of what was measured, the exact parameters it
was measured under, the raw runtime numbers, and the provenance of each number.
Dimensions stay separate: there is no universal quality score and no winner, and
an unavoidable configuration difference is recorded rather than smoothed over.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dante.contracts import StrictModel, utc_now
from dante.contracts.hardware import Fact

BENCHMARK_PROBE = 'benchmark-arena-v1'
Phase = Literal['cold', 'warm']
Disposition = Literal['completed', 'failed', 'timeout', 'resource_exhausted', 'load_failed']
GpuExecution = Literal['active', 'inactive', 'unknown']

# Fixed vocabulary so a result can never grow an unmeasured or invented metric.
Metric = Literal[
    'cold_load_s', 'warm_latency_s', 'time_to_first_token_s', 'prompt_tokens_per_second',
    'generation_tokens_per_second', 'total_latency_s', 'wall_clock_s', 'vram_resident_bytes',
    'vram_delta_bytes', 'ram_working_set_bytes', 'ram_delta_bytes', 'gpu_utilization_peak_percent',
    'gpu_utilization_mean_percent', 'gpu_temperature_peak_c', 'gpu_power_peak_w',
    'gpu_free_vram_min_bytes', 'prompt_tokens', 'output_tokens',
]
METRIC_KEYS = tuple(Metric.__args__)


def text_digest(value: str) -> str:
    """Stable digest of a prompt or output, so identity is provable without the text."""
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class RuntimeResidency(StrictModel):
    """What one runtime held at one moment. Empty residency is normal, not an error."""

    endpoint: str = Field(min_length=1, max_length=200)
    observed_at: datetime = Field(default_factory=utc_now)
    models: tuple[tuple[str, int, int], ...] = ()
    gpu_used_bytes: int | None = Field(default=None, ge=0)
    gpu_free_bytes: int | None = Field(default=None, ge=0)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, _, _ in self.models))

    def same_models_as(self, other: 'RuntimeResidency') -> bool:
        return self.models == other.models


class VramBudget(StrictModel):
    """A deliberately small GPU allowance, so production residency is never at risk."""

    free_vram_bytes: int = Field(ge=0)
    budget_bytes: int = Field(gt=0)
    reserve_bytes: int = Field(ge=0)
    kv_reserve_bytes: int = Field(ge=0)
    full_gpu_estimate_bytes: int | None = Field(default=None, ge=0)
    overhead_ratio: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    num_gpu_layers: int = Field(ge=0)
    block_count: int | None = Field(default=None, ge=0)
    full_gpu_possible: bool
    basis: str = Field(min_length=1, max_length=500)

    @model_validator(mode='after')
    def budget_is_bounded(self):
        if self.budget_bytes + self.reserve_bytes > self.free_vram_bytes:
            raise ValueError('Budget plus reserve cannot exceed observed free VRAM')
        if self.kv_reserve_bytes >= self.budget_bytes:
            raise ValueError('KV reserve must leave room for weights inside the budget')
        return self


class BenchmarkIsolation(StrictModel):
    """Proof that benchmarking never touched the qualified production runtime."""

    production_endpoint: str = Field(min_length=1, max_length=200)
    benchmark_endpoint: str = Field(min_length=1, max_length=200)
    production_before: RuntimeResidency
    production_after: RuntimeResidency | None = None
    production_unchanged: bool | None = None
    production_checked_batches: int = Field(default=0, ge=0)
    production_disturbances: tuple[str, ...] = ()
    benchmark_before: RuntimeResidency
    benchmark_after: RuntimeResidency | None = None
    benchmark_restored: bool | None = None
    cleanup_attempted: bool = False
    models_loaded: tuple[str, ...] = ()
    models_unloaded: tuple[str, ...] = ()
    leaked_residency: tuple[str, ...] = ()
    policy: str = Field(min_length=1, max_length=500)
    violations: tuple[str, ...] = ()

    @model_validator(mode='after')
    def endpoints_must_differ(self):
        # The single most important invariant: production and benchmark are never
        # the same runtime, so a benchmark load cannot evict a production model.
        if self.production_endpoint == self.benchmark_endpoint:
            raise ValueError('Benchmark runtime must never be the qualified production runtime')
        return self

    def verified(self) -> bool:
        return (self.production_unchanged is True and self.benchmark_restored is True
                and not self.production_disturbances and not self.violations
                and not self.leaked_residency)


class BenchmarkParameters(StrictModel):
    """Sampling configuration. Identical across every model in a comparison."""

    temperature: float = Field(allow_inf_nan=False)
    seed: int
    output_token_budget: int = Field(gt=0, le=8192)
    context_token_budget: int = Field(gt=0, le=131072)
    num_gpu_layers: int | None = Field(default=None, ge=0)
    think: bool = False
    structured_output: bool = False

    def sampling_key(self) -> tuple:
        """The part that must be identical for two runs to be comparable.

        `structured_output` is deliberately excluded: JSON mode is a separate
        capability measurement, not a sampling difference. Samples measured in
        JSON mode are kept out of the speed medians instead.
        """
        return (self.temperature, self.seed, self.output_token_budget,
                self.context_token_budget, self.think)


class BenchmarkPrompt(StrictModel):
    """A fixed test input, identified by digest so the exact text is provable."""

    prompt_id: str = Field(pattern=r'^[A-Z0-9_]{1,32}$')
    kind: Literal['latency', 'structured_json', 'instruction', 'reasoning', 'coding',
                  'tool_use', 'synthesis', 'long_context']
    digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    characters: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=20000)

    @model_validator(mode='after')
    def digest_matches_text(self):
        if text_digest(self.text) != self.digest:
            raise ValueError('Prompt digest does not match the prompt text')
        return self


class BenchmarkSample(StrictModel):
    """One measurement of one prompt, on one model, at one repetition."""

    model_id: str = Field(min_length=1, max_length=200)
    digest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    quantization: str | None = Field(default=None, max_length=32)
    parameter_size_label: str | None = Field(default=None, max_length=32)
    runtime_endpoint: str = Field(min_length=1, max_length=200)
    runtime_version: str | None = Field(default=None, max_length=32)
    hardware_snapshot_digest: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    prompt_id: str = Field(pattern=r'^[A-Z0-9_]{1,32}$')
    prompt_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    prompt_kind: str = Field(min_length=1, max_length=32)
    parameters: BenchmarkParameters
    repetition: int = Field(ge=1, le=1000)
    phase: Phase
    gpu_execution: GpuExecution = 'unknown'
    started_at: datetime
    completed_at: datetime
    measurements: dict[str, Fact] = Field(default_factory=dict)
    done_reason: str | None = Field(default=None, max_length=64)
    output_excerpt: str = Field(default='', max_length=600)
    output_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    output_parsed: bool | None = None
    disposition: Disposition = 'completed'
    failure: str | None = Field(default=None, max_length=500)
    sensor_source: str = Field(min_length=1, max_length=64)

    @field_validator('measurements')
    @classmethod
    def known_metrics_only(cls, value):
        unknown = set(value) - set(METRIC_KEYS)
        if unknown:
            raise ValueError(f'Unknown benchmark metric: {sorted(unknown)}')
        return value

    @model_validator(mode='after')
    def coherent(self):
        if self.completed_at < self.started_at:
            raise ValueError('Sample completed before it started')
        if self.disposition == 'completed':
            if self.failure:
                raise ValueError('A completed sample carries no failure')
        elif not self.failure:
            raise ValueError('An incomplete sample must record why')
        if self.phase == 'cold' and 'cold_load_s' not in self.measurements:
            raise ValueError('A cold sample must record load duration')
        return self

    def value(self, metric: str):
        fact = self.measurements.get(metric)
        return fact.value if fact is not None and fact.kind != 'unknown' else None


class ModelBenchmarkProfile(StrictModel):
    """Everything known about one model before and after its run."""

    model_id: str = Field(min_length=1, max_length=200)
    digest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    quantization: str | None = Field(default=None, max_length=32)
    parameter_size_label: str | None = Field(default=None, max_length=32)
    parameter_count: int | None = Field(default=None, ge=0)
    artifact_bytes: int | None = Field(default=None, ge=0)
    architecture: str | None = Field(default=None, max_length=64)
    native_context_tokens: int | None = Field(default=None, ge=0)
    block_count: int | None = Field(default=None, ge=0)
    runtime_endpoint: str = Field(min_length=1, max_length=200)
    tested: bool = False
    skip_reason: str | None = Field(default=None, max_length=500)
    load_succeeded: bool | None = None
    load_failure: str | None = Field(default=None, max_length=500)
    vram_budget: VramBudget | None = None
    observed_vram_resident_bytes: int | None = Field(default=None, ge=0)
    observed_ram_working_set_bytes: int | None = Field(default=None, ge=0)
    gpu_execution: GpuExecution = 'unknown'
    sample_count: int = Field(default=0, ge=0)
    completed_samples: int = Field(default=0, ge=0)
    failed_samples: int = Field(default=0, ge=0)
    timed_out_samples: int = Field(default=0, ge=0)
    malformed_outputs: int = Field(default=0, ge=0)
    median_generation_tokens_per_second: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    median_prompt_tokens_per_second: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    median_cold_load_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    median_warm_latency_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    median_time_to_first_token_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    dimension_scores: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def either_tested_or_skipped(self):
        if not self.tested and not self.skip_reason:
            raise ValueError('An untested model must record why it was skipped')
        if self.tested and self.skip_reason:
            raise ValueError('A tested model cannot also be skipped')
        if self.failed_samples + self.completed_samples > self.sample_count:
            raise ValueError('Sample tallies exceed the sample count')
        return self


class BenchmarkRun(StrictModel):
    """One model's complete run, with its samples in deterministic order."""

    profile: ModelBenchmarkProfile
    started_at: datetime
    completed_at: datetime
    samples: tuple[BenchmarkSample, ...] = ()
    batch_count: int = Field(ge=0)
    production_unchanged_through_run: bool | None = None
    abort_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode='after')
    def ordered_and_scoped(self):
        if self.completed_at < self.started_at:
            raise ValueError('Run completed before it started')
        keys = [(item.repetition, item.prompt_id, item.phase) for item in self.samples]
        if keys != sorted(keys):
            raise ValueError('Samples must be in deterministic order')
        for sample in self.samples:
            if sample.model_id != self.profile.model_id:
                raise ValueError('A run may only carry its own model samples')
        return self


class DimensionComparison(StrictModel):
    """One dimension compared across models. Never a universal verdict."""

    dimension: str = Field(pattern=r'^[a-z0-9_]{1,48}$')
    unit: str | None = Field(default=None, max_length=24)
    comparable: bool = True
    values: dict[str, Fact] = Field(default_factory=dict)
    best_model_id: str | None = None
    basis: str = Field(min_length=1, max_length=500)
    blocking_differences: tuple[str, ...] = ()

    @model_validator(mode='after')
    def no_winner_without_comparability(self):
        if self.best_model_id is not None and not self.comparable:
            raise ValueError('A blocked comparison may not name a best model')
        return self


class BenchmarkComparison(StrictModel):
    """Dimension-by-dimension comparison. Explicitly no overall score and no winner."""

    # An empty model list is legitimate: it is what a run where nothing could be
    # measured looks like, and it must stay representable rather than crash.
    models: tuple[str, ...] = ()
    dimensions: tuple[DimensionComparison, ...] = ()
    shared_parameters_identical: bool
    recorded_differences: tuple[str, ...] = ()
    universal_score: Literal[None] = None
    overall_winner: Literal[None] = None
    role_assignments: tuple[str, ...] = Field(default=(), max_length=0)
    verdict: Literal['dimension_only'] = 'dimension_only'
    policy: str = Field(min_length=1, max_length=500)

    @model_validator(mode='after')
    def dimensions_may_not_be_flattened(self):
        if self.universal_score is not None or self.overall_winner is not None:
            raise ValueError('Stage B2 may not publish a universal score or an overall winner')
        if self.role_assignments:
            raise ValueError('Role assignment requires a later, benchmarked decision')
        names = [item.dimension for item in self.dimensions]
        if names != sorted(names):
            raise ValueError('Dimensions must be in deterministic order')
        for dimension in self.dimensions:
            if not set(dimension.values) <= set(self.models):
                raise ValueError('A dimension may only report known models')
        return self

    def by_dimension(self, dimension: str):
        for item in self.dimensions:
            if item.dimension == dimension:
                return item
        return None


class BenchmarkResult(StrictModel):
    """The complete Stage B2 measurement result for one job."""

    observed_at: datetime = Field(default_factory=utc_now)
    probe_version: str = Field(default=BENCHMARK_PROBE,
                               pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    isolation: BenchmarkIsolation
    hardware_snapshot_digest: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    runs: tuple[BenchmarkRun, ...] = ()
    comparison: BenchmarkComparison
    smoke_only: bool = True
    models_downloaded: Literal[0] = 0
    models_deleted: Literal[0] = 0
    qualification_changed: Literal[False] = False
    role_assignments_made: Literal[0] = 0
    campaign_exhausted: bool = False

    @model_validator(mode='after')
    def ordered_and_complete(self):
        names = [run.profile.model_id for run in self.runs]
        if names != sorted(names):
            raise ValueError('Benchmark runs must be in deterministic order')
        if set(self.comparison.models) != set(self.tested_models()):
            raise ValueError('Comparison must cover exactly the models that were tested')
        if self.isolation.production_endpoint == self.isolation.benchmark_endpoint:
            raise ValueError('Benchmark result must never target the production runtime')
        return self

    def tested_models(self) -> tuple[str, ...]:
        return tuple(run.profile.model_id for run in self.runs if run.profile.tested)

    def fact_count(self) -> tuple[int, int, int]:
        counts = {'measured': 0, 'derived': 0, 'unknown': 0}
        for run in self.runs:
            for sample in run.samples:
                for fact in sample.measurements.values():
                    counts[fact.kind] += 1
        return counts['measured'], counts['derived'], counts['unknown']


__all__ = [
    'BENCHMARK_PROBE', 'METRIC_KEYS', 'BenchmarkComparison', 'BenchmarkIsolation',
    'BenchmarkParameters', 'BenchmarkPrompt', 'BenchmarkResult', 'BenchmarkRun',
    'BenchmarkSample', 'DimensionComparison', 'ModelBenchmarkProfile', 'RuntimeResidency',
    'VramBudget', 'text_digest',
]
