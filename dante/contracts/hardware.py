"""Normalized, provenance-carrying hardware inventory.

Every fact is explicitly MEASURED, DERIVED or UNKNOWN. UNKNOWN facts must carry a
reason and never a value, so an absent sensor can never be mistaken for a reading
and a value can never be invented. Nothing here interprets suitability: Stage A
records facts and deterministic arithmetic only, and holds recommendations back.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from dante.contracts import StrictModel, utc_now


FactKind = Literal['measured', 'derived', 'unknown']
FactValue = float | int | str | bool
Source = Annotated[str, Field(min_length=1, max_length=160)]
Reason = Annotated[str, Field(min_length=1, max_length=240)]


class Fact(StrictModel):
    """One normalized observation. Absent is represented, never defaulted."""

    kind: FactKind
    value: FactValue | None = None
    unit: str | None = Field(default=None, max_length=24)
    source: Source
    reason: Reason | None = None
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode='after')
    def no_fabricated_values(self):
        if self.kind == 'unknown':
            if self.value is not None or self.unit is not None or not self.reason:
                raise ValueError('Unknown facts carry a reason and no value')
        elif self.value is None or self.reason is not None:
            raise ValueError('Known facts carry a value and no reason')
        return self

    @classmethod
    def measured(cls, value, *, source: str, unit: str | None = None) -> 'Fact':
        return cls(kind='measured', value=value, unit=unit, source=source)

    @classmethod
    def derived(cls, value, *, source: str, unit: str | None = None) -> 'Fact':
        return cls(kind='derived', value=value, unit=unit, source=source)

    @classmethod
    def unknown(cls, reason: str, *, source: str) -> 'Fact':
        return cls(kind='unknown', source=source, reason=reason)


class SystemFacts(StrictModel):
    os_name: Fact
    os_version: Fact
    os_build: Fact
    architecture: Fact
    uptime_seconds: Fact
    power_plan_guid: Fact
    power_plan_name: Fact


class CpuFacts(StrictModel):
    model: Fact
    physical_cores: Fact
    logical_threads: Fact
    base_clock_mhz: Fact
    current_clock_mhz: Fact
    utilization_percent: Fact
    temperature_c: Fact


class MemoryFacts(StrictModel):
    installed_bytes: Fact
    available_bytes: Fact
    speed_mhz: Fact
    configured_speed_mhz: Fact
    module_count: Fact
    channel_layout: Fact
    memory_type: Fact


class GpuFacts(StrictModel):
    slot: str
    vendor: str
    name: str
    uuid: Fact
    driver_version: Fact
    compute_capability: Fact
    cuda_driver_api: Fact
    vram_total_bytes: Fact
    vram_used_bytes: Fact
    vram_free_bytes: Fact
    utilization_percent: Fact
    temperature_c: Fact
    power_draw_w: Fact
    power_limit_w: Fact
    sm_clock_mhz: Fact
    memory_clock_mhz: Fact
    max_sm_clock_mhz: Fact
    pcie_generation: Fact
    pcie_link_width: Fact


class DiskFacts(StrictModel):
    index: str
    model: Fact
    interface_type: Fact
    media_type: Fact
    size_bytes: Fact


class VolumeFacts(StrictModel):
    mount_point: str
    filesystem: Fact
    size_bytes: Fact
    free_bytes: Fact


class ModelStoreFacts(StrictModel):
    name: str
    exists: Fact
    file_count: Fact
    size_bytes: Fact


class RuntimeModelFacts(StrictModel):
    runtime_reference: str
    size_bytes: Fact
    digest: Fact
    quantization: Fact


class RuntimeFacts(StrictModel):
    endpoint: str
    runtime: str
    version: Fact
    reachable: Fact
    qualified_node0: Fact
    installed_models: tuple[RuntimeModelFacts, ...] = ()
    loaded_models: tuple[RuntimeModelFacts, ...] = ()
    vram_allocated_bytes: Fact


class SensorCoverage(StrictModel):
    """Which optional sensor families were actually available this collection."""
    monitoring_reachable: bool = False
    monitoring_source: Source
    monitoring_observed_at: datetime | None = None
    available: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()


def _without_timestamps(value):
    """Recursively drop per-fact observation timestamps from a dumped payload."""
    if isinstance(value, dict):
        return {k: _without_timestamps(v) for k, v in value.items() if k != 'observed_at'}
    if isinstance(value, list):
        return [_without_timestamps(item) for item in value]
    return value


def _section_facts(section):
    """Yield every Fact held directly by one inventory section model."""
    for name in type(section).model_fields:
        value = getattr(section, name)
        if isinstance(value, Fact):
            yield value
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, Fact):
                    yield item


VOLATILE_FIELDS = {
    'system': {'uptime_seconds'},
    'cpu': {'utilization_percent', 'current_clock_mhz', 'temperature_c'},
    'memory': {'available_bytes'},
}
VOLATILE_GROUP_FIELDS = {
    'gpus': {'temperature_c', 'power_draw_w', 'power_limit_w', 'utilization_percent',
             'vram_used_bytes', 'vram_free_bytes', 'sm_clock_mhz', 'memory_clock_mhz'},
    'volumes': {'free_bytes'},
    'runtimes': {'loaded_models', 'vram_allocated_bytes'},
}


class HardwareSnapshot(StrictModel):
    observed_at: datetime = Field(default_factory=utc_now)
    probe_version: str = Field(default='hardware-v1', pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    read_only: Literal[True] = True
    system: SystemFacts
    cpu: CpuFacts
    memory: MemoryFacts
    gpus: tuple[GpuFacts, ...] = ()
    disks: tuple[DiskFacts, ...] = ()
    volumes: tuple[VolumeFacts, ...] = ()
    model_stores: tuple[ModelStoreFacts, ...] = ()
    runtimes: tuple[RuntimeFacts, ...] = ()
    coverage: SensorCoverage

    def fact_count(self) -> tuple[int, int, int]:
        """(measured, derived, unknown) totals across every normalized fact."""
        counts = {'measured': 0, 'derived': 0, 'unknown': 0}
        for fact in self.facts():
            counts[fact.kind] += 1
        return counts['measured'], counts['derived'], counts['unknown']

    def facts(self):
        """Every normalized fact in this snapshot, in stable section order."""
        for section in (self.system, self.cpu, self.memory):
            yield from _section_facts(section)
        for group in (self.gpus, self.disks, self.volumes, self.model_stores, self.runtimes):
            for item in group:
                yield from _section_facts(item)

    def unknown_facts(self) -> tuple[str, ...]:
        """Dotted paths of every fact that could not be observed."""
        paths = []
        for section in ('system', 'cpu', 'memory'):
            for name, fact in getattr(self, section):
                if fact.kind == 'unknown':
                    paths.append(f'{section}.{name}')
        for group, label in ((self.gpus, 'gpus'), (self.disks, 'disks'), (self.volumes, 'volumes'),
                             (self.model_stores, 'model_stores'), (self.runtimes, 'runtimes')):
            for item in group:
                key = getattr(item, 'endpoint', None) or getattr(item, 'mount_point', None) \
                    or getattr(item, 'slot', None) or getattr(item, 'index', None) \
                    or getattr(item, 'name', None)
                for name, fact in item:
                    if isinstance(fact, Fact) and fact.kind == 'unknown':
                        paths.append(f'{label}[{key}].{name}')
        return tuple(paths)

    def digest_data(self) -> dict:
        """Identity of the machine-independent content, for cache/provenance use.

        Volatile readings and every observation timestamp are excluded so the same
        machine yields the same digest across polls, while static identity and the
        facts that describe it still do.
        """
        data = _without_timestamps(self.model_dump(mode='json', exclude={'observed_at', 'coverage'}))
        for section, volatile in VOLATILE_FIELDS.items():
            if section in data:
                data[section] = {k: v for k, v in data[section].items() if k not in volatile}
        for group, volatile in VOLATILE_GROUP_FIELDS.items():
            for item in data.get(group) or ():
                for key in volatile:
                    item.pop(key, None)
        return data


class ModelCapacityFact(StrictModel):
    """A deterministic hardware dimension a future ModelScout could filter on.

    This is deliberately not advice: no recommendation, ranking or fit verdict is
    expressed here, because suitability depends on a model that has not been chosen.
    """

    dimension: str = Field(min_length=1, max_length=64)
    kind: FactKind
    value: FactValue | None = None
    unit: str | None = Field(default=None, max_length=24)
    basis: tuple[str, ...] = ()
    reason: Reason | None = None

    @model_validator(mode='after')
    def consistent(self):
        if self.kind == 'unknown':
            if self.value is not None or not self.reason:
                raise ValueError('Unknown capacity facts carry a reason and no value')
        elif self.value is None or self.reason is not None:
            raise ValueError('Known capacity facts carry a value and no reason')
        return self


class HardwareCapabilityProfile(StrictModel):
    """Normalized model-capacity view derived from one snapshot. Advice withheld."""

    observed_at: datetime = Field(default_factory=utc_now)
    source_snapshot_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    recommendations_withheld: Literal[True] = True
    reasons_withheld: tuple[str, ...] = (
        'No model has been selected, so no fit, ranking or download recommendation is computed.',
        'Sustained throughput, thermal throttling and context-length cost are unmeasured.',
    )
    facts: tuple[ModelCapacityFact, ...] = ()

    def measured(self) -> tuple[ModelCapacityFact, ...]:
        return tuple(fact for fact in self.facts if fact.kind == 'measured')

    def unknown(self) -> tuple[ModelCapacityFact, ...]:
        return tuple(fact for fact in self.facts if fact.kind == 'unknown')
