"""Node evidence contracts. Unknown observations are never fabricated defaults."""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, model_validator

from dante.contracts import utc_now
from dante.recovery import digest

Label = Annotated[str, Field(min_length=1, max_length=160, pattern=r'^[A-Za-z0-9][A-Za-z0-9 ._()+-]*$')]
Sha256 = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
Measurement = Annotated[float, Field(ge=0, allow_inf_nan=False)]
SUITE_VERSION = 'node0-v1'


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False)


class GPUProfile(EvidenceModel):
    slot: Label  # Local ordinal, never a hardware serial number.
    vendor: Label
    name: Label
    uuid: Label | None = None
    vram_bytes: PositiveInt | None = None
    compute_capability: Label | None = None
    driver_version: Label | None = None


class MachineProfile(EvidenceModel):
    node_id: UUID  # Random provisioned identity, not hostname/MAC/serial.
    os: Label
    os_version: Label
    architecture: Label
    cpu: Label | None = None
    logical_cpus: PositiveInt | None = None
    physical_cores: PositiveInt | None = None
    ram_bytes: PositiveInt | None = None
    gpus: tuple[GPUProfile, ...] | None = None  # None = unprobed; () = observed CPU-only.
    cuda_runtime: Label | None = None
    cuda_toolkit: Label | None = None
    storage_class: Label | None = None
    storage_capacity_bytes: PositiveInt | None = None
    storage_free_bytes: Annotated[StrictInt, Field(ge=0)] | None = None
    observed_at: AwareDatetime = Field(default_factory=utc_now)
    probe_version: Label = 'stdlib-v1'

    @model_validator(mode='after')
    def consistent_inventory(self):
        if self.physical_cores and self.logical_cpus and self.physical_cores > self.logical_cpus:
            raise ValueError('Physical cores exceed logical CPUs')
        if self.gpus is not None and len({gpu.slot for gpu in self.gpus}) != len(self.gpus):
            raise ValueError('Duplicate GPU slot')
        if (self.storage_free_bytes is not None and self.storage_capacity_bytes is not None
                and self.storage_free_bytes > self.storage_capacity_bytes):
            raise ValueError('Free storage exceeds capacity')
        return self

    def identity_data(self) -> dict:
        data = self.model_dump(mode='json', exclude={'observed_at', 'storage_free_bytes'})
        if data['gpus'] is not None:
            data['gpus'] = sorted(data['gpus'], key=lambda gpu: gpu['slot'])
        return data


class RuntimeObservation(EvidenceModel):
    runtime_id: Label  # DANTE provider identity, not a server path or URL.
    runtime: Label  # Open vocabulary; discovery support is explicitly registered.
    version: Label | None = None
    backend: Label | None = None
    binary_sha256: Sha256 | None = None
    configuration_sha256: Sha256 | None = None
    capabilities: tuple[Label, ...] = ()  # Reported, not measured/qualified.
    model_inventory: tuple[Label, ...] = ()  # Opaque aliases only.
    observed_at: AwareDatetime = Field(default_factory=utc_now)
    probe_version: Label

    def identity_data(self) -> dict:
        return self.model_dump(mode='json', exclude={'observed_at', 'capabilities', 'model_inventory'})


class QualificationIdentity(EvidenceModel):
    schema_version: Literal[1] = 1
    suite_version: Label = SUITE_VERSION
    machine: MachineProfile
    runtime: RuntimeObservation
    model_id: Label
    artifact_sha256: Sha256 | None = None
    artifact_kind: Literal['file', 'runtime_manifest'] = 'file'
    quantization: Label | None = None
    context_tokens: PositiveInt | None = None
    configuration_sha256: Sha256 | None = None

    def identity_data(self) -> dict:
        return {**self.model_dump(mode='json', exclude={'machine', 'runtime'}),
                'machine': self.machine.identity_data(), 'runtime': self.runtime.identity_data()}

    @property
    def fingerprint(self) -> str:
        return digest(self.identity_data())

    @property
    def scope(self) -> str:
        return digest([str(self.machine.node_id), self.runtime.runtime_id,
                       self.runtime.backend, self.model_id])

    def missing_facts(self) -> tuple[str, ...]:
        def unknown(value):
            return value is None or (isinstance(value, str) and value.lower() in {'unknown', 'unverified', 'unqualified'})
        missing = [name for name in ('artifact_sha256', 'quantization', 'context_tokens',
                   'configuration_sha256') if unknown(getattr(self, name))]
        missing += ['runtime.' + name for name in ('version', 'backend', 'configuration_sha256')
                    if unknown(getattr(self.runtime, name))]
        missing += ['machine.' + name for name in ('os', 'os_version', 'architecture', 'cpu', 'logical_cpus', 'ram_bytes', 'gpus')
                    if unknown(getattr(self.machine, name))]
        if self.machine.gpus is not None:
            missing += ['gpu.' + gpu.slot for gpu in self.machine.gpus
                        if gpu.vram_bytes is None or unknown(gpu.driver_version)]
        if self.runtime.backend == 'cuda' and unknown(self.machine.cuda_runtime):
            missing.append('machine.cuda_runtime')
        if self.runtime.backend == 'cuda' and self.machine.gpus is not None:
            missing += ['gpu.' + gpu.slot + '.uuid' for gpu in self.machine.gpus
                        if gpu.vendor == 'NVIDIA' and unknown(gpu.uuid)]
        return tuple(missing)


class Check(StrEnum):
    LOAD = 'load'
    GENERATION = 'generation'
    STRUCTURED_OUTPUT = 'structured_output'
    TOOL_CALLING = 'tool_calling'
    CONTEXT = 'context'
    CANCELLATION = 'cancellation'
    TIMEOUT = 'timeout'
    RUNTIME_FAILURE = 'runtime_failure'
    RECOVERY = 'recovery'
    STABILITY = 'stability'


REQUIRED_CHECKS = frozenset(Check) - {Check.STRUCTURED_OUTPUT, Check.TOOL_CALLING}


class CheckEvidence(EvidenceModel):
    check: Check
    outcome: Literal['passed', 'failed', 'unknown', 'unsupported']
    evidence_sha256: Sha256 | None = None  # Digest of locally retained test evidence.
    failure: Literal['assertion_failed', 'timeout', 'unavailable', 'invalid_response', 'probe_error'] | None = None
    duration_s: Measurement | None = None

    @model_validator(mode='after')
    def coherent_outcome(self):
        if self.outcome == 'passed' and (self.evidence_sha256 is None or self.failure is not None):
            raise ValueError('A passed check requires evidence and no failure')
        if self.outcome == 'failed' and self.failure is None:
            raise ValueError('Failed checks require a safe failure code')
        return self


class PerformanceEvidence(EvidenceModel):
    peak_ram_bytes: PositiveInt | None = None
    peak_vram_bytes: PositiveInt | None = None
    load_time_s: Measurement | None = None
    prompt_tokens_per_s: Measurement | None = None
    generation_tokens_per_s: Measurement | None = None
    peak_temperature_c: Measurement | None = None


class QualificationState(StrEnum):
    UNKNOWN = 'unknown'
    QUALIFIED = 'qualified'
    FAILED = 'failed'
    STALE = 'stale'


class ModelQualification(EvidenceModel):
    qualification_id: UUID
    attempt_id: UUID | None = None  # None identifies records written before the harness.
    phase: Literal['intent', 'completed'] = 'completed'
    identity: QualificationIdentity
    source: Literal['synthetic', 'hardware']
    checks: tuple[CheckEvidence, ...] = ()
    performance: PerformanceEvidence = Field(default_factory=PerformanceEvidence)
    started_at: AwareDatetime = Field(default_factory=utc_now)
    completed_at: AwareDatetime | None = None
    interrupted: bool = False

    @model_validator(mode='after')
    def coherent_run(self):
        if len({check.check for check in self.checks}) != len(self.checks):
            raise ValueError('Duplicate check evidence')
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError('Completion precedes start')
        if self.phase == 'intent' and (self.completed_at is not None or self.checks):
            raise ValueError('Intent records cannot contain completed evidence')
        return self


class QualificationAssessment(EvidenceModel):
    state: QualificationState
    reasons: tuple[str, ...]
    measured_capabilities: tuple[Check, ...] = ()
