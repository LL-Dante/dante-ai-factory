from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml

from dante.contracts import ModelRef, LifecycleState
from dante.recovery import file_digest


class LocalQualificationGate(Protocol):
    def __call__(self, model: ModelRef, *, tool_use: bool = False) -> None: ...


class ModelRegistry:
    def __init__(self, models: list[ModelRef] | None = None, *, machine_profile: str = "default",
                 qualification_gate: LocalQualificationGate | None = None) -> None:
        self.machine_profile = machine_profile
        self.qualification_gate = qualification_gate
        self._models = {model.model_id: model for model in models or []}

    @classmethod
    def from_yaml(cls, path: Path, *, machine_profile: str = "default",
                  qualification_gate: LocalQualificationGate | None = None) -> "ModelRegistry":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([ModelRef.model_validate(item) for item in raw.get("models", [])], machine_profile=machine_profile,
                   qualification_gate=qualification_gate)

    def register(self, model: ModelRef) -> None:
        self._models[model.model_id] = model

    def candidates(self) -> list[ModelRef]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelRef:
        return self._models[model_id]


    def require_automatic(self, model: ModelRef, *, tool_use: bool = False) -> None:
        if not model.local:
            return  # Existing cloud qualification/routing behavior is unchanged.
        metadata = model.local_metadata
        if (metadata is None or metadata.qualification != 'qualified'
                or model.lifecycle not in {LifecycleState.APPROVED, LifecycleState.PRODUCTION}
                or not model.available or not metadata.identity_verified
                or not metadata.exact_identity or metadata.exact_identity.upper() in {'UNKNOWN','UNVERIFIED'}
                or not metadata.runtime_reference or not metadata.eval_refs
                or metadata.license_status != 'verified' or not metadata.license_id
                or metadata.license_id.upper() in {'UNKNOWN','UNVERIFIED'}
                or metadata.context_tokens is None or self.machine_profile not in metadata.machine_profiles
                or (tool_use and (metadata.tool_use is not True or 'tool_calling' not in model.capabilities))):
            raise ValueError('Local model is not qualified for automatic use')
        if self.qualification_gate is not None:
            self.qualification_gate(model, tool_use=tool_use)
        self.verify_identity(model)

    @staticmethod
    def verify_identity(model: ModelRef) -> None:
        metadata = model.local_metadata
        if metadata is None or not metadata.exact_identity or not metadata.runtime_reference:
            raise ValueError('Missing local model identity')
        if metadata.local_path is not None:
            try:
                if not metadata.sha256 or file_digest(metadata.local_path) != metadata.sha256:
                    raise ValueError('Local model digest mismatch')
            except OSError:
                raise ValueError('Local model artifact unavailable') from None
        elif not metadata.identity_limitations or not metadata.identity_verified:
            raise ValueError('Runtime-managed identity needs verification and explicit limitations')

    def promote(self, model_id: str, target: LifecycleState) -> ModelRef:
        model = self.get(model_id)
        stages = [LifecycleState.CANDIDATE, LifecycleState.IMPORTED, LifecycleState.VERIFIED,
                  LifecycleState.EVALUATED, LifecycleState.APPROVED, LifecycleState.PRODUCTION]
        if not model.local or model.lifecycle not in stages or stages.index(target) != stages.index(model.lifecycle) + 1:
            raise ValueError('Invalid local qualification transition')
        candidate = model.model_copy(update={'lifecycle': target})
        if stages.index(target) >= stages.index(LifecycleState.VERIFIED):
            self.verify_identity(candidate)
        if stages.index(target) >= stages.index(LifecycleState.EVALUATED) and not candidate.local_metadata.eval_refs:
            raise ValueError('Evaluation evidence required')
        if target in {LifecycleState.APPROVED, LifecycleState.PRODUCTION}:
            self.require_automatic(candidate)
        self.register(candidate)
        return candidate
