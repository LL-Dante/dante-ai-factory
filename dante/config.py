from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import yaml
from dante.contracts.runtime import RuntimeProfile
from dante.contracts.continuity import ContinuityPolicy
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator


def _validate_secret_ref(value: str) -> str:
    if not value.startswith("env://") or not value.removeprefix("env://").isidentifier():
        raise ValueError("Secret references must use env://VARIABLE")
    return value


SecretRef = Annotated[str, AfterValidator(_validate_secret_ref)]


def resolve_secret(reference: str) -> str:
    variable = reference.removeprefix("env://")
    value = os.environ.get(variable, "")
    if not value:
        raise RuntimeError(f"Secret reference is not configured: env://{variable}")
    return value


class DanteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    environment: str = "development"
    machine_profile: str = "default"
    continuity: ContinuityPolicy | None = None
    local_runtimes: dict[str, RuntimeProfile] = Field(default_factory=dict)
    bind: str = "127.0.0.1"
    database_path: Path = Path("data/dante.db")
    artifact_store: Path = Path("data/artifacts")
    workspace: Path
    max_automatic_cost: float = Field(default=0, ge=0, le=0)
    policy_version: str = "privacy-v0"
    allow_internal_cloud: bool = True
    litellm_base_url: str = "http://127.0.0.1:4000/v1"
    litellm_api_key: SecretRef = "env://LITELLM_MASTER_KEY"

    @field_validator("bind")
    @classmethod
    def loopback_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("DANTE foundation is loopback-only")
        return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, *, overrides: dict[str, Any] | None = None) -> DanteConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    return DanteConfig.model_validate(_merge(raw, overrides or {}))


def redacted_config(config: DanteConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    data["litellm_api_key"] = str(config.litellm_api_key)
    return data
