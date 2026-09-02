from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from dante.contracts.runtime import LocalModelMetadata


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class TaskStatus(StrEnum):
    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    WAITING_APPROVAL = "waiting_approval"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    CANCELLED = "cancelled"


class CostClass(StrEnum):
    ZERO = "zero"
    PAID = "paid"
    LOCAL_COMPUTE = "local_compute"
    COST_UNVERIFIED = "cost_unverified"


class LifecycleState(StrEnum):
    CANDIDATE = "candidate"
    IMPORTED = "imported"
    VERIFIED = "verified"
    EVALUATED = "evaluated"
    TESTED = "tested"
    APPROVED = "approved"
    PRODUCTION = "production"
    RETIRED = "retired"


class Task(StrictModel):
    task_id: str = Field(default_factory=lambda: new_id("tsk"))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: TaskStatus = TaskStatus.CREATED
    goal: str = Field(min_length=1, max_length=10_000)
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL
    selected_route: str | None = None
    workspace: str
    attempts: int = Field(default=0, ge=0)
    current_step: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    approval_state: str = "not_required"
    trace_id: str = Field(default_factory=lambda: new_id("trc"))
    checkpoint: dict[str, Any] = Field(default_factory=dict)


class ModelRef(StrictModel):
    model_id: str
    provider_id: str
    logical_alias: str
    version: str
    capabilities: frozenset[str] = frozenset()
    cost_class: CostClass = CostClass.ZERO
    privacy_eligibility: frozenset[PrivacyClass]
    available: bool = True
    lifecycle: LifecycleState = LifecycleState.CANDIDATE
    runtime: str
    local: bool = False
    local_metadata: LocalModelMetadata | None = None
    context_tokens: int | None = Field(default=None, gt=0)
    cost_verification: Literal['COST_UNVERIFIED', 'VERIFIED_ZERO'] = 'COST_UNVERIFIED'


class RouteDecision(StrictModel):
    route_id: str = Field(default_factory=lambda: new_id("rte"))
    task_id: str
    trace_id: str
    selected_model: ModelRef
    candidates: tuple[str, ...]
    reasons: tuple[str, ...]
    policy_version: str
    automatic_cost: float = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class ToolManifest(StrictModel):
    tool_id: str
    version: str = "1"
    permissions: frozenset[str]
    risk: str
    filesystem_scope: str
    network_scope: str = "none"
    secret_access: str = "none"
    approval_policy: str = "never"
    timeout_s: int = Field(default=30, gt=0)
    output_limit_bytes: int = Field(default=500_000, gt=0)
    arguments_schema: dict[str, Any] | None = None
    path_permissions: dict[str, str] = Field(default_factory=dict)


class Artifact(StrictModel):
    artifact_id: str = Field(default_factory=lambda: new_id("art"))
    task_id: str
    media_type: str
    path: str
    checksum_sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    producer: str
    provenance: dict[str, Any]
    eval_status: str = "not_evaluated"

    @field_validator("checksum_sha256")
    @classmethod
    def valid_checksum(cls, value: str) -> str:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise ValueError("checksum_sha256 must be 64 hexadecimal characters")
        return value.lower()


class Approval(StrictModel):
    approval_id: str = Field(default_factory=lambda: new_id("apr"))
    task_id: str
    action: str
    target: str
    plan_hash: str
    status: str = "pending"
    requested_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None
    decided_by: str | None = None


class EvalRun(StrictModel):
    eval_run_id: str = Field(default_factory=lambda: new_id("evl"))
    suite: str
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: str = "running"
    results: list[dict[str, Any]] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)


__all__ = [
    "Approval", "Artifact", "CostClass", "EvalRun", "LifecycleState", "ModelRef",
    "PrivacyClass", "RouteDecision", "Task", "TaskStatus", "ToolManifest", "new_id", "utc_now",
]
