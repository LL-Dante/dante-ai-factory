"""Bounded, serializable agent definitions and durable task/result contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dante.contracts import StrictModel, utc_now
from dante.contracts.inference import ThinkingPolicy


class AgentModelTarget(StrictModel):
    model_id: str = Field(min_length=1, max_length=200)
    runtime_reference: str = Field(min_length=1, max_length=200)
    digest_sha256: str = Field(pattern=r'^[0-9a-fA-F]{64}$')
    context_tokens: int = Field(gt=0, le=131072)
    local_only: Literal[True] = True

    @field_validator('digest_sha256')
    @classmethod
    def normalize_digest(cls, value: str) -> str:
        return value.lower().removeprefix('sha256:')


class AgentRetryPolicy(StrictModel):
    maximum_attempts: int = Field(default=2, ge=1, le=3)
    retry_base_s: float = Field(default=2, ge=0.1, le=60, allow_inf_nan=False)
    retry_max_s: float = Field(default=10, ge=0.1, le=120, allow_inf_nan=False)

    @model_validator(mode='after')
    def bounded_backoff(self):
        if self.retry_base_s > self.retry_max_s:
            raise ValueError('Retry base exceeds retry maximum')
        return self


AgentCapability = Literal['LOCAL_INFERENCE', 'READ_ONLY_INVENTORY']
READ_ONLY_CAPABILITIES = ('READ_ONLY_INVENTORY',)


class AgentDefinition(StrictModel):
    agent_id: str = Field(pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$', max_length=64)
    name: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=500)
    instructions: str = Field(min_length=1, max_length=4000)
    capabilities: tuple[AgentCapability, ...] = Field(
        default=('LOCAL_INFERENCE',), min_length=1, max_length=1)
    model_target: AgentModelTarget
    thinking: ThinkingPolicy = ThinkingPolicy.OFF
    output_token_budget: int = Field(default=384, ge=64, le=512)
    timeout_s: float = Field(default=120, gt=0, le=600, allow_inf_nan=False)
    retry_policy: AgentRetryPolicy = Field(default_factory=AgentRetryPolicy)
    created_at: datetime = Field(default_factory=utc_now)
    version: str = Field(default='1.0.0', pattern=r'^\d+\.\d+\.\d+$')

    @model_validator(mode='after')
    def local_inference_only(self):
        if self.capabilities not in (('LOCAL_INFERENCE',), READ_ONLY_CAPABILITIES):
            raise ValueError('An agent declares exactly one supported capability')
        return self


class AgentTaskPayload(StrictModel):
    agent_id: str = Field(pattern=r'^[a-z0-9]+(?:-[a-z0-9]+)*$', max_length=64)
    objective: str = Field(min_length=1, max_length=4000)
    context: str | None = Field(default=None, max_length=4000)
    requested_output_tokens: int | None = Field(default=None, ge=64, le=512)
    definition_version: str = Field(pattern=r'^\d+\.\d+\.\d+$')
    definition_digest: str = Field(pattern=r'^[0-9a-f]{64}$')

    @field_validator('objective')
    @classmethod
    def objective_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Objective must not be blank')
        return value

    @field_validator('context')
    @classmethod
    def context_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError('Context must not be blank when supplied')
        return value


class ResearchDraft(StrictModel):
    summary: str = Field(min_length=1, max_length=1200)
    findings: tuple[str, ...] = Field(min_length=1, max_length=6)
    recommended_next_actions: tuple[str, ...] = Field(min_length=1, max_length=6)
    limitations: tuple[str, ...] = Field(default=(), max_length=6)

    @field_validator('findings', 'recommended_next_actions', 'limitations')
    @classmethod
    def bound_items(cls, values):
        if any(not item.strip() or len(item) > 500 for item in values):
            raise ValueError('Research result items must be nonempty and at most 500 characters')
        return values


class AgentResult(ResearchDraft):
    agent_id: str
    model: str = Field(min_length=1, max_length=200)
    qualification_id: str = Field(min_length=1, max_length=128)
    started_at: datetime
    completed_at: datetime
