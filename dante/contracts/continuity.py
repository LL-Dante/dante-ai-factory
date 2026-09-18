"""Backend operation and continuation policy are independent of model qualification."""
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BackendState(StrEnum):
    AVAILABLE='available'
    DEGRADED='degraded'
    RATE_LIMITED='rate_limited'
    QUOTA_EXHAUSTED='quota_exhausted'
    UNAVAILABLE='unavailable'
    COOLDOWN='cooldown'
    UNKNOWN='unknown'


class ContinuityDisposition(StrEnum):
    CONTINUE_NOW='continue_now'
    RETRY_LATER='retry_later'
    TERMINAL='terminal'


class ExecutionPolicy(StrEnum):
    LOCAL_ONLY='LOCAL_ONLY'
    LOCAL_PREFERRED='LOCAL_PREFERRED'
    CLOUD_PREFERRED='CLOUD_PREFERRED'
    CLOUD_ONLY='CLOUD_ONLY'
    # Accepted for existing P6 configuration compatibility.
    LOCAL_FIRST='LOCAL_FIRST'
    CLOUD_FIRST_WITH_LOCAL_FALLBACK='CLOUD_FIRST_WITH_LOCAL_FALLBACK'
    SPECIFIC_ALLOWED_BACKENDS='SPECIFIC_ALLOWED_BACKENDS'


class ContinuityPolicy(BaseModel):
    model_config=ConfigDict(extra='forbid')
    mode: ExecutionPolicy=ExecutionPolicy.LOCAL_ONLY
    allowed_backends: frozenset[str]=frozenset()
    base_backoff_s: float=Field(default=5,gt=0,allow_inf_nan=False)
    max_backoff_s: float=Field(default=300,gt=0,allow_inf_nan=False)
    observation_ttl_s: float=Field(default=300,gt=0,allow_inf_nan=False)
    window_s: float=Field(default=60,gt=0,allow_inf_nan=False)
    max_route_attempts: int=Field(default=4,ge=1,le=100)

    @model_validator(mode='after')
    def bounded_backoff(self):
        if self.max_backoff_s < self.base_backoff_s:
            raise ValueError('Maximum backoff must be at least the base backoff')
        return self
