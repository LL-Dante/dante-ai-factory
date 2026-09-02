"""Local qualification and runtime configuration; absent facts remain unknown."""
from datetime import datetime, timezone
import ipaddress
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


class LocalModelMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    exact_identity: str | None = None
    runtime_reference: str | None = None
    local_path: Path | None = None
    sha256: str | None = None
    runtime_digest: str | None = None
    format: str | None = None
    quantization: str | None = None
    context_tokens: StrictInt | None = Field(default=None, gt=0)
    tool_use: bool | None = None
    license_id: str | None = None
    license_status: Literal['unknown', 'unverified', 'verified', 'rejected'] = 'unknown'
    machine_profiles: frozenset[str] = frozenset()
    qualification: Literal['unverified', 'qualified', 'rejected'] = 'unverified'
    identity_verified: bool = False
    identity_limitations: tuple[str, ...] = ()
    eval_refs: tuple[str, ...] = ()
    benchmark_refs: tuple[str, ...] = ()

    @field_validator('sha256', 'runtime_digest')
    @classmethod
    def checksum(cls, value):
        if value is not None:
            value = value.removeprefix('sha256:').lower()
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Expected SHA-256 digest')
        return value


class RuntimeProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    runtime: Literal['ollama', 'llamacpp']
    base_url: str
    allow_remote: bool = False
    timeout_s: float = Field(default=60, gt=0, allow_inf_nan=False)
    max_response_bytes: int = Field(default=2_000_000, gt=0)

    @model_validator(mode='after')
    def endpoint(self):
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in {'','/'}):
            raise ValueError('Runtime endpoint must be an HTTP origin without credentials')
        _ = parsed.port
        host = parsed.hostname
        if host == 'localhost':
            host = '127.0.0.1'  # Avoid DNS/proxy escape of the default loopback boundary.
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if not local and not self.allow_remote:
            raise ValueError('Remote runtime requires explicit authorization')
        host = f'[{host}]' if ':' in host else host
        self.base_url = f'{parsed.scheme}://{host}' + (f':{parsed.port}' if parsed.port else '')
        return self


class RuntimeHealth(BaseModel):
    model_config = ConfigDict(extra='forbid')
    runtime: str
    runtime_reachable: bool
    model_present: bool | None = None
    adapter_operational: bool = False
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
