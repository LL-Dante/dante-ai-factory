"""Bounded, observation-only llama.cpp discovery for Node 0."""
from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, field_validator

from dante.contracts.qualification import EvidenceModel, Label, RuntimeObservation
from dante.contracts.runtime import RuntimeProfile
from dante.inference import (AdapterUnavailable, InferenceError, InferenceTimeout,
                             InvalidResponse, PolicyDenied, RateQuotaUnavailable)
from dante.local_runtime import LlamaCppAdapter
from dante.recovery import digest


RuntimeReference = Annotated[str, Field(min_length=1, max_length=512)]
Backend = Literal['cpu', 'cuda', 'vulkan']


def _label(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('Expected text metadata')
    value = value.strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,159}', value):
        raise ValueError('Unsafe runtime metadata')
    return value


class LlamaCppModelObservation(EvidenceModel):
    """A server-reported model reference, not a qualified model identity."""
    runtime_reference: RuntimeReference

    @field_validator('runtime_reference')
    @classmethod
    def safe_reference(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError('Invalid llama.cpp model reference')
        return value


class LlamaCppProbeResult(EvidenceModel):
    state: Literal['available', 'unavailable', 'error']
    endpoint: str
    server_state: Literal['ready'] | None = None
    build: Label | None = None
    observation: RuntimeObservation | None = None
    models: tuple[LlamaCppModelObservation, ...] = ()
    failure: Literal['unavailable', 'timeout', 'invalid_response', 'policy_denied',
                     'capacity_unavailable', 'runtime_error'] | None = None


def _failure(profile: RuntimeProfile, state: Literal['unavailable', 'error'], failure: str) -> LlamaCppProbeResult:
    return LlamaCppProbeResult(state=state, endpoint=profile.base_url, failure=failure)


def probe_llamacpp(profile: RuntimeProfile | None = None, *, transport=None,
                   runtime_id: str = 'llamacpp', backend: Backend | None = None) -> LlamaCppProbeResult:
    """Read health, inventory and optional build metadata; never execute a model."""
    profile = profile or RuntimeProfile(runtime='llamacpp', base_url=LlamaCppAdapter.default_url)
    if profile.runtime != 'llamacpp':
        raise ValueError('llama.cpp runtime profile required')
    if backend not in {None, 'cpu', 'cuda', 'vulkan'}:
        raise ValueError('Unsupported llama.cpp backend')
    adapter = LlamaCppAdapter(profile, transport=transport, provider_id=runtime_id)
    try:
        raw_inventory = adapter.inventory()
        properties = adapter.properties() or {}
        if not isinstance(properties, dict):
            raise InvalidResponse('Invalid llama.cpp properties')
        reported_backend = properties.get('backend')
        if reported_backend is not None and reported_backend not in {'cpu', 'cuda', 'vulkan'}:
            raise InvalidResponse('Invalid llama.cpp backend')
        if reported_backend is not None and backend is not None and reported_backend != backend:
            raise InvalidResponse('llama.cpp backend does not match configuration')
        selected_backend = reported_backend or backend
        version = _label(properties.get('version'))
        build = _label(properties.get('build_info'))
        references = [item['id'] for item in raw_inventory]
        if len(references) != len(set(references)):
            raise InvalidResponse('Duplicate llama.cpp model reference')
        models = tuple(sorted((LlamaCppModelObservation(runtime_reference=item['id'])
                               for item in raw_inventory), key=lambda model: model.runtime_reference))
        aliases = tuple('model-' + digest(model.runtime_reference)[:16] for model in models)
        configuration = {'profile': profile.model_dump(mode='json'), 'backend': selected_backend}
        observation = RuntimeObservation(runtime_id=runtime_id, runtime='llamacpp',
            version=version or build, backend=selected_backend,
            configuration_sha256=digest(configuration), model_inventory=aliases,
            probe_version='llamacpp-http-v1')
        return LlamaCppProbeResult(state='available', endpoint=profile.base_url,
            server_state='ready', build=build, observation=observation, models=models)
    except InferenceTimeout:
        return _failure(profile, 'unavailable', 'timeout')
    except RateQuotaUnavailable:
        return _failure(profile, 'error', 'capacity_unavailable')
    except AdapterUnavailable:
        return _failure(profile, 'unavailable', 'unavailable')
    except InvalidResponse:
        return _failure(profile, 'error', 'invalid_response')
    except PolicyDenied:
        return _failure(profile, 'error', 'policy_denied')
    except InferenceError:
        return _failure(profile, 'error', 'runtime_error')
    except (KeyError, TypeError, ValueError, AttributeError):
        return _failure(profile, 'error', 'invalid_response')
