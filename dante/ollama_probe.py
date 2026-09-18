"""Bounded, observation-only Ollama discovery for Node 0."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from dante.contracts.qualification import EvidenceModel, Label, RuntimeObservation, Sha256
from dante.contracts.runtime import RuntimeProfile
from dante.inference import (AdapterUnavailable, InferenceError, InferenceTimeout,
                             InvalidResponse, PolicyDenied, RateQuotaUnavailable)
from dante.local_runtime import OllamaAdapter
from dante.recovery import digest


RuntimeReference = Annotated[str, Field(min_length=1, max_length=512)]


class OllamaModelObservation(EvidenceModel):
    """Provider inventory data; it is evidence, not a qualified model record."""
    runtime_reference: RuntimeReference
    runtime_digest: Sha256 | None = None
    format: Label | None = None
    quantization: Label | None = None

    @field_validator('runtime_reference')
    @classmethod
    def safe_reference(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError('Invalid Ollama model reference')
        return value

    @field_validator('runtime_digest', mode='before')
    @classmethod
    def normalize_digest(cls, value):
        return value.removeprefix('sha256:').lower() if isinstance(value, str) else value


class OllamaProbeResult(EvidenceModel):
    state: Literal['available', 'unavailable', 'error']
    endpoint: str
    observation: RuntimeObservation | None = None
    models: tuple[OllamaModelObservation, ...] = ()
    failure: Literal['unavailable', 'timeout', 'invalid_response', 'policy_denied',
                     'capacity_unavailable', 'runtime_error'] | None = None


def _failure(profile: RuntimeProfile, state: Literal['unavailable', 'error'], failure: str) -> OllamaProbeResult:
    return OllamaProbeResult(state=state, endpoint=profile.base_url, failure=failure)


def probe_ollama(profile: RuntimeProfile | None = None, *, transport=None,
                 runtime_id: str = 'ollama') -> OllamaProbeResult:
    """Read version and model inventory without starting Ollama or executing a model."""
    profile = profile or RuntimeProfile(runtime='ollama', base_url=OllamaAdapter.default_url)
    if profile.runtime != 'ollama':
        raise ValueError('Ollama runtime profile required')
    adapter = OllamaAdapter(profile, transport=transport, provider_id=runtime_id)
    try:
        version = adapter.version()
        raw_inventory = adapter.inventory()
        models = []
        for item in raw_inventory:
            details = item.get('details', {})
            if details is None:
                details = {}
            if not isinstance(details, dict):
                raise InvalidResponse('Invalid Ollama model details')
            models.append(OllamaModelObservation(
                runtime_reference=item['id'], runtime_digest=item.get('digest'),
                format=details.get('format'), quantization=details.get('quantization_level')))
        models = tuple(sorted(models, key=lambda model: model.runtime_reference))
        aliases = tuple('model-' + digest(model.runtime_reference)[:16] for model in models)
        observation = RuntimeObservation(runtime_id=runtime_id, runtime='ollama', version=version,
            configuration_sha256=digest(profile.model_dump(mode='json')), model_inventory=aliases,
            probe_version='ollama-http-v1')
        return OllamaProbeResult(state='available', endpoint=profile.base_url,
                                 observation=observation, models=models)
    except InferenceTimeout:
        return _failure(profile, 'unavailable', 'timeout')
    except AdapterUnavailable:
        return _failure(profile, 'unavailable', 'unavailable')
    except InvalidResponse:
        return _failure(profile, 'error', 'invalid_response')
    except PolicyDenied:
        return _failure(profile, 'error', 'policy_denied')
    except RateQuotaUnavailable:
        return _failure(profile, 'error', 'capacity_unavailable')
    except InferenceError:
        return _failure(profile, 'error', 'runtime_error')
    except (KeyError, TypeError, ValueError, AttributeError):
        return _failure(profile, 'error', 'invalid_response')
