from __future__ import annotations

import http.client
import json
import math
import time
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.request
from typing import Protocol

from dante.config import resolve_secret
from dante.contracts import CostClass, RouteDecision
from dante.contracts.inference import (
    AssistantMessage, InferenceRequest, InferenceResponse, ToolCall, ToolResult, Usage,
)
from dante.registry import ModelRegistry
from dante.telemetry import JsonlAudit


class InferenceError(RuntimeError):
    """Safe, typed failure; provider payloads must not be included."""

    def __init__(self, message='', *, retry_after: float | None = None, retry_at: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after if retry_after is not None and math.isfinite(retry_after) and retry_after >= 0 else None
        self.retry_at = retry_at if retry_at is not None and math.isfinite(retry_at) and retry_at >= 0 else None


class AdapterUnavailable(InferenceError):
    pass


class InferenceTimeout(AdapterUnavailable):
    pass


class RateQuotaUnavailable(AdapterUnavailable):
    pass


class RateLimitUnavailable(RateQuotaUnavailable):
    pass


class QuotaExhausted(RateQuotaUnavailable):
    pass


class InvalidResponse(InferenceError):
    pass


class InvalidRequest(InvalidResponse):
    """Non-retryable rejected request; retains the P1 compatibility base."""


class ContextExceeded(InferenceError):
    pass


class PolicyDenied(InferenceError):
    pass


class QualificationDenied(PolicyDenied):
    pass


class ConfigurationDenied(PolicyDenied):
    pass


def retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


def reset_at_seconds(value: str | None) -> float | None:
    """Normalize a provider reset timestamp without retaining provider headers."""
    if value is None:
        return None
    try:
        reset_at = float(value)
    except (TypeError, ValueError):
        return None
    return reset_at if math.isfinite(reset_at) and reset_at >= 0 else None


# Compatibility name; responses now carry typed messages, calls and usage.
InferenceResult = InferenceResponse


class InferenceAdapter(Protocol):
    provider_id: str
    automatic_cost: float

    def complete(self, request: InferenceRequest) -> InferenceResponse: ...


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError('Non-finite JSON number')


def _loads(value):
    return json.loads(value, object_pairs_hook=_object, parse_constant=_invalid_constant)


class AICloudLiteLLMAdapter:
    provider_id = 'ai-cloud-free'
    automatic_cost = 0.0

    def __init__(self, base_url: str, api_key_ref: str) -> None:
        if not base_url.startswith(('http://127.0.0.1:', 'http://localhost:')):
            raise ValueError('AI Cloud adapter must use a loopback gateway')
        self.base_url = base_url.rstrip('/')
        self.api_key_ref = api_key_ref

    @staticmethod
    def _message(message):
        if isinstance(message, ToolResult):
            return {'role': 'tool', 'tool_call_id': message.call_id, 'content': message.content}
        result = {'role': message.role, 'content': message.content}
        if isinstance(message, AssistantMessage) and message.tool_calls:
            result['tool_calls'] = [
                {'id': call.call_id, 'type': 'function', 'function': {
                    'name': call.name, 'arguments': json.dumps(call.arguments, allow_nan=False)}}
                for call in message.tool_calls
            ]
        return result

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> InferenceError:
        if exc.code == 429:
            error_class = RateLimitUnavailable
            try:
                code = _loads(exc.read(65536)).get('error', {}).get('code')
                if code in {'insufficient_quota', 'quota_exhausted'}:
                    error_class = QuotaExhausted
            except (ValueError, AttributeError, TypeError, OSError):
                pass
            reset = exc.headers.get('RateLimit-Reset') or exc.headers.get('X-RateLimit-Reset')
            return error_class('Provider rate/quota unavailable',
                               retry_after=retry_after_seconds(exc.headers.get('Retry-After')),
                               retry_at=reset_at_seconds(reset))
        if exc.code in {408, 504}:
            return InferenceTimeout('Provider timed out', retry_after=retry_after_seconds(exc.headers.get('Retry-After')))
        if exc.code in {401, 403}:
            return PolicyDenied('Provider access denied')
        if exc.code == 400:
            try:
                error = _loads(exc.read(65536)).get('error', {})
                if error.get('code') in {'context_length_exceeded', 'context_window_exceeded'}:
                    return ContextExceeded('Context limit exceeded')
            except (ValueError, AttributeError, TypeError, OSError):
                pass
        if exc.code >= 500:
            return AdapterUnavailable('Provider unavailable', retry_after=retry_after_seconds(exc.headers.get('Retry-After')))
        return InvalidRequest('Provider rejected request')

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        # Explicit allowlist: never serialize task/trace IDs or ModelRef internals.
        body = {'model': request.model.logical_alias,
                'messages': [self._message(message) for message in request.messages],
                'temperature': request.temperature}
        if request.max_output_tokens is not None:
            body['max_tokens'] = request.max_output_tokens
        if request.tools:
            body['tools'] = [{'type': 'function', 'function': tool.model_dump()} for tool in request.tools]
        wire = urllib.request.Request(
            f'{self.base_url}/chat/completions', data=json.dumps(body, allow_nan=False).encode('utf-8'), method='POST',
            headers={'Authorization': f'Bearer {resolve_secret(self.api_key_ref)}', 'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(wire, timeout=180) as response:
                raw = response.read()
                cost_header = response.headers.get('x-litellm-response-cost')
        except urllib.error.HTTPError as exc:
            try:
                error = self._http_error(exc)
            finally:
                exc.close()
            raise error from None
        except TimeoutError:
            raise InferenceTimeout('Inference timed out') from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise InferenceTimeout('Inference timed out') from None
            raise AdapterUnavailable('Provider connection unavailable') from None
        except (OSError, http.client.HTTPException):
            raise AdapterUnavailable('Provider transport unavailable') from None
        try:
            payload = _loads(raw)
            choices = payload['choices']
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('Expected one choice')
            choice = choices[0]
            message = choice['message']
            if message.get('role', 'assistant') != 'assistant':
                raise ValueError('Invalid assistant role')
            calls = []
            for call in message.get('tool_calls', []):
                if call['type'] != 'function':
                    raise ValueError('Unsupported tool type')
                arguments = _loads(call['function']['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be an object')
                calls.append(ToolCall(call_id=call['id'], name=call['function']['name'], arguments=arguments))
            assistant = AssistantMessage(content=message.get('content'), tool_calls=tuple(calls))
            if not calls and not (assistant.content and assistant.content.strip()):
                raise ValueError('Empty assistant response')
            usage = payload.get('usage')
            usage = {} if usage is None else usage
            reason = choice.get('finish_reason')
            reason = reason if reason in {'stop', 'tool_calls', 'length', 'content_filter'} else 'unknown'
            result = InferenceResponse(
                model=request.model, assistant_message=assistant, finish_reason=reason,
                usage=Usage(input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'),
                            total_tokens=usage.get('total_tokens')),
                cost=None if cost_header is None else float(cost_header),
            )
            if reason == 'tool_calls' and not calls:
                raise ValueError('Missing tool calls')
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise InvalidResponse('Invalid inference response') from None
        if result.cost is not None and result.cost != 0:
            raise PolicyDenied('Non-zero inference cost rejected')
        return result


class DeterministicFreeAdapter:
    automatic_cost = 0.0

    def __init__(self, provider_id: str, response: str = 'FREE_PROVIDER_OK', *, unavailable: bool = False) -> None:
        self.provider_id, self.response, self.unavailable = provider_id, response, unavailable

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        if self.unavailable:
            raise AdapterUnavailable('Simulated provider unavailable')
        # This fixture has a known zero cost; it is not inferred from absent metadata.
        return InferenceResponse(model=request.model, assistant_message=AssistantMessage(content=self.response),
                                 finish_reason='stop', cost=0)


class InferenceGateway:
    def __init__(self, registry: ModelRegistry, adapters: list[InferenceAdapter], audit: JsonlAudit | None = None) -> None:
        self.registry = registry
        self.adapters = {adapter.provider_id: adapter for adapter in adapters}
        self.audit = audit

    def infer(self, decision: RouteDecision, request: InferenceRequest) -> InferenceResponse:
        if request.model != decision.selected_model:
            raise PolicyDenied('Request model does not match route')
        if request.task_id not in {None, decision.task_id} or request.trace_id not in {None, decision.trace_id}:
            raise PolicyDenied('Request metadata does not match route')
        last_error = None
        for index, model_id in enumerate(decision.candidates):
            model = self.registry.get(model_id)
            adapter = self.adapters.get(model.provider_id)
            if model.local != decision.selected_model.local:
                continue  # Never cross the cloud/local boundary on failure.
            if (model.cost_class not in {CostClass.ZERO, CostClass.LOCAL_COMPUTE}
                    or (model.cost_class == CostClass.LOCAL_COMPUTE and not model.local)
                    or not adapter or adapter.automatic_cost != 0):
                continue
            try:
                self.registry.require_automatic(model, tool_use=bool(request.tools))
            except ValueError:
                raise QualificationDenied('Local model qualification failed') from None
            try:
                result = adapter.complete(request.model_copy(update={'model': model}))
                if result.model != model:
                    raise InvalidResponse('Inference adapter returned a mismatched model')
                if result.cost is not None and result.cost != 0:
                    raise PolicyDenied('Non-zero inference cost rejected')
                result = result.model_copy(update={'fallback': index > 0})
                if self.audit:
                    self.audit.write('inference.completed', provider=model.provider_id, model=model.model_id,
                                     fallback=result.fallback, cost=result.cost)
                return result
            except AdapterUnavailable as exc:
                last_error = exc
                if self.audit:
                    self.audit.write('inference.failed', provider=model.provider_id, model=model.model_id,
                                     error_type=type(exc).__name__)
        if last_error is not None:
            raise last_error
        raise PolicyDenied('No eligible inference adapter')

    def infer_once(self, decision: RouteDecision, request: InferenceRequest) -> InferenceResponse:
        """Execute one selected cloud or local route; Continuity Manager owns reevaluation."""
        if decision.candidates != (decision.selected_model.model_id,):
            raise ConfigurationDenied('Single-route decision required')
        return self.infer(decision, request)

    def complete(self, decision: RouteDecision, messages: list[dict[str, str]]) -> InferenceResponse:
        """Legacy request bridge; the response and adapter protocol remain typed."""
        return self.infer(decision, InferenceRequest(model=decision.selected_model, messages=messages,
                          task_id=decision.task_id, trace_id=decision.trace_id))
