"""Local HTTP runtime adapters. Wire formats never cross this boundary."""
import json
from uuid import uuid4

import httpx

from dante.contracts import ModelRef
from dante.contracts.runtime import RuntimeHealth, RuntimeProfile
from dante.contracts.inference import AssistantMessage, InferenceRequest, InferenceResponse, ToolCall, ToolResult, Usage
from dante.inference import (AdapterUnavailable, ContextExceeded, InferenceError, InferenceTimeout,
                             InvalidResponse, PolicyDenied, InvalidRequest, RateQuotaUnavailable, RateLimitUnavailable,
                             QualificationDenied, retry_after_seconds, _loads)
from dante.registry import ModelRegistry


class LocalRuntimeAdapter:
    automatic_cost = 0.0  # Policy eligibility, not an observed monetary cost.
    runtime = ''
    default_url = ''

    def __init__(self, profile: RuntimeProfile | None = None, *, machine_profile='default', transport=None,
                 provider_id: str | None = None):
        self.profile = profile or RuntimeProfile(runtime=self.runtime, base_url=self.default_url)
        if self.profile.runtime != self.runtime:
            raise ValueError('Runtime profile mismatch')
        self.provider_id = provider_id or self.runtime
        self.machine_profile, self.transport = machine_profile, transport

    def _json(self, path, body=None):
        try:
            with httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
                              timeout=self.profile.timeout_s) as client:
                with client.stream('GET' if body is None else 'POST', self.profile.base_url + path,
                                   **({} if body is None else {'json': body})) as response:
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > self.profile.max_response_bytes:
                            raise InvalidResponse('Runtime response exceeds limit')
                    if response.status_code != 200:
                        try:
                            self._error(response.status_code, bytes(raw), retry_after_seconds(response.headers.get('Retry-After')))
                        except InferenceError as exc:
                            exc.runtime_reachable = True
                            raise
            payload = _loads(bytes(raw))
            if not isinstance(payload, dict) or 'error' in payload:
                raise InvalidResponse('Invalid runtime envelope')
            return payload
        except httpx.TimeoutException:
            raise InferenceTimeout('Local runtime timed out') from None
        except httpx.TransportError:
            raise AdapterUnavailable('Local runtime unavailable') from None
        except (ValueError, TypeError):
            raise InvalidResponse('Malformed runtime JSON') from None

    @staticmethod
    def _error(status, raw, retry_after=None):
        if status in {408,504}:
            raise InferenceTimeout('Local runtime timed out', retry_after=retry_after)
        if status == 429:
            raise RateLimitUnavailable('Local runtime capacity unavailable', retry_after=retry_after)
        if status in {401,403} or 300 <= status < 400:
            raise PolicyDenied('Runtime access or redirect denied')
        if status in {400,413,422}:
            text = raw.decode('utf-8', errors='replace').lower()
            if any(marker in text for marker in ('context_length_exceeded', 'context_window_exceeded',
                   'exceeds the available context', 'exceeds the context', 'context length', 'context window')):
                raise ContextExceeded('Local context limit exceeded')
        if status == 404 or status >= 500:
            raise AdapterUnavailable('Local runtime or model unavailable', retry_after=retry_after)
        raise InvalidRequest('Local runtime rejected request')

    def inventory(self):
        raise NotImplementedError

    def health(self, model: ModelRef | None = None) -> RuntimeHealth:
        reachable = False
        present = None
        try:
            models = self.inventory()
            reachable = True
            if model is not None:
                ref = model.local_metadata.runtime_reference if model.local_metadata else None
                if ref is None:
                    return RuntimeHealth(runtime=self.runtime, runtime_reachable=True, adapter_operational=True,
                                         error='model_reference_unknown')
                present = any(item['id'] == ref for item in models)
                if not present:
                    return RuntimeHealth(runtime=self.runtime, runtime_reachable=True, model_present=False,
                                         adapter_operational=True, error='model_not_present')
            return RuntimeHealth(runtime=self.runtime, runtime_reachable=True, model_present=present, adapter_operational=True)
        except InferenceError as exc:
            # A valid HTTP response with malformed data still proves transport reachability.
            reachable = reachable or getattr(exc, "runtime_reachable", False) or isinstance(exc, (InvalidResponse, PolicyDenied, RateQuotaUnavailable))
            return RuntimeHealth(runtime=self.runtime, runtime_reachable=reachable, model_present=present,
                                 error=type(exc).__name__)

    def _qualified(self, request):
        model = request.model
        if not model.local or model.runtime != self.runtime or model.provider_id != self.provider_id:
            raise PolicyDenied('Local runtime/model mismatch')
        try:
            ModelRegistry(machine_profile=self.machine_profile).require_automatic(model, tool_use=bool(request.tools))
        except ValueError:
            raise QualificationDenied('Local model qualification or integrity failed') from None
        if request.max_output_tokens is not None and request.max_output_tokens > model.local_metadata.context_tokens:
            raise ContextExceeded('Output request exceeds declared context')
        return model.local_metadata.runtime_reference

    @staticmethod
    def _assistant(content, calls):
        message = AssistantMessage(content=content, tool_calls=tuple(calls))
        if not calls and not (message.content and message.content.strip()):
            raise ValueError('Empty assistant response')
        return message


class OllamaAdapter(LocalRuntimeAdapter):
    runtime = 'ollama'
    default_url = 'http://127.0.0.1:11434'

    def version(self):
        try:
            version = self._json('/api/version')['version']
            if not isinstance(version, str) or not version.strip():
                raise ValueError('Expected version string')
            return version.strip()
        except (KeyError, TypeError, ValueError, AttributeError):
            raise InvalidResponse('Invalid Ollama version response') from None

    def inventory(self):
        try:
            models = self._json('/api/tags')['models']
            if not isinstance(models, list) or any(not isinstance(item.get('name'), str) for item in models):
                raise ValueError('Expected models list')
            return [dict(item, id=item['name']) for item in models]
        except (KeyError, TypeError, ValueError, AttributeError):
            raise InvalidResponse('Invalid Ollama model inventory') from None

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        ref = self._qualified(request)
        entries = [entry for entry in self.inventory() if entry['id'] == ref]
        if len(entries) != 1:
            raise AdapterUnavailable('Local model is not present')
        entry = entries[0]
        # Do not let the local daemon route this adapter to Ollama cloud models.
        if not isinstance(entry.get('details'), dict) or not isinstance(entry.get('digest'), str):
            raise InvalidResponse('Malformed Ollama model identity')
        if ('cloud' in ref.lower() or entry.get('remote_host') or entry.get('remote_model')
                or entry.get('details', {}).get('format') != 'gguf'
                or not request.model.local_metadata.runtime_digest
                or entry.get('digest', '').removeprefix('sha256:') != request.model.local_metadata.runtime_digest):
            raise PolicyDenied('Local model identity not verified')
        details = self._json('/api/show', {'model': ref})
        if not isinstance(details.get('details'), dict) or not isinstance(details.get('model_info'), dict):
            raise InvalidResponse('Malformed Ollama model details')
        if (details.get('remote_host') or details.get('remote_model')
                or details.get('details', {}).get('format') != 'gguf' or not details.get('model_info')):
            raise PolicyDenied('Local model execution not verified')
        messages, names = [], {}
        for message in request.messages:
            if isinstance(message, ToolResult):
                messages.append({'role': 'tool', 'tool_name': names[message.call_id], 'content': message.content})
            else:
                item = {'role': message.role, 'content': message.content or ''}
                if isinstance(message, AssistantMessage) and message.tool_calls:
                    item['tool_calls'] = [{'function': {'name': c.name, 'arguments': c.arguments}} for c in message.tool_calls]
                    names.update({c.call_id: c.name for c in message.tool_calls})
                messages.append(item)
        options = {'temperature': request.temperature, 'num_ctx': request.model.local_metadata.context_tokens}
        if request.max_output_tokens is not None:
            options['num_predict'] = request.max_output_tokens
        body = {'model': ref, 'messages': messages, 'stream': False, 'options': options}
        if request.tools:
            body['tools'] = [{'type': 'function', 'function': tool.model_dump()} for tool in request.tools]
        payload = self._json('/api/chat', body)
        try:
            if payload.get('done') is not True or payload.get('model') != ref:
                raise ValueError('Incomplete or mismatched runtime response')
            message = payload['message']
            if message['role'] != 'assistant':
                raise ValueError('Invalid role')
            calls = []
            raw_calls = message.get('tool_calls', [])
            if not isinstance(raw_calls, list):
                raise ValueError('Invalid tool calls')
            for call in raw_calls:
                args = call['function']['arguments']
                if not isinstance(args, dict):
                    raise ValueError('Ollama tool arguments must be an object')
                calls.append(ToolCall(call_id=call['id'] if 'id' in call else 'call_' + uuid4().hex,
                                      name=call['function']['name'], arguments=args))
            reason = payload.get('done_reason')
            reason = reason if reason in {'stop','length'} else 'unknown'
            if calls and reason == 'stop':
                reason = 'tool_calls'
            return InferenceResponse(model=request.model, assistant_message=self._assistant(message.get('content'), calls),
                finish_reason=reason, usage=Usage(input_tokens=payload.get('prompt_eval_count'),
                output_tokens=payload.get('eval_count')), cost=None)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise InvalidResponse('Invalid Ollama response') from None


class LlamaCppAdapter(LocalRuntimeAdapter):
    runtime = 'llamacpp'
    default_url = 'http://127.0.0.1:8080'

    def inventory(self):
        try:
            health = self._json('/health')
            if health.get('status') != 'ok':
                error = AdapterUnavailable('llama.cpp server is not ready')
                error.runtime_reachable = True
                raise error
            models = self._json('/v1/models')['data']
            if not isinstance(models, list) or any(not isinstance(item.get('id'), str) for item in models):
                raise ValueError('Invalid model list')
            return models
        except (KeyError, TypeError, ValueError, AttributeError):
            raise InvalidResponse('Invalid llama.cpp model inventory') from None

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        ref = self._qualified(request)
        if not any(item['id'] == ref for item in self.inventory()):
            raise AdapterUnavailable('Local model is not present')
        messages = []
        for message in request.messages:
            if isinstance(message, ToolResult):
                messages.append({'role': 'tool', 'tool_call_id': message.call_id, 'content': message.content})
            else:
                item = {'role': message.role, 'content': message.content}
                if isinstance(message, AssistantMessage) and message.tool_calls:
                    item['tool_calls'] = [{'id': c.call_id, 'type': 'function', 'function':
                        {'name': c.name, 'arguments': json.dumps(c.arguments, allow_nan=False)}} for c in message.tool_calls]
                messages.append(item)
        body = {'model': ref, 'messages': messages, 'stream': False, 'temperature': request.temperature}
        if request.max_output_tokens is not None:
            body['max_tokens'] = request.max_output_tokens
        if request.tools:
            body['tools'] = [{'type': 'function', 'function': tool.model_dump()} for tool in request.tools]
        payload = self._json('/v1/chat/completions', body)
        try:
            choices = payload['choices']
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('Expected one choice')
            message = choices[0]['message']
            if message['role'] != 'assistant' or payload.get('model', ref) != ref:
                raise ValueError('Invalid assistant/model')
            calls = []
            raw_calls = message.get('tool_calls', [])
            if not isinstance(raw_calls, list):
                raise ValueError('Invalid tool calls')
            for call in raw_calls:
                if call['type'] != 'function':
                    raise ValueError('Unsupported tool type')
                args = _loads(call['function']['arguments'])
                if not isinstance(args, dict):
                    raise ValueError('Tool arguments must be an object')
                calls.append(ToolCall(call_id=call['id'], name=call['function']['name'], arguments=args))
            reason = choices[0].get('finish_reason')
            reason = reason if reason in {'stop','length','tool_calls','content_filter'} else 'unknown'
            if reason == 'tool_calls' and not calls:
                raise ValueError('Missing tool calls')
            usage = payload.get('usage')
            usage = {} if usage is None else usage
            if not isinstance(usage, dict):
                raise ValueError('Invalid usage metadata')
            return InferenceResponse(model=request.model, assistant_message=self._assistant(message.get('content'), calls),
                finish_reason=reason, usage=Usage(input_tokens=usage.get('prompt_tokens'),
                output_tokens=usage.get('completion_tokens'), total_tokens=usage.get('total_tokens')), cost=None)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise InvalidResponse('Invalid llama.cpp response') from None


def configured_adapters(config):
    """Separate runtime endpoints and machine eligibility from application code."""
    classes = {'ollama': OllamaAdapter, 'llamacpp': LlamaCppAdapter}
    return [classes[profile.runtime](profile, provider_id=provider_id, machine_profile=config.machine_profile)
            for provider_id, profile in config.local_runtimes.items()]
