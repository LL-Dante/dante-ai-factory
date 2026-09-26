"""Local HTTP runtime adapters. Wire formats never cross this boundary."""
import json
import re
from uuid import uuid4

import httpx

from dante.contracts import ModelRef
from dante.contracts.runtime import RuntimeHealth, RuntimeProfile
from dante.contracts.inference import (AssistantMessage, InferenceRequest, InferenceResponse,
    ThinkingPolicy, ToolCall, ToolResult, Usage)
from dante.inference import (AdapterUnavailable, ContextExceeded, InferenceError, InferenceTimeout,
                             InferenceCancelled, InvalidResponse, PolicyDenied, InvalidRequest, RateQuotaUnavailable, RateLimitUnavailable,
                             QualificationDenied, retry_after_seconds, _loads)
from dante.registry import ModelRegistry


class _RuntimePayload(dict):
    def __init__(self, value, *, diagnostic):
        super().__init__(value)
        self.diagnostic = diagnostic


_SAFE_KEY = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
_SAFE_MODEL = re.compile(r'^[A-Za-z0-9._:/-]{1,128}$')
_SAFE_MEDIA_TYPE = re.compile(r'^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$')


def _json_type(value):
    if value is None: return 'null'
    if type(value) is bool: return 'boolean'
    if type(value) is str: return 'string'
    if type(value) is int: return 'integer'
    if type(value) is float: return 'number'
    if type(value) is list: return 'array'
    if type(value) is dict: return 'object'
    return 'other'


def _safe_json_keys(value):
    if not isinstance(value, dict): return []
    return [key if isinstance(key, str) and _SAFE_KEY.fullmatch(key) else '<nonstandard-key>'
            for key in list(value)[:24]]


def _safe_identifier(value, *, pattern=None, maximum=64):
    if not isinstance(value, str): return None
    if len(value) > maximum or any(ord(char) < 32 for char in value): return '<unrecognized-string>'
    if pattern is not None and not pattern.fullmatch(value): return '<unrecognized-string>'
    return value


def _response_diagnostic(status, content_type, raw_length, *, parsed=None,
                         validation_rule=None, failed_field=None, expected=None, actual=None):
    media_type = (content_type or '').split(';', 1)[0].strip()
    if not _SAFE_MEDIA_TYPE.fullmatch(media_type):
        media_type = '<unrecognized>' if content_type else None
    top = parsed if isinstance(parsed, dict) else {}
    message = top.get('message') if isinstance(top, dict) else None
    message_object = message if isinstance(message, dict) else {}
    content = message_object.get('content')
    thinking_present = 'thinking' in message_object or 'thinking' in top
    thinking = message_object.get('thinking', top.get('thinking'))
    done = top.get('done')
    model = top.get('model')
    role = message_object.get('role')
    tool_calls_present = 'tool_calls' in message_object
    tool_calls = message_object.get('tool_calls')
    done_reason = top.get('done_reason')
    diagnostic = {
        'http_status': status,
        'http_content_type': media_type,
        'response_bytes': raw_length,
        'json_decoded': parsed is not None,
        'top_level_type': _json_type(parsed) if parsed is not None else None,
        'top_level_keys': _safe_json_keys(parsed),
        'done_present': isinstance(parsed, dict) and 'done' in parsed,
        'done_type': _json_type(done) if isinstance(parsed, dict) and 'done' in parsed else None,
        'done_value': done if type(done) is bool else (
            done if type(done) in (int, float) else _safe_identifier(done,
                pattern=re.compile(r'^(stop|length|tool_calls|unknown|true|false)$'), maximum=32)),
        'model_present': isinstance(parsed, dict) and 'model' in parsed,
        'model_type': _json_type(model) if isinstance(parsed, dict) and 'model' in parsed else None,
        'model_value': _safe_identifier(model, pattern=_SAFE_MODEL, maximum=128),
        'message_present': isinstance(parsed, dict) and 'message' in parsed,
        'message_type': _json_type(message) if isinstance(parsed, dict) and 'message' in parsed else None,
        'message_keys': _safe_json_keys(message),
        'role_present': isinstance(message, dict) and 'role' in message,
        'role_type': _json_type(role) if isinstance(message, dict) and 'role' in message else None,
        'role_value': _safe_identifier(role, pattern=re.compile(r'^(assistant|user|system|tool)$'), maximum=16),
        'content_present': isinstance(message, dict) and 'content' in message,
        'content_type': _json_type(content) if isinstance(message, dict) and 'content' in message else None,
        'content_length': len(content) if isinstance(content, str) else None,
        'thinking_present': thinking_present,
        'thinking_type': _json_type(thinking) if thinking_present else None,
        'thinking_length': len(thinking) if isinstance(thinking, str) else None,
        'tool_calls_present': tool_calls_present,
        'tool_calls_type': _json_type(tool_calls) if tool_calls_present else None,
        'tool_calls_count': len(tool_calls) if isinstance(tool_calls, list) else None,
        'generation_completed': isinstance(parsed, dict) and done is True,
        'done_reason': _safe_identifier(done_reason,
            pattern=re.compile(r'^(stop|length|tool_calls|unknown)$'), maximum=32),
        'failed_validation_rule': validation_rule,
        'failed_field': failed_field,
        'expected': expected,
        'actual': actual,
    }
    return diagnostic


def _invalid_ollama_response(payload, rule, field, expected, actual):
    diagnostic = dict(getattr(payload, 'diagnostic', {}))
    diagnostic.update(failed_validation_rule=rule, failed_field=field,
                      expected=expected, actual=actual)
    raise InvalidResponse('Invalid Ollama response', diagnostic=diagnostic) from None


class LocalRuntimeAdapter:
    automatic_cost = 0.0  # Policy eligibility, not an observed monetary cost.
    runtime = ''
    default_url = ''

    def __init__(self, profile: RuntimeProfile | None = None, *, machine_profile='default', transport=None,
                 provider_id: str | None = None, registry: ModelRegistry | None = None):
        self.profile = profile or RuntimeProfile(runtime=self.runtime, base_url=self.default_url)
        if self.profile.runtime != self.runtime:
            raise ValueError('Runtime profile mismatch')
        self.provider_id = provider_id or self.runtime
        self.machine_profile, self.transport, self.registry = machine_profile, transport, registry

    def _json(self, path, body=None, *, allow_not_found=False, cancellation=None):
        try:
            with httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False,
                              timeout=self.profile.timeout_s) as client:
                with client.stream('GET' if body is None else 'POST', self.profile.base_url + path,
                                   **({} if body is None else {'json': body})) as response:
                    raw = bytearray()
                    status = response.status_code
                    content_type = response.headers.get('Content-Type')
                    for chunk in response.iter_bytes():
                        if cancellation is not None and cancellation.is_set():
                            raise InferenceCancelled('Local inference cancelled')
                        raw.extend(chunk)
                        if len(raw) > self.profile.max_response_bytes:
                            diagnostic = _response_diagnostic(status, content_type, len(raw),
                                validation_rule='response_size_limit', failed_field='response_bytes',
                                expected=f'<= {self.profile.max_response_bytes}', actual='over_limit')
                            raise InvalidResponse('Runtime response exceeds limit', diagnostic=diagnostic)
                    if response.status_code == 404 and allow_not_found:
                        return None
                    if response.status_code != 200:
                        try:
                            self._error(response.status_code, bytes(raw), retry_after_seconds(response.headers.get('Retry-After')))
                        except InferenceError as exc:
                            exc.runtime_reachable = True
                            if isinstance(exc, InvalidResponse):
                                exc.diagnostic = _response_diagnostic(status, content_type, len(raw),
                                    validation_rule='http_status_rejected', failed_field='http_status',
                                    expected='200', actual=str(status))
                            raise
        except httpx.TimeoutException:
            raise InferenceTimeout('Local runtime timed out') from None
        except httpx.TransportError:
            raise AdapterUnavailable('Local runtime unavailable') from None
        try:
            payload = _loads(bytes(raw))
        except (ValueError, TypeError):
            diagnostic = _response_diagnostic(status, content_type, len(raw),
                validation_rule='json_decode_failed', failed_field='$',
                expected='valid JSON object', actual='malformed_json')
            raise InvalidResponse('Malformed runtime JSON', diagnostic=diagnostic) from None
        if not isinstance(payload, dict):
            diagnostic = _response_diagnostic(status, content_type, len(raw), parsed=payload,
                validation_rule='top_level_not_object', failed_field='$',
                expected='object', actual=_json_type(payload))
            raise InvalidResponse('Invalid runtime envelope', diagnostic=diagnostic)
        if 'error' in payload:
            diagnostic = _response_diagnostic(status, content_type, len(raw), parsed=payload,
                validation_rule='runtime_error_envelope', failed_field='error',
                expected='absent', actual='present')
            raise InvalidResponse('Invalid runtime envelope', diagnostic=diagnostic)
        diagnostic = _response_diagnostic(status, content_type, len(raw), parsed=payload)
        return _RuntimePayload(payload, diagnostic=diagnostic)

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
            (self.registry or ModelRegistry(machine_profile=self.machine_profile)).require_automatic(
                model, tool_use=bool(request.tools))
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
        return self._complete(request)

    def complete_cancellable(self, request: InferenceRequest, cancellation) -> InferenceResponse:
        return self._complete(request, cancellation=cancellation)

    def _complete(self, request: InferenceRequest, cancellation=None) -> InferenceResponse:
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
        native_capabilities = details.get('capabilities')
        if native_capabilities is not None and (not isinstance(native_capabilities, list)
                or any(not isinstance(value, str) for value in native_capabilities)):
            raise InvalidResponse('Malformed Ollama model capabilities')
        declared_thinking = 'thinking' in getattr(request.model, 'capabilities', ())
        family = ref.split(':', 1)[0].lower()
        # Qwen3 is the known legacy-compatible native Ollama thinking family;
        # other models must declare the capability in registry or /api/show.
        supports_thinking = (declared_thinking or
            ('thinking' in native_capabilities if native_capabilities is not None else family == 'qwen3'))
        if request.thinking == ThinkingPolicy.ON and not supports_thinking:
            raise PolicyDenied('Requested thinking mode is not supported by this model')
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
        options['num_predict'] = request.max_output_tokens
        body = {'model': ref, 'messages': messages, 'stream': False, 'options': options}
        if supports_thinking:
            body['think'] = request.thinking == ThinkingPolicy.ON
        if request.tools:
            body['tools'] = [{'type': 'function', 'function': tool.model_dump()} for tool in request.tools]
        payload = self._json('/api/chat', body, cancellation=cancellation)
        if payload.get('done') is not True:
            _invalid_ollama_response(payload, 'generation_not_complete', 'done', 'true',
                                     _json_type(payload.get('done')))
        if payload.get('model') != ref:
            _invalid_ollama_response(payload, 'model_mismatch', 'model', ref,
                                     _json_type(payload.get('model')))
        if 'message' not in payload:
            _invalid_ollama_response(payload, 'message_missing', 'message', 'object', 'missing')
        message = payload['message']
        if not isinstance(message, dict):
            _invalid_ollama_response(payload, 'message_not_object', 'message', 'object', _json_type(message))
        if 'role' not in message:
            _invalid_ollama_response(payload, 'message_role_missing', 'message.role', 'assistant', 'missing')
        if message['role'] != 'assistant':
            _invalid_ollama_response(payload, 'message_role_mismatch', 'message.role', 'assistant',
                                     _safe_identifier(message['role'],
                                         pattern=re.compile(r'^(assistant|user|system|tool)$'), maximum=16)
                                     or _json_type(message['role']))

        raw_calls = message.get('tool_calls', [])
        if not isinstance(raw_calls, list):
            _invalid_ollama_response(payload, 'tool_calls_not_array', 'message.tool_calls', 'array',
                                     _json_type(raw_calls))
        calls = []
        for index, call in enumerate(raw_calls):
            prefix = f'message.tool_calls[{index}]'
            if not isinstance(call, dict):
                _invalid_ollama_response(payload, 'tool_call_not_object', prefix, 'object', _json_type(call))
            function = call.get('function')
            if not isinstance(function, dict):
                _invalid_ollama_response(payload, 'tool_function_not_object', prefix + '.function',
                                         'object', _json_type(function))
            args = function.get('arguments')
            if not isinstance(args, dict):
                _invalid_ollama_response(payload, 'tool_arguments_not_object',
                                         prefix + '.function.arguments', 'object', _json_type(args))
            name = function.get('name')
            if not isinstance(name, str) or not name:
                _invalid_ollama_response(payload, 'tool_name_invalid', prefix + '.function.name',
                                         'non-empty string', _json_type(name))
            call_id = call.get('id', 'call_' + uuid4().hex)
            if not isinstance(call_id, str) or not call_id:
                _invalid_ollama_response(payload, 'tool_call_id_invalid', prefix + '.id',
                                         'non-empty string when present', _json_type(call_id))
            calls.append(ToolCall(call_id=call_id, name=name, arguments=args))

        content = message.get('content')
        if content is not None and not isinstance(content, str):
            _invalid_ollama_response(payload, 'content_not_string', 'message.content',
                                     'string or null', _json_type(content))
        if not calls and not (content and content.strip()):
            _invalid_ollama_response(payload, 'content_empty', 'message.content',
                                     'non-empty string unless tool calls exist',
                                     'missing' if 'content' not in message else
                                     ('empty_string' if isinstance(content, str) else _json_type(content)))
        call_ids = [call.call_id for call in calls]
        if len(call_ids) != len(set(call_ids)):
            _invalid_ollama_response(payload, 'duplicate_tool_call_id', 'message.tool_calls[].id',
                                     'unique values', 'duplicate')

        for field in ('prompt_eval_count', 'eval_count'):
            value = payload.get(field)
            if value is not None and (type(value) is not int or value < 0):
                _invalid_ollama_response(payload, 'usage_count_invalid', field,
                                         'non-negative integer or null', _json_type(value))
        reason = payload.get('done_reason')
        reason = reason if reason in {'stop','length'} else 'unknown'
        if calls and reason == 'stop':
            reason = 'tool_calls'
        return InferenceResponse(model=request.model, assistant_message=self._assistant(content, calls),
            finish_reason=reason, usage=Usage(input_tokens=payload.get('prompt_eval_count'),
            output_tokens=payload.get('eval_count')), cost=None)


class LlamaCppAdapter(LocalRuntimeAdapter):
    runtime = 'llamacpp'
    default_url = 'http://127.0.0.1:8080'

    def properties(self):
        """Return optional server metadata without making `/props` mandatory."""
        return self._json('/props', allow_not_found=True)

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
        body['max_tokens'] = request.max_output_tokens
        if request.thinking == ThinkingPolicy.ON:
            raise PolicyDenied('Thinking mode is not supported by this runtime')
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
