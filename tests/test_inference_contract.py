"""Offline P1 adapter fixtures; urlopen is always mocked."""
import io
import json
import unittest
import tempfile
from pathlib import Path
import urllib.error
from unittest.mock import patch

from pydantic import ValidationError

from dante.contracts import ModelRef, PrivacyClass, RouteDecision, LifecycleState
from dante.agent_host import AgentHostFoundation
from dante.ledger import TaskLedger
from dante.privacy import PrivacyGate
from dante.routing import RuleBasedRouter, RouteDenied
from dante.tool_broker import ToolBroker
from dante.contracts.inference import (
    AssistantMessage, InferenceRequest, ToolCall, ToolDefinition, ToolResult,
)
from dante.inference import (
    AICloudLiteLLMAdapter, AdapterUnavailable, ContextExceeded, DeterministicFreeAdapter,
    InferenceGateway, InferenceTimeout, InvalidResponse, PolicyDenied, RateQuotaUnavailable,
)
from dante.registry import ModelRegistry


def model(provider='ai-cloud-free'):
    return ModelRef(model_id=provider, provider_id=provider, logical_alias='fixture', version='1',
                    privacy_eligibility=frozenset({PrivacyClass.PUBLIC}), runtime='fixture')


def request(**updates):
    return InferenceRequest(model=model(), messages=[{'role': 'user', 'content': 'public fixture'}], **updates)


def call(call_id='call-1', arguments='{"path":"a.txt","nested":{"n":2}}'):
    return {'id': call_id, 'type': 'function', 'function': {'name': 'read_file', 'arguments': arguments}}


def payload(content='hello', calls=None, reason='stop'):
    message = {'role': 'assistant', 'content': content}
    if calls is not None:
        message['tool_calls'] = calls
    return {'choices': [{'message': message, 'finish_reason': reason}]}


class Fixture(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(json.dumps(data).encode())
        self.headers = headers or {}


class TypedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.adapter = AICloudLiteLLMAdapter('http://127.0.0.1:4000/v1', 'env://FIXTURE_ONLY')
        self.secret = patch('dante.inference.resolve_secret', return_value='fixture-only')
        self.secret.start()
        self.addCleanup(self.secret.stop)
        self.transport = patch('dante.inference.urllib.request.urlopen')
        self.urlopen = self.transport.start()
        self.addCleanup(self.transport.stop)

    def complete(self, data, headers=None, req=None):
        self.urlopen.return_value = Fixture(data, headers)
        return self.adapter.complete(req or request())

    def test_text_and_unknown_metadata(self):
        result = self.complete(payload())
        self.assertEqual(result.assistant_message, AssistantMessage(content='hello'))
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(result.finish_reason, 'stop')
        self.assertIsNone(result.cost)
        self.assertIsNone(result.usage.input_tokens)
        self.assertIsNone(result.usage.total_tokens)

    def test_one_tool_call_with_no_text(self):
        result = self.complete(payload(None, [call()], 'tool_calls'))
        self.assertEqual(result.tool_calls[0].call_id, 'call-1')
        self.assertEqual(result.tool_calls[0].arguments, {'path': 'a.txt', 'nested': {'n': 2}})
        self.assertIsNone(result.assistant_message.content)

    def test_two_calls_preserved_even_with_stop(self):
        result = self.complete(payload(None, [call(), call('call-2')]))
        self.assertEqual([c.call_id for c in result.tool_calls], ['call-1', 'call-2'])
        self.assertEqual(result.finish_reason, 'stop')

    def test_malformed_arguments_rejected(self):
        for args in ['{', '[]', 'null', '"text"', '{"x":NaN}', '{"x":1,"x":2}', {'x': 1}]:
            with self.subTest(args=args), self.assertRaises(InvalidResponse):
                self.complete(payload(None, [call(arguments=args)]))

    def test_nonfinite_typed_arguments_rejected(self):
        with self.assertRaises(ValidationError):
            ToolCall(call_id='a', name='read', arguments={'x': float('nan')})

    def test_invalid_calls_rejected(self):
        cases = [payload(None, [call(), call()]), payload(None, [{'type': 'function'}]),
                 payload('hello', reason='tool_calls'), payload(None, [call(call_id='')]),
                 payload('hello', calls='not-a-list')]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(InvalidResponse):
                self.complete(data)

    def test_empty_invalid_response(self):
        for data in [payload(''), payload('  '), payload(None), {'choices': []}, {}, [], payload(12)]:
            with self.subTest(data=data), self.assertRaises(InvalidResponse):
                self.complete(data)
        self.urlopen.return_value = type('BadFixture', (Fixture,), {'read': lambda self: b'not json'})(payload())
        with self.assertRaises(InvalidResponse):
            self.adapter.complete(request())

    def test_usage_present_partial_and_invalid(self):
        data = payload()
        data['usage'] = {'prompt_tokens': 7, 'completion_tokens': 2}
        result = self.complete(data, {'x-litellm-response-cost': '0'})
        self.assertEqual(result.usage.input_tokens, 7)
        self.assertEqual(result.usage.output_tokens, 2)
        self.assertIsNone(result.usage.total_tokens)
        self.assertEqual(result.cost, 0)
        for value in [-1, '2', True]:
            data['usage'] = {'prompt_tokens': value}
            with self.assertRaises(InvalidResponse):
                self.complete(data)

    def test_cost_invalid_or_nonzero(self):
        for value in ['NaN', 'inf', '-1', 'bad']:
            with self.assertRaises(InvalidResponse):
                self.complete(payload(), {'x-litellm-response-cost': value})
        with self.assertRaises(PolicyDenied):
            self.complete(payload(), {'x-litellm-response-cost': '0.1'})

    def test_timeouts(self):
        for exc in [TimeoutError('private'), urllib.error.URLError(TimeoutError('private'))]:
            self.urlopen.side_effect = exc
            with self.assertRaises(InferenceTimeout) as caught:
                self.adapter.complete(request())
            self.assertNotIn('private', str(caught.exception))

    def test_http_error_mapping(self):
        for code, expected in [(429, RateQuotaUnavailable), (503, AdapterUnavailable),
                               (408, InferenceTimeout), (504, InferenceTimeout),
                               (401, PolicyDenied), (403, PolicyDenied), (400, InvalidResponse)]:
            self.urlopen.side_effect = urllib.error.HTTPError('http://fixture', code, 'private', {}, io.BytesIO(b'{}'))
            with self.subTest(code=code), self.assertRaises(expected) as caught:
                self.adapter.complete(request())
            self.assertNotIn('private', str(caught.exception))

    def test_context_and_connection_errors(self):
        body = json.dumps({'error': {'code': 'context_length_exceeded', 'message': 'private'}}).encode()
        self.urlopen.side_effect = urllib.error.HTTPError('http://fixture', 400, '', {}, io.BytesIO(body))
        with self.assertRaises(ContextExceeded):
            self.adapter.complete(request())
        self.urlopen.side_effect = urllib.error.URLError('private')
        with self.assertRaises(AdapterUnavailable):
            self.adapter.complete(request())

    def test_continuation_wire_and_metadata_isolation(self):
        first = self.complete(payload(None, [call(), call('call-2')], 'tool_calls'))
        tool = ToolDefinition(name='read_file', parameters={'type': 'object', 'properties': {'path': {'type': 'string'}}})
        req = InferenceRequest(model=model(), messages=[
            {'role': 'user', 'content': 'public fixture'}, first.assistant_message,
            ToolResult(call_id='call-1', content='first'), ToolResult(call_id='call-2', content='second')],
            tools=(tool,), task_id='internal-task', trace_id='internal-trace', max_output_tokens=30, temperature=0.2)
        result = self.complete(payload('done'), req=req)
        wire = json.loads(self.urlopen.call_args.args[0].data)
        self.assertEqual(result.content, 'done')
        self.assertEqual(wire['messages'][2]['tool_call_id'], 'call-1')
        self.assertEqual(json.loads(wire['messages'][1]['tool_calls'][0]['function']['arguments']), first.tool_calls[0].arguments)
        self.assertEqual(wire['tools'][0]['function']['name'], 'read_file')
        self.assertEqual(wire['max_tokens'], 30)
        self.assertEqual(wire['temperature'], 0.2)
        self.assertEqual(set(wire), {'model', 'messages', 'tools', 'max_tokens', 'temperature'})
        self.assertNotIn('internal-', json.dumps(wire))

    def test_continuation_rejects_orphan_duplicate_missing_ids(self):
        assistant = AssistantMessage(tool_calls=(ToolCall(call_id='a', name='read', arguments={}),))
        for messages in [[ToolResult(call_id='a', content='x')], [assistant],
                         [assistant, ToolResult(call_id='a', content='x'), ToolResult(call_id='a', content='x')]]:
            with self.assertRaises(ValidationError):
                InferenceRequest(model=model(), messages=messages)

    def test_gateway_preserves_unknown_and_typed_failure(self):
        m = model()
        decision = RouteDecision(task_id='t', trace_id='r', selected_model=m, candidates=(m.model_id,), reasons=(), policy_version='test')
        gateway = InferenceGateway(ModelRegistry([m]), [self.adapter])
        self.urlopen.return_value = Fixture(payload())
        result = gateway.infer(decision, request())
        self.assertIsNone(result.cost)
        self.urlopen.side_effect = TimeoutError()
        with self.assertRaises(InferenceTimeout):
            gateway.infer(decision, request())
        with self.assertRaises(PolicyDenied):
            gateway.infer(decision, request(task_id='wrong'))

    def test_host_typed_request_tools_and_privacy(self):
        m = model().model_copy(update={'lifecycle': LifecycleState.APPROVED, 'capabilities': frozenset({'tool_calling'})})
        registry = ModelRegistry([m])
        adapter = DeterministicFreeAdapter(m.provider_id)
        with tempfile.TemporaryDirectory() as directory:
            host = AgentHostFoundation(TaskLedger(Path(directory) / 'ledger.db'), PrivacyGate(),
                                       RuleBasedRouter(registry), InferenceGateway(registry, [adapter]), ToolBroker())
            task = host.start('fixture', directory, PrivacyClass.PUBLIC)
            tool = ToolDefinition(name='read', parameters={'type': 'object'})
            with patch.object(adapter, 'complete', wraps=adapter.complete) as invoke:
                response = host.infer(task.task_id, 'public', set(), tools=(tool,))
                sent = invoke.call_args.args[0]
                self.assertIsInstance(sent, InferenceRequest)
                self.assertEqual(sent.tools, (tool,))
                self.assertEqual(sent.task_id, task.task_id)
                self.assertEqual(response.content, 'FREE_PROVIDER_OK')
            private_tool = tool.model_copy(update={'description': 'api_key=synthetic-fixture'})
            with self.assertRaises(RouteDenied):
                host.infer(task.task_id, 'public', set(), tools=(private_tool,))

    def test_gateway_fallback_keeps_typed_response(self):
        a, b = model('first'), model('second')
        decision = RouteDecision(task_id='t', trace_id='r', selected_model=a, candidates=(a.model_id, b.model_id), reasons=(), policy_version='test')
        gateway = InferenceGateway(ModelRegistry([a, b]), [DeterministicFreeAdapter('first', unavailable=True), DeterministicFreeAdapter('second')])
        result = gateway.infer(decision, InferenceRequest(model=a, messages=[{'role': 'user', 'content': 'x'}]))
        self.assertTrue(result.fallback)
        self.assertEqual(result.model, b)
        self.assertEqual(result.assistant_message.content, 'FREE_PROVIDER_OK')


if __name__ == '__main__':
    unittest.main()
