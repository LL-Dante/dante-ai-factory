import json
import tempfile
import time
import unittest
from threading import Event
from types import SimpleNamespace

from dante.contracts import CostClass, ModelRef, PrivacyClass, Task
from dante.contracts.inference import AssistantMessage, ToolCall, Usage
from dante.contracts.runtime import LocalModelMetadata
from dante.contracts.tools import ToolResult, ToolStatus
from dante.dev_agent_loop import DevelopmentAgentLoop
from dante.inference import InferenceCancelled


def model():
    return ModelRef(model_id='qwen-dev', provider_id='dev-qwen', logical_alias='qwen', version='dev',
        runtime='ollama', local=True, cost_class=CostClass.LOCAL_COMPUTE,
        privacy_eligibility={PrivacyClass.INTERNAL},
        local_metadata=LocalModelMetadata(runtime_reference='dante-qwen-agent:latest',
            runtime_digest='a' * 64, context_tokens=4096))


def response(content=None, calls=(), input_tokens=1, output_tokens=2, reason='stop'):
    return SimpleNamespace(assistant_message=AssistantMessage(content=content, tool_calls=tuple(calls)),
        content=content or '', tool_calls=tuple(calls), finish_reason=reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens))


class FakeAdapter:
    def __init__(self, replies, delay=0):
        self.replies, self.requests, self.delay = list(replies), [], delay

    def complete_cancellable(self, request, cancellation):
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        if cancellation.is_set():
            raise InferenceCancelled()
        return self.replies.pop(0)


class FakeBroker:
    def __init__(self):
        self.calls = []

    def manifest(self, _tool_id):
        return SimpleNamespace(arguments_schema={'type': 'object', 'properties': {},
            'required': [], 'additionalProperties': False})

    def invoke(self, task, tool_id, args, *, idempotency_key):
        self.calls.append((task.task_id, tool_id, args, idempotency_key))
        return ToolResult(status=ToolStatus.SUCCESS, data={'ok': True, 'content': 'read result'})


class DevelopmentAgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.task = Task(goal='bounded loop test', workspace=self.tmp.name)
        self.model = model()
        self.broker = FakeBroker()

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_tool_then_final_and_usage_accumulation(self):
        call = ToolCall(call_id='c1', name='READ_FILE', arguments={'path': 'a.txt'})
        adapter = FakeAdapter([response(calls=(call,), input_tokens=3, output_tokens=4),
                               response('done', input_tokens=5, output_tokens=6)])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',))
        result = loop.run('inspect one file', Event())
        self.assertEqual((result.status, result.summary), ('DONE', 'done'))
        self.assertEqual((result.model_calls, result.tool_calls), (2, 1))
        self.assertEqual((result.input_tokens, result.output_tokens), (8, 10))
        self.assertEqual(len(self.broker.calls), 1)
        self.assertEqual(adapter.requests[1].messages[-1].role, 'tool')

    def test_unregistered_tool_denied_without_broker_call(self):
        call = ToolCall(call_id='c1', name='RUN_TESTS', arguments={})
        adapter = FakeAdapter([response(calls=(call,)), response('denied')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',))
        result = loop.run('try an unknown action', Event())
        self.assertEqual(result.status, 'DONE')
        self.assertEqual(self.broker.calls, [])
        tool_message = json.loads(adapter.requests[1].messages[-1].content)
        self.assertEqual(tool_message['status'], 'policy_denied')

    def test_step_cap_is_explicit(self):
        call = ToolCall(call_id='c1', name='READ_FILE', arguments={'path': 'a.txt'})
        adapter = FakeAdapter([response(calls=(call,))])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',), max_steps=1)
        self.assertEqual(loop.run('bounded', Event()).status, 'LIMIT_REACHED')
        self.assertEqual(len(adapter.requests), 1)

    def test_deadline_cancels_inflight_request(self):
        adapter = FakeAdapter([response('late')], delay=0.03)
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',), max_wall_s=0.01)
        result = loop.run('deadline', Event())
        self.assertEqual(result.status, 'LIMIT_REACHED')

    def test_tool_message_remains_valid_json_and_bounded(self):
        outcome = ToolResult(status=ToolStatus.SUCCESS, data={'content': 'x' * 5000})
        message = DevelopmentAgentLoop._tool_message(outcome)
        self.assertLessEqual(len(message), 1000)
        self.assertTrue(json.loads(message)['truncated'])

    def test_unknown_usage_stays_unknown(self):
        adapter = FakeAdapter([response('done', input_tokens=None, output_tokens=None)])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',))
        result = loop.run('usage unavailable', Event())
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)


if __name__ == '__main__':
    unittest.main()
