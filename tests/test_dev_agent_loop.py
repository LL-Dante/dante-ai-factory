import json
import hashlib
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
from dante.inference import InferenceCancelled, AdapterUnavailable


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


class FailingAdapter(FakeAdapter):
    def complete_cancellable(self, request, cancellation):
        raise AdapterUnavailable('safe failure', diagnostic={'endpoint': '127.0.0.1:11434',
            'http_status': 500, 'request_ended_at': '2026-09-27T20:23:07Z', 'duration_s': 20.094,
            'safe_error_body': {'safe_markers': ['unexpected end of json input']}})


class FakeBroker:
    def __init__(self):
        self.calls = []
        self.run_test_outcomes = []

    def manifest(self, _tool_id):
        return SimpleNamespace(arguments_schema={'type': 'object', 'properties': {},
            'required': [], 'additionalProperties': False})

    def invoke(self, task, tool_id, args, *, idempotency_key):
        self.calls.append((task.task_id, tool_id, args, idempotency_key))
        if tool_id == 'RUN_TESTS':
            passed = self.run_test_outcomes.pop(0) if self.run_test_outcomes else True
            return ToolResult(status=ToolStatus.SUCCESS,
                              data={'ok': True, 'tests_passed': passed, 'output': 'private test output'})
        if tool_id == 'LIST_FILES':
            return ToolResult(status=ToolStatus.SUCCESS,
                              data={'ok': True, 'path': args.get('path'), 'entries': ['a.txt']})
        if tool_id == 'WRITE_FILE':
            return ToolResult(status=ToolStatus.SUCCESS,
                              data={'ok': True, 'path': args.get('path'),
                                    'bytes_written': len(args.get('content', '').encode('utf-8'))})
        return ToolResult(status=ToolStatus.SUCCESS, data={'ok': True, 'content': 'read result'})


class DevelopmentAgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.task = Task(goal='bounded loop test', workspace=self.tmp.name)
        self.model = model()
        self.broker = FakeBroker()
        self.events = []

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

    def test_model_failure_emits_safe_durable_event_and_stops(self):
        events = []
        loop = DevelopmentAgentLoop(FailingAdapter([]), self.broker, self.task, self.model,
            ('READ_FILE',), emit=lambda name, **data: events.append((name, data)))
        result = loop.run('bounded task', Event())
        self.assertEqual(result.status, 'FAILED')
        name, event = events[0]
        self.assertEqual(name, 'model.failed')
        self.assertEqual(event['call_sequence'], 1)
        self.assertEqual(event['http_status'], 500)
        self.assertNotIn('prompt', json.dumps(event))
        self.assertNotIn('response', json.dumps(event))

    def test_coding_budget_reserves_baseline_edit_and_verification(self):
        reads = [ToolCall(call_id=f'r{i}', name='READ_FILE', arguments={'path': f'f{i}'}) for i in range(1, 5)]
        baseline = ToolCall(call_id='base', name='RUN_TESTS', arguments={})
        edit = ToolCall(call_id='edit', name='PATCH_FILE', arguments={'path':'f1','expected':'x','replacement':'y'})
        verify = ToolCall(call_id='verify', name='RUN_TESTS', arguments={})
        replies = [*(response(calls=(call,)) for call in reads), response(calls=(baseline,)),
                   response(calls=(edit,)), response(calls=(verify,)), response('complete')]
        adapter = FakeAdapter(replies)
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model,
            ('READ_FILE', 'RUN_TESTS', 'PATCH_FILE'))
        result = loop.run('inspect edit verify', Event())
        self.assertEqual(result.status, 'DONE')
        self.assertLessEqual(result.tool_calls, 8)
        self.assertIn('RUN_TESTS', [name for _, name, _, _ in self.broker.calls])

    def test_step_cap_is_explicit(self):
        call = ToolCall(call_id='c1', name='READ_FILE', arguments={'path': 'a.txt'})
        adapter = FakeAdapter([response(calls=(call,))])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',), max_steps=1)
        self.assertEqual(loop.run('bounded', Event()).status, 'LIMIT_REACHED')
        self.assertEqual(len(adapter.requests), 1)

    def test_default_budget_allows_final_after_six_tool_steps(self):
        replies = []
        for index in range(6):
            call = ToolCall(call_id=f'c{index}', name='READ_FILE', arguments={'path': f'file_{index}.txt'})
            replies.append(response(calls=(call,)))
        replies.append(response('all checks passed'))
        adapter = FakeAdapter(replies)
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',))
        result = loop.run('inspect, execute and finalize', Event())
        self.assertEqual((result.status, result.summary), ('DONE', 'all checks passed'))
        self.assertEqual((result.model_calls, result.tool_calls), (7, 6))

    def test_run_tests_event_preserves_boolean_test_outcome(self):
        call = ToolCall(call_id='test', name='RUN_TESTS', arguments={})
        adapter = FakeAdapter([response(calls=(call,)), response('test failed as expected')])
        self.broker.run_test_outcomes = [False]
        events = []
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model,
                                    ('RUN_TESTS',), emit=lambda name, **data: events.append((name, data)))
        result = loop.run('run the test', Event())
        self.assertEqual(result.status, 'DONE')
        test_event = next(data for name, data in events if name == 'tool.completed')
        self.assertIs(test_event['tests_passed'], False)

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

    def test_range_tool_message_preserves_numbered_content_and_valid_json(self):
        outcome = ToolResult(status=ToolStatus.SUCCESS, data={'ok': True, 'path': 'a.py',
            'start_line': 8, 'end_line': 8, 'total_lines': 20, 'returned_lines': 1,
            'truncated': False, 'result_sha256': 'a' * 64,
            'lines': [{'line': 8, 'text': 'x' * 2000}]})
        message = DevelopmentAgentLoop._tool_message(outcome)
        parsed = json.loads(message)
        self.assertLessEqual(len(message), 1000)
        self.assertEqual(parsed['data']['start_line'], 8)
        self.assertEqual(parsed['data']['lines'][0]['line'], 8)
        self.assertEqual(len(parsed['data']['lines'][0]['text']), 400)
        self.assertTrue(parsed['data']['truncated'])

    def test_test_failure_feedback_names_failing_cases_without_raw_output(self):
        outcome = ToolResult(status=ToolStatus.SUCCESS, data={
            'ok': True, 'target': 'tests/test_report_cache.py', 'tests_passed': False,
            'return_code': 1, 'timed_out': False, 'output_truncated': False,
            'output': ('FAIL: test_suffix_is_case_insensitive (ReportCacheTests)\n'
                       'AssertionError: expected-private-value\n'
                       'Ran 4 tests in 0.2s\nFAILED (failures=1)')})
        message = DevelopmentAgentLoop._tool_message(outcome)
        self.assertLessEqual(len(message), 1000)
        data = json.loads(message)['data']
        self.assertEqual(data['failed_tests'], ['test_suffix_is_case_insensitive'])
        self.assertEqual(data['test_summary'], 'Ran 4 tests in 0.2s')
        self.assertNotIn('expected-private-value', message)

    def test_system_prompt_tells_worker_to_stop_exploring_after_baseline_failure(self):
        adapter = FakeAdapter([response('done')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',))
        loop.run('bounded task', Event())
        prompt = adapter.requests[0].messages[0].content
        self.assertIn('do not search or reread unchanged files', prompt)
        self.assertIn('post-edit test run', prompt)

    def test_tool_definitions_explain_targeted_read_and_patch(self):
        adapter = FakeAdapter([response('done')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model,
            ('READ_FILE', 'READ_FILE_RANGE', 'PATCH_FILE'))
        loop.run('inspect', Event())
        definitions = {item.name: item.description for item in adapter.requests[0].tools}
        self.assertIn('READ_FILE_RANGE', definitions['READ_FILE'])
        self.assertIn('numbered lines', definitions['READ_FILE_RANGE'])
        self.assertIn('exact expected text', definitions['PATCH_FILE'])

    def test_unknown_usage_stays_unknown(self):
        adapter = FakeAdapter([response('done', input_tokens=None, output_tokens=None)])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',))
        result = loop.run('usage unavailable', Event())
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)

    def test_duplicate_list_files_in_one_response_is_guarded_and_ids_map(self):
        first = ToolCall(call_id='list-1', name='LIST_FILES', arguments={'path': '.'})
        duplicate = ToolCall(call_id='list-2', name='LIST_FILES', arguments={'path': '.'})
        adapter = FakeAdapter([response(calls=(first, duplicate)), response('done')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('LIST_FILES',))
        result = loop.run('list once', Event())
        self.assertEqual(result.status, 'DONE')
        self.assertEqual(len(self.broker.calls), 1)
        messages = [message for message in adapter.requests[1].messages if message.role == 'tool']
        self.assertEqual([message.call_id for message in messages], ['list-1', 'list-2'])
        duplicate_payload = json.loads(messages[-1].content)
        self.assertEqual(duplicate_payload['guard'], 'DUPLICATE_TOOL_REQUEST')
        self.assertEqual(duplicate_payload['message'], 'PREVIOUS RESULT STILL VALID')
        self.assertEqual(duplicate_payload['previous_call_id'], 'list-1')

    def test_duplicate_read_file_across_turns_reuses_result(self):
        first = ToolCall(call_id='read-1', name='READ_FILE', arguments={'path': 'a.txt'})
        duplicate = ToolCall(call_id='read-2', name='READ_FILE', arguments={'path': 'a.txt'})
        adapter = FakeAdapter([response(calls=(first,)), response(calls=(duplicate,)), response('done')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('READ_FILE',))
        result = loop.run('read once', Event())
        self.assertEqual(result.status, 'DONE')
        self.assertEqual(len(self.broker.calls), 1)
        message = adapter.requests[2].messages[-1]
        self.assertEqual(message.call_id, 'read-2')
        self.assertEqual(json.loads(message.content)['previous_call_id'], 'read-1')

    def test_read_cache_invalidated_after_write_file(self):
        read1 = ToolCall(call_id='r1', name='READ_FILE', arguments={'path': 'a.txt'})
        write = ToolCall(call_id='w1', name='WRITE_FILE', arguments={'path': 'a.txt', 'content': 'new'})
        read2 = ToolCall(call_id='r2', name='READ_FILE', arguments={'path': 'a.txt'})
        adapter = FakeAdapter([response(calls=(read1,)), response(calls=(write,)),
                               response(calls=(read2,)), response('done')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model,
                                    ('READ_FILE', 'WRITE_FILE'))
        result = loop.run('edit then reread', Event())
        self.assertEqual(result.status, 'DONE')
        self.assertEqual([call[1] for call in self.broker.calls], ['READ_FILE', 'WRITE_FILE', 'READ_FILE'])

    def test_read_cache_invalidated_after_run_tests(self):
        read1 = ToolCall(call_id='r1', name='READ_FILE', arguments={'path': 'a.txt'})
        run = ToolCall(call_id='t1', name='RUN_TESTS', arguments={'target': 'tests/test_a.py'})
        read2 = ToolCall(call_id='r2', name='READ_FILE', arguments={'path': 'a.txt'})
        adapter = FakeAdapter([response(calls=(read1,)), response(calls=(run,)),
                               response(calls=(read2,)), response('done')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model,
                                    ('READ_FILE', 'RUN_TESTS'))
        result = loop.run('test then reread', Event())
        self.assertEqual(result.status, 'DONE')
        self.assertEqual([call[1] for call in self.broker.calls], ['READ_FILE', 'RUN_TESTS', 'READ_FILE'])

    def test_repeated_write_file_is_never_cached(self):
        write1 = ToolCall(call_id='w1', name='WRITE_FILE', arguments={'path': 'a.txt', 'content': 'same'})
        write2 = ToolCall(call_id='w2', name='WRITE_FILE', arguments={'path': 'a.txt', 'content': 'same'})
        adapter = FakeAdapter([response(calls=(write1,)), response(calls=(write2,)), response('done')])
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('WRITE_FILE',))
        result = loop.run('write twice', Event())
        self.assertEqual(result.status, 'DONE')
        self.assertEqual(len(self.broker.calls), 2)

    def test_third_identical_read_request_returns_agent_stuck(self):
        calls = [ToolCall(call_id=f'l{i}', name='LIST_FILES', arguments={'path': '.'})
                 for i in range(1, 4)]
        adapter = FakeAdapter([*(response(calls=(call,)) for call in calls)])
        events = []
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('LIST_FILES',),
                                    emit=lambda name, **data: events.append((name, data)))
        result = loop.run('list repeatedly', Event())
        self.assertEqual(result.status, 'AGENT_STUCK')
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(len(self.broker.calls), 1)
        tool_events = [data for name, data in events if name == 'tool.completed']
        self.assertEqual([event['call_id'] for event in tool_events], ['l1', 'l2', 'l3'])

    def test_normal_inspect_fail_edit_pass_final_flow_and_call_ids(self):
        self.broker.run_test_outcomes = [False, True]
        calls = [
            ToolCall(call_id='c1', name='LIST_FILES', arguments={'path': '.'}),
            ToolCall(call_id='c2', name='READ_FILE', arguments={'path': 'calculator.py'}),
            ToolCall(call_id='c3', name='RUN_TESTS', arguments={'target': 'tests/test_calculator.py'}),
            ToolCall(call_id='c4', name='WRITE_FILE', arguments={'path': 'calculator.py', 'content': 'fixed'}),
            ToolCall(call_id='c5', name='RUN_TESTS', arguments={'target': 'tests/test_calculator.py'}),
        ]
        adapter = FakeAdapter([*(response(calls=(call,)) for call in calls), response('tests pass')])
        events = []
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model,
            ('LIST_FILES', 'READ_FILE', 'RUN_TESTS', 'WRITE_FILE'),
            emit=lambda name, **data: events.append((name, data)))
        result = loop.run('inspect, test, edit, retest', Event())
        self.assertEqual((result.status, result.summary), ('DONE', 'tests pass'))
        self.assertEqual([call[1] for call in self.broker.calls],
                         ['LIST_FILES', 'READ_FILE', 'RUN_TESTS', 'WRITE_FILE', 'RUN_TESTS'])
        tool_events = [data for name, data in events if name == 'tool.completed']
        self.assertEqual([event['call_id'] for event in tool_events], ['c1', 'c2', 'c3', 'c4', 'c5'])
        self.assertEqual([event['tests_passed'] for event in tool_events if 'tests_passed' in event],
                         [False, True])
        self.assertNotIn('private test output', json.dumps(tool_events))
        self.assertNotIn('read result', json.dumps(tool_events))

    def test_tool_trace_redacts_write_content_and_summarizes_results(self):
        secret = 'do-not-log-this-source'
        write = ToolCall(call_id='write-secret', name='WRITE_FILE',
                         arguments={'path': 'a.txt', 'content': secret})
        adapter = FakeAdapter([response(calls=(write,)), response('done')])
        events = []
        loop = DevelopmentAgentLoop(adapter, self.broker, self.task, self.model, ('WRITE_FILE',),
                                    emit=lambda name, **data: events.append((name, data)))
        self.assertEqual(loop.run('write', Event()).status, 'DONE')
        event = next(data for name, data in events if name == 'tool.completed')
        self.assertEqual(event['call_id'], 'write-secret')
        normalized = json.loads(event['normalized_arguments'])
        self.assertNotIn('content', normalized)
        self.assertEqual(normalized['content_sha256'], hashlib.sha256(secret.encode()).hexdigest())
        self.assertEqual(normalized['content_bytes'], len(secret.encode()))
        self.assertNotIn(secret, json.dumps(event))
        self.assertTrue(event['result_digest'])
        model_event = next(data for name, data in events if name == 'model.completed')
        self.assertEqual(model_event['tool_requested'], ['WRITE_FILE'])
        summary = json.loads(model_event['tool_arguments_summary'])
        self.assertNotIn(secret, json.dumps(summary))
        self.assertEqual(summary[0]['arguments']['content_bytes'], len(secret.encode()))
        self.assertIsNone(model_event['context_remaining_tokens'])
        self.assertIn('tool_wall_s', event)


if __name__ == '__main__':
    unittest.main()
