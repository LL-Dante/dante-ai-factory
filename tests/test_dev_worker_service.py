import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from threading import Event
from types import SimpleNamespace
import httpx

from dante.contracts import CostClass, ModelRef, PrivacyClass, Task, TaskStatus
from dante.contracts.agents import AgentTaskPayload
from dante.contracts.inference import AssistantMessage, ToolCall, Usage
from dante.contracts.runtime import LocalModelMetadata
from dante.dev_qwen_runtime import DevelopmentQwenAdapter
from dante.dev_worker_service import (DevelopmentAgentStuck, DevelopmentWorkerService,
                                      _CodingAgent, discover_development_qwen)
from dante.dev_worker_tools import register_coding_tools
from dante.ledger import TaskLedger
from dante.tool_broker import ToolBroker


def dev_model():
    return ModelRef(model_id='qwen-dev', provider_id='dev-qwen', logical_alias='qwen', version='dev',
        runtime='ollama', local=True, cost_class=CostClass.LOCAL_COMPUTE,
        privacy_eligibility={PrivacyClass.INTERNAL},
        local_metadata=LocalModelMetadata(runtime_reference='dante-qwen-agent:latest',
            runtime_digest='a' * 64, context_tokens=4096))


class DevelopmentWorkerServiceTests(unittest.TestCase):
    def test_coding_agent_completes_with_real_ledger_transition_api(self):
        class FakeAdapter:
            def __init__(self, replies):
                self.replies = iter(replies)

            def complete_cancellable(self, _request, _cancellation):
                return next(self.replies)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir()
            (workspace / 'calculator.py').write_text('def add(a, b): return a + b\n', encoding='utf-8')
            model = dev_model()
            ledger = TaskLedger(root / 'dante.db')
            broker = ToolBroker(ledger=ledger)
            tools = register_coding_tools(broker, str(workspace))
            call = ToolCall(call_id='read-1', name='READ_FILE', arguments={'path': 'calculator.py'})
            replies = [
                SimpleNamespace(assistant_message=AssistantMessage(tool_calls=(call,)), content='',
                    tool_calls=(call,), finish_reason='tool_calls', usage=Usage(input_tokens=10, output_tokens=5)),
                SimpleNamespace(assistant_message=AssistantMessage(content='File inspected; tests not run.'),
                    content='File inspected; tests not run.', tool_calls=(), finish_reason='stop',
                    usage=Usage(input_tokens=12, output_tokens=8)),
            ]
            adapter = FakeAdapter(replies)
            agent = _CodingAgent(adapter=adapter, broker=broker, ledger=ledger, model=model,
                                 workspace=workspace.resolve(), tool_ids=tools)
            task = Task(task_id='job_' + 'a' * 32, goal='Read calculator.py', workspace=str(workspace.resolve()))
            payload = AgentTaskPayload(agent_id='local-coding-worker', objective='Read calculator.py',
                requested_output_tokens=384, definition_version='1.0.0', definition_digest='a' * 64)
            spec = SimpleNamespace(metadata={'workspace': str(workspace.resolve())})
            emitted = []
            result = agent.execute(None, SimpleNamespace(job_id=task.task_id), None, payload, spec,
                                   Event(), lambda name, **meta: emitted.append((name, meta)))
            self.assertEqual(result.text, 'File inspected; tests not run.')
            self.assertEqual(result.metadata['model_calls'], 2)
            self.assertEqual(result.metadata['tool_calls'], 1)
            self.assertEqual(ledger.get_task(task.task_id).status, TaskStatus.COMPLETED)
            self.assertIn('tool.completed', [name for name, _meta in emitted])

    def test_coding_agent_exposes_agent_stuck_failure_code(self):
        class FakeAdapter:
            def __init__(self, replies):
                self.replies = iter(replies)

            def complete_cancellable(self, _request, _cancellation):
                return next(self.replies)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'workspace'
            workspace.mkdir()
            (workspace / 'a.txt').write_text('alpha', encoding='utf-8')
            model = dev_model()
            ledger = TaskLedger(root / 'dante.db')
            broker = ToolBroker(ledger=ledger)
            tools = register_coding_tools(broker, str(workspace))
            replies = []
            for index in range(3):
                call = ToolCall(call_id=f'read-{index}', name='READ_FILE',
                                arguments={'path': 'a.txt'})
                replies.append(SimpleNamespace(assistant_message=AssistantMessage(tool_calls=(call,)),
                    content='', tool_calls=(call,), finish_reason='tool_calls',
                    usage=Usage(input_tokens=1, output_tokens=1)))
            agent = _CodingAgent(adapter=FakeAdapter(replies), broker=broker, ledger=ledger,
                model=model, workspace=workspace.resolve(), tool_ids=tools)
            task = Task(task_id='job_' + 'b' * 32, goal='read repeatedly', workspace=str(workspace.resolve()))
            payload = AgentTaskPayload(agent_id='local-coding-worker', objective='read repeatedly',
                requested_output_tokens=384, definition_version='1.0.0', definition_digest='b' * 64)
            spec = SimpleNamespace(metadata={'workspace': str(workspace.resolve())})
            with self.assertRaises(DevelopmentAgentStuck) as caught:
                agent.execute(None, SimpleNamespace(job_id=task.task_id), None, payload, spec,
                              Event(), lambda *_args, **_kwargs: None)
            self.assertEqual(caught.exception.failure_code, 'AGENT_STUCK')
            self.assertEqual(ledger.get_task(task.task_id).status, TaskStatus.FAILED_TERMINAL)

    def test_readonly_model_discovery_marks_qualification_unknown(self):
        digest = 'a' * 64

        def transport(request):
            if request.url.path == '/api/tags':
                return httpx.Response(200, json={'models': [{
                    'name': 'dante-qwen-agent:latest', 'digest': digest,
                    'details': {'format': 'gguf', 'quantization_level': 'Q3_K_M'},
                }]})
            if request.url.path == '/api/show':
                return httpx.Response(200, json={'details': {'format': 'gguf',
                    'quantization_level': 'Q3_K_M'}, 'model_info': {'general.architecture': 'qwen35'},
                    'capabilities': ['tools']})
            return httpx.Response(404)

        adapter = DevelopmentQwenAdapter(transport=httpx.MockTransport(transport))
        model = discover_development_qwen(adapter)
        self.assertEqual(model.local_metadata.runtime_digest, digest)
        self.assertEqual(model.local_metadata.context_tokens, 8192)
        self.assertEqual(model.local_metadata.qualification, 'unverified')
        self.assertEqual(model.local_metadata.license_status, 'unknown')

    def test_reuses_one_store_and_registers_only_workspace_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'sandbox'
            workspace.mkdir()
            service = DevelopmentWorkerService(root / 'data' / 'dante.db',
                DevelopmentQwenAdapter(), dev_model(), workspace)
            try:
                self.assertEqual(service.store.path, service.ledger.path)
                self.assertEqual(service.tool_ids,
                    ('READ_FILE', 'LIST_FILES', 'SEARCH_TEXT', 'WRITE_FILE', 'RUN_TESTS'))
                self.assertEqual(service.orchestrator.resource_gate.max_gpu_jobs, 1)
                self.assertEqual(service.orchestrator.state, 'STOPPED')
            finally:
                service.close()

    def test_refuses_b2_2_database_path_before_opening_it(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaises(ValueError):
                DevelopmentWorkerService(Path('C:/DanteAI/data/blocked.db'),
                    DevelopmentQwenAdapter(), dev_model(), workspace)

    def test_refuses_workspace_with_linked_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / 'sandbox-parent'
            workspace = parent / 'sandbox'
            workspace.mkdir(parents=True)
            with patch('dante.dev_worker_service._is_link', side_effect=lambda path: Path(path) == parent):
                with self.assertRaisesRegex(ValueError, 'cannot be a link or junction'):
                    DevelopmentWorkerService(Path(directory) / 'dante.db',
                        DevelopmentQwenAdapter(), dev_model(), workspace)

    def test_submit_uses_existing_workload_queue_and_supports_queued_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / 'sandbox'
            workspace.mkdir()
            service = DevelopmentWorkerService(root / 'dante.db',
                DevelopmentQwenAdapter(), dev_model(), workspace)
            service.start = lambda: None  # Keep this queue unit test off the model.
            try:
                job = service.submit('Create a small text file')
                self.assertEqual(job.state.value, 'queued')
                self.assertEqual(service.queue_position(str(job.job_id)), 1)
                self.assertEqual(service.events(str(job.job_id))[0]['event'], 'submitted')
                cancelled = service.cancel(str(job.job_id))
                self.assertEqual(cancelled.state.value, 'cancelled')
            finally:
                service.close()


if __name__ == '__main__':
    unittest.main()
