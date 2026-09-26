from __future__ import annotations

import contextlib
import ctypes
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from dante import node0_cli, node0_control
from dante.contracts.continuity import ExecutionPolicy
from dante.inference import InferenceTimeout, QualificationDenied
from dante.continuity import ContinuityOutcome, ContinuitySignal, Disposition
from dante.node0_control import (ControlError, MAX_INFER_OUTPUT_TOKENS,
    MAX_INFER_PROMPT_CHARS, MAX_REQUEST_BYTES, Node0ControlService, decode_payload,
    encode_frame)


class ControlPlaneTests(unittest.TestCase):
    def setUp(self):
        metadata = SimpleNamespace(runtime_reference='qwen3:4b')
        self.model = SimpleNamespace(model_id='qwen3-4b-node0', local_metadata=metadata,
            runtime='ollama', provider_id='ollama')
        response = SimpleNamespace(content='ready', fallback=False, provider_id='ollama',
            model=self.model)
        response.finish_reason = 'stop'
        response.usage = SimpleNamespace(output_tokens=3)
        self.supervisor = Mock()
        self.supervisor.config.qualification.model_reference = 'qwen3:4b'
        self.supervisor.operator_status.return_value = {
            'state': 'QUALIFIED', 'gate_accepted': True, 'ready_for_local_routing': True}
        self.supervisor.operator_health.return_value = {
            'healthy': True, 'gpu_execution_verified': True,
            'qualification_id': 'qualification-fixture', 'gpu_uuid': 'GPU-fixture',
            'gpu_vram_bytes': 1, 'runtime_version': '0.34.4'}
        self.supervisor.infer.return_value = response
        self.supervisor.node.continuity.policy.mode = ExecutionPolicy.LOCAL_ONLY
        self.supervisor.start = Mock()
        self.supervisor.runtime.start = Mock()
        self.service = Node0ControlService(self.supervisor)

    def request(self, **changes):
        request = {'op': 'infer', 'model': 'qwen3:4b', 'prompt': 'Say ready.',
                   'max_output_tokens': 64, 'thinking': 'off'}
        request.update(changes)
        return request

    def test_status_and_health_query_the_existing_supervisor(self):
        self.supervisor.operator_status.return_value = {'state': 'QUALIFIED'}
        self.supervisor.operator_health.return_value = {'healthy': True}
        self.assertEqual(self.service.dispatch({'op': 'status'})['result'], {'state': 'QUALIFIED'})
        self.assertEqual(self.service.dispatch({'op': 'health'})['result'], {'healthy': True})
        self.assertIs(self.service.supervisor, self.supervisor)

    def test_inference_uses_existing_supervisor_and_reports_local_route(self):
        result = self.service.dispatch(self.request())['result']
        self.supervisor.infer.assert_called_once_with('Say ready.', model_reference='qwen3:4b',
            max_output_tokens=64, thinking='off')
        self.assertEqual(result['provider'], 'ollama')
        self.assertFalse(result['fallback'])
        self.assertEqual(result['response'], 'ready')
        self.assertEqual(result['thinking'], 'off')
        self.assertEqual(result['max_output_tokens'], 64)
        self.assertTrue(result['done'])
        self.assertEqual(result['done_reason'], 'stop')

    def test_control_inference_defaults_to_bounded_budget_and_thinking_off(self):
        request = {'op': 'infer', 'model': 'qwen3:4b', 'prompt': 'Say ready.'}
        result = self.service.dispatch(request)['result']
        self.supervisor.infer.assert_called_once_with('Say ready.', model_reference='qwen3:4b',
            max_output_tokens=512, thinking='off')
        self.assertEqual(result['thinking'], 'off')
        self.assertEqual(result['max_output_tokens'], 512)

    def test_inference_rejects_unconfigured_model_and_oversized_inputs(self):
        invalid = [self.request(model='other:latest'), self.request(prompt='x' * (MAX_INFER_PROMPT_CHARS + 1)),
                   self.request(max_output_tokens=MAX_INFER_OUTPUT_TOKENS + 1),
                   self.request(max_output_tokens=True), self.request(extra='ignored')]
        for request in invalid:
            with self.subTest(request=list(request)):
                with self.assertRaises(ControlError):
                    self.service.dispatch(request)
        self.supervisor.infer.assert_not_called()

    def test_qualification_and_runtime_fail_closed(self):
        self.supervisor.operator_status.return_value = {'ready_for_local_routing': False}
        with self.assertRaisesRegex(ControlError, 'qualification_or_runtime_not_ready'):
            self.service.dispatch(self.request())
        self.supervisor.operator_status.return_value = {'ready_for_local_routing': True}
        self.supervisor.operator_health.return_value = {'healthy': False, 'gpu_execution_verified': False}
        with self.assertRaisesRegex(ControlError, 'gpu_execution_not_verified'):
            self.service.dispatch(self.request())

    def test_local_only_policy_is_required_and_never_falls_back(self):
        self.supervisor.node.continuity.policy.mode = ExecutionPolicy.LOCAL_PREFERRED
        with self.assertRaisesRegex(ControlError, 'local_only_policy_required'):
            self.service.dispatch(self.request())
        self.supervisor.node.continuity.policy.mode = ExecutionPolicy.LOCAL_ONLY
        self.supervisor.infer.side_effect = QualificationDenied('denied')
        with self.assertRaisesRegex(ControlError, 'qualification_denied'):
            self.service.dispatch(self.request())

    def test_timeout_is_returned_as_typed_bounded_failure(self):
        self.supervisor.infer.side_effect = InferenceTimeout('runtime timed out')
        with self.assertRaisesRegex(ControlError, 'inference_timeout'):
            self.service.dispatch(self.request())

    def test_invalid_response_diagnostic_survives_control_error_safely(self):
        diagnostic = {
            'http_status': 200, 'http_content_type': 'application/json',
            'response_bytes': 128, 'json_decoded': True,
            'top_level_type': 'object', 'top_level_keys': ['done', 'message', 'model'],
            'done_present': False, 'done_type': None, 'done_value': None,
            'model_present': True, 'model_type': 'string', 'model_value': 'qwen3:4b',
            'message_present': True, 'message_type': 'object', 'message_keys': ['content', 'role'],
            'role_present': True, 'role_type': 'string', 'role_value': 'assistant',
            'content_present': True, 'content_type': 'string', 'content_length': 20,
            'thinking_present': False, 'thinking_type': None, 'thinking_length': None,
            'tool_calls_present': False, 'tool_calls_type': None, 'tool_calls_count': None,
            'generation_completed': False, 'done_reason': None,
            'failed_validation_rule': 'generation_not_complete', 'failed_field': 'done',
            'expected': 'true', 'actual': 'null',
        }
        self.supervisor.infer.side_effect = ContinuitySignal(ContinuityOutcome(
            Disposition.RETRY_LATER, 'routes_temporarily_unavailable', diagnostic=diagnostic))
        with self.assertRaisesRegex(ControlError, 'routes_temporarily_unavailable') as raised:
            self.service.dispatch(self.request())
        self.assertEqual(raised.exception.diagnostic, diagnostic)
        self.assertNotIn('generated text', str(raised.exception.diagnostic))

    def test_malformed_and_oversized_frames_are_rejected(self):
        with self.assertRaises(ControlError):
            decode_payload(b'{"op":"status","op":"infer"}')
        with self.assertRaises(ControlError):
            decode_payload(b'[]')
        with self.assertRaises(ControlError):
            encode_frame({'body': 'x' * MAX_REQUEST_BYTES})

    def test_arbitrary_commands_and_unknown_operations_are_unavailable(self):
        for operation in ('exec', 'python', 'shell', 'filesystem', 'launch_runtime'):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ControlError, 'operation_not_supported'):
                    self.service.dispatch({'op': operation, 'command': 'whoami'})
        for key in ('command', 'script', 'path'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ControlError, 'invalid_request_fields'):
                    self.service.dispatch({'op': 'status', key: 'untrusted'})

    def test_pipe_read_loop_enforces_expired_deadline(self):
        class Kernel:
            @staticmethod
            def ReadFile(*_args):
                ctypes.set_last_error(232)  # ERROR_NO_DATA: nonblocking pipe has no bytes yet
                return 0

        with patch.object(node0_control, '_win_api', return_value=(Kernel(), None)), \
             patch.object(node0_control.time, 'monotonic', side_effect=(1.0,)):
            with self.assertRaisesRegex(TimeoutError, 'pipe read timed out'):
                node0_control._read_exact(1, 1, deadline=0.5)

    def test_pipe_configuration_rejects_remote_clients(self):
        invalid_handle = node0_control.wintypes.HANDLE(-1).value
        kernel = Mock()
        kernel.CreateNamedPipeW.return_value = invalid_handle
        server = node0_control.Node0ControlServer(self.service)
        attributes = node0_control._SecurityAttributes(ctypes.sizeof(node0_control._SecurityAttributes), None, False)
        with patch.object(node0_control, '_win_api', return_value=(kernel, Mock())), \
             patch.object(node0_control, '_pipe_security_attributes',
                           return_value=(kernel, None, attributes)):
            server._accept()
        pipe_mode = kernel.CreateNamedPipeW.call_args.args[2]
        self.assertTrue(pipe_mode & 0x00000008)  # PIPE_REJECT_REMOTE_CLIENTS

    def test_pipe_acl_allows_current_user_and_excludes_other_users(self):
        authorized_sid = 'S-1-5-21-100-200-300-1001'
        other_sid = 'S-1-5-21-100-200-300-1002'
        with patch.object(node0_control, '_current_user_sid', return_value=authorized_sid):
            sddl = node0_control.current_user_pipe_sddl()
        self.assertIn(f'(A;;GA;;;{authorized_sid})', sddl)
        self.assertNotIn(other_sid, sddl)
        self.assertIn('(A;;GA;;;SY)', sddl)
        self.assertIn('(A;;GA;;;BA)', sddl)

    def test_control_operations_reuse_supervisor_without_starting_runtime(self):
        with patch('dante.node0_supervisor.Node0Supervisor') as supervisor_type:
            self.service.dispatch({'op': 'status'})
            self.service.dispatch(self.request())
        supervisor_type.assert_not_called()
        self.supervisor.start.assert_not_called()
        self.supervisor.runtime.start.assert_not_called()


class CliTests(unittest.TestCase):
    def run_cli(self, args):
        client = Mock()
        client.request.return_value = {'accepted': True}
        output = io.StringIO()
        with patch.object(node0_cli, 'Node0ControlClient', return_value=client), \
             patch.object(node0_cli, '_print', side_effect=lambda value: output.write(str(value))):
            exit_code = node0_cli.main(args)
        return exit_code, client, output.getvalue()

    def test_status_health_and_infer_commands_dispatch_only_control_requests(self):
        cases = [(['status'], {'op': 'status'}),
                 (['health'], {'op': 'health'}),
                 (['infer', '--model', 'qwen3:4b', '--prompt', 'Say ready.',
                   '--max-output-tokens', '17'],
                  {'op': 'infer', 'model': 'qwen3:4b', 'prompt': 'Say ready.',
                   'max_output_tokens': 17, 'thinking': 'off'}),
                 (['infer', '--model', 'qwen3:4b', '--prompt', 'Say ready.', '--thinking', 'on'],
                  {'op': 'infer', 'model': 'qwen3:4b', 'prompt': 'Say ready.',
                   'max_output_tokens': 512, 'thinking': 'on'})]
        for args, expected in cases:
            with self.subTest(args=args):
                code, client, _ = self.run_cli(args)
                self.assertEqual(code, 0)
                client.request.assert_called_once_with(expected)

    def test_invalid_cli_arguments_fail_before_client_creation(self):
        stderr = io.StringIO()
        with patch.object(node0_cli, 'Node0ControlClient') as client, \
             contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                node0_cli.main(['infer', '--not-a-real-option'])
        self.assertEqual(raised.exception.code, 2)
        client.assert_not_called()
        self.assertIn('usage:', stderr.getvalue())

    def test_cli_does_not_construct_supervisor_or_start_runtime(self):
        client = Mock()
        client.request.return_value = {'state': 'QUALIFIED'}
        with patch.object(node0_cli, 'Node0ControlClient', return_value=client), \
             patch('dante.node0_supervisor.Node0Supervisor') as supervisor_type, \
             patch('dante.local_runtime.OllamaAdapter') as adapter_type:
            self.assertEqual(node0_cli.main(['status']), 0)
        supervisor_type.assert_not_called()
        adapter_type.assert_not_called()
        client.request.assert_called_once_with({'op': 'status'})

    def test_cli_prints_bounded_control_error_diagnostic(self):
        diagnostic = {'failed_validation_rule': 'model_mismatch', 'failed_field': 'model',
                      'expected': 'qwen3:4b', 'actual': 'fixture:other'}
        client = Mock()
        client.request.side_effect = ControlError('routes_temporarily_unavailable', diagnostic=diagnostic)
        output = io.StringIO()
        with patch.object(node0_cli, 'Node0ControlClient', return_value=client), \
             patch.object(node0_cli, '_print', side_effect=lambda value: output.write(str(value))):
            self.assertEqual(node0_cli.main(['infer', '--model', 'qwen3:4b', '--prompt', 'Say ready.']), 2)
        self.assertIn('routes_temporarily_unavailable', output.getvalue())
        self.assertIn('model_mismatch', output.getvalue())


@unittest.skipUnless(os.name == 'nt', 'Windows named-pipe integration test')
class WindowsPipeIntegrationTests(unittest.TestCase):
    def test_real_local_pipe_status_round_trip_and_clean_stop(self):
        payload = {'state': 'QUALIFIED', 'ready_for_local_routing': True}
        supervisor = SimpleNamespace(operator_status=lambda: payload)
        pipe_name = rf'\\.\pipe\DanteNode0.Test.{uuid4().hex}'
        server = node0_control.Node0ControlServer(
            Node0ControlService(supervisor), pipe_name=pipe_name)
        client = node0_control.Node0ControlClient(pipe_name=pipe_name, timeout_s=3.0)
        try:
            server.start()
            self.assertTrue(server.started)
            self.assertTrue(server.ready)
            self.assertIsNone(server.startup_error)
            self.assertEqual(client.request({'op': 'status'}), payload)
        finally:
            server.stop(timeout_s=3.0)
        self.assertFalse(server.started)
        self.assertFalse(server.ready)

    def test_named_pipe_round_trips_bounded_inference_diagnostic(self):
        from dante.continuity import ContinuityOutcome, ContinuitySignal, Disposition
        from dante.node0_control import ControlError
        diagnostic = {'http_status': 200, 'json_decoded': True,
                      'failed_validation_rule': 'generation_not_complete',
                      'failed_field': 'done', 'expected': 'true', 'actual': 'false'}
        supervisor = SimpleNamespace(
            config=SimpleNamespace(qualification=SimpleNamespace(model_reference='qwen3:4b')),
            node=SimpleNamespace(continuity=SimpleNamespace(policy=SimpleNamespace(mode=ExecutionPolicy.LOCAL_ONLY))),
            operator_status=lambda: {'ready_for_local_routing': True},
            infer=Mock(side_effect=ContinuitySignal(ContinuityOutcome(
                Disposition.RETRY_LATER, 'routes_temporarily_unavailable', diagnostic=diagnostic))))
        pipe_name = rf'\\.\pipe\DanteNode0.Diagnostic.{uuid4().hex}'
        server = node0_control.Node0ControlServer(Node0ControlService(supervisor), pipe_name=pipe_name)
        client = node0_control.Node0ControlClient(pipe_name=pipe_name, timeout_s=3.0)
        try:
            server.start()
            with self.assertRaisesRegex(ControlError, 'routes_temporarily_unavailable') as raised:
                client.request({'op': 'infer', 'model': 'qwen3:4b', 'prompt': 'fixture',
                                'max_output_tokens': 8})
            self.assertEqual(raised.exception.diagnostic, diagnostic)
        finally:
            server.stop(timeout_s=3.0)

if __name__ == '__main__':
    unittest.main()
