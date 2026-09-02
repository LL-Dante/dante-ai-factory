import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from dante.agent_host import AgentHostFoundation
from dante.approvals import ApprovalManager
from dante.contracts import PrivacyClass, Task, ToolManifest
from dante.contracts.tools import ToolStatus
from dante.ledger import TaskLedger
from dante.tool_broker import ToolBroker, ToolDenied, workspace_method
from tools.ai_cloud_workspace import Tools


EMPTY = {'type': 'object', 'properties': {}, 'additionalProperties': False}
VALUE = {'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']}


class EnforcementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root / 'ledger.db')
        self.broker = ToolBroker(ledger=self.ledger)
        self.host = AgentHostFoundation(self.ledger, None, None, None, self.broker)
        self.task = self.host.start('P3 local fixture', str(self.root), PrivacyClass.PUBLIC)
        self.calls = []

    def register(self, handler=None, *, required=frozenset(), **changes):
        values = dict(tool_id='fixture', version='1', permissions=required, risk='R0',
                      filesystem_scope='none', arguments_schema=EMPTY)
        values.update(changes)
        manifest = ToolManifest(**values)
        def default(**args):
            self.calls.append(args)
            return {'ok': True}
        self.broker.register(manifest, handler or default, required_permissions=required)
        return manifest

    def invoke(self, args=None, **kwargs):
        return self.broker.invoke(self.task, 'fixture', args or {}, **kwargs)

    def approve(self, args, plan=''):
        manager = ApprovalManager(self.ledger)
        approval, token = manager.request_tool(self.broker, self.task, 'fixture', args, plan_hash=plan)
        manager.decide(approval.approval_id, token, approval.plan_hash, 'fixture-human', approve=True)
        return approval.approval_id

    def assert_denied(self, result, status):
        self.assertEqual(result.status, status)
        self.assertEqual(self.calls, [])
        self.assertFalse(result.effect_uncertain)
        events = [e for e in self.ledger.events(self.task.task_id) if e['event'] == 'tool.outcome']
        self.assertTrue(events)

    def test_valid_R0_read_executes(self):
        (self.root / 'input.txt').write_text('fixture')
        def read(path):
            self.calls.append(path)
            return {'ok': True, 'content': (self.root / path).read_text()}
        self.register(read, required={'read_workspace'}, filesystem_scope=str(self.root),
                      arguments_schema={'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']},
                      path_permissions={'path': 'read_workspace'})
        outcome = self.invoke({'path': 'input.txt'})
        self.assertEqual(outcome.status, ToolStatus.SUCCESS)
        self.assertEqual(outcome.data['content'], 'fixture')
        self.assertEqual(len(self.calls), 1)

    def test_malformed_and_unknown_arguments_have_zero_effect(self):
        self.register(arguments_schema=VALUE)
        for args in ({'value': 5}, {'value': 'a', 'extra': True}, {}, {'value': float('nan')}):
            self.assert_denied(self.invoke(args), ToolStatus.VALIDATION_DENIED)
        self.assertEqual(self.ledger.steps(self.task.task_id), [])

    def test_unknown_tool_denied(self):
        self.assert_denied(self.invoke(), ToolStatus.POLICY_DENIED)

    def test_undeclared_permission_denied(self):
        self.register(required={'write_workspace'}, permissions=set(), risk='R1', filesystem_scope=str(self.root))
        self.assert_denied(self.invoke(), ToolStatus.POLICY_DENIED)

    def test_missing_schema_or_capability_metadata_denied(self):
        self.register(arguments_schema=None)
        self.assert_denied(self.invoke(), ToolStatus.POLICY_DENIED)
        manifest = self.register()
        self.broker.register(manifest, lambda: self.calls.append('bad'))
        self.assert_denied(self.invoke(), ToolStatus.POLICY_DENIED)

    def paths(self):
        self.register(required={'write_workspace'}, risk='R1', filesystem_scope=str(self.root),
                      arguments_schema=VALUE, path_permissions={'value': 'write_workspace'})

    def test_traversal_denied(self):
        self.paths()
        for value in ('../escape', '..\\escape', 'folder/../../escape'):
            self.assert_denied(self.invoke({'value': value}), ToolStatus.POLICY_DENIED)

    def test_absolute_escape_denied(self):
        self.paths()
        self.assert_denied(self.invoke({'value': str(self.root.parent / 'outside')}), ToolStatus.POLICY_DENIED)

    def test_symlink_or_junction_escape_denied(self):
        self.paths()
        link = self.root / 'link'
        # Windows junctions do not require symlink privilege; no shell tool is exposed by DANTE.
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(str(self.root.parent), str(link))
        else:
            link.symlink_to(self.root.parent, target_is_directory=True)
        try:
            self.assert_denied(self.invoke({'value': 'link/escape'}), ToolStatus.POLICY_DENIED)
        finally:
            if os.name == 'nt':
                link.rmdir()
            else:
                link.unlink()

    def test_path_permission_not_declared_denied(self):
        self.register(required={'read_workspace'}, filesystem_scope=str(self.root), arguments_schema=VALUE,
                      path_permissions={'value': 'write_workspace'})
        self.assert_denied(self.invoke({'value': 'x'}), ToolStatus.POLICY_DENIED)

    def test_R0_write_and_process_network_higher_risk_denied(self):
        for changes in ({'required': {'write_workspace'}}, {'required': {'process_execution'}},
                        {'required': {'network_access'}}, {'risk': 'R2'}, {'risk': 'R3'},
                        {'approval_policy': 'unknown'}, {'network_scope': 'internet'}):
            self.register(**changes)
            self.assert_denied(self.invoke(), ToolStatus.POLICY_DENIED)

    def test_absent_approval_and_boolean_bypass_denied(self):
        self.register(approval_policy='always')
        self.assert_denied(self.invoke(), ToolStatus.APPROVAL_REQUIRED)
        with self.assertRaises(ToolDenied) as caught:
            self.broker.execute(self.task, 'fixture', {}, approved=True)
        self.assertEqual(caught.exception.result.status, ToolStatus.APPROVAL_REQUIRED)
        self.assertEqual(self.calls, [])

    def test_one_use_approval_and_verified_reuse(self):
        self.register(approval_policy='always')
        approval = self.approve({})
        self.assertEqual(self.invoke(approval_id=approval).status, ToolStatus.SUCCESS)
        self.assertEqual(self.invoke(approval_id=approval).status, ToolStatus.SUCCESS)
        self.assertEqual(len(self.calls), 1)
        denied = self.invoke(approval_id=approval, idempotency_key='second-effect')
        self.assertEqual(denied.status, ToolStatus.APPROVAL_INVALID)
        self.assertEqual(len(self.calls), 1)

    def test_approval_arguments_binding(self):
        self.register(arguments_schema=VALUE, approval_policy='always')
        approval = self.approve({'value': 'A'})
        self.assert_denied(self.invoke({'value': 'B'}, approval_id=approval), ToolStatus.APPROVAL_INVALID)
        self.assertEqual(self.invoke({'value': 'A'}, approval_id=approval).status, ToolStatus.SUCCESS)

    def test_approval_version_plan_and_task_binding(self):
        self.register(approval_policy='always')
        approval = self.approve({}, 'plan-A')
        self.assert_denied(self.invoke(approval_id=approval, plan_hash='plan-B'), ToolStatus.APPROVAL_INVALID)
        self.register(approval_policy='always', version='2')
        self.assert_denied(self.invoke(approval_id=approval, plan_hash='plan-A'), ToolStatus.APPROVAL_INVALID)
        self.register(approval_policy='always')
        other = self.host.start('other', str(self.root), PrivacyClass.PUBLIC)
        outcome = self.broker.invoke(other, 'fixture', {}, approval_id=approval, plan_hash='plan-A')
        self.assertEqual(outcome.status, ToolStatus.APPROVAL_INVALID)
        self.assertEqual(self.calls, [])

    def test_timeout_enforced_and_never_replayed_or_marked_success(self):
        release, stopped = threading.Event(), threading.Event()
        def slow():
            self.calls.append('started')
            release.wait(5)
            stopped.set()
            return {'ok': True}
        self.register(slow, timeout_s=1)
        try:
            started = time.monotonic()
            result = self.invoke()
            self.assertEqual(result.status, ToolStatus.TIMEOUT)
            self.assertLess(time.monotonic() - started, 3)
            self.assertTrue(result.effect_uncertain)
            self.assertEqual(self.invoke().status, ToolStatus.UNCERTAIN)
            step = self.ledger.steps(self.task.task_id)[0]
            self.assertEqual((step.state, step.error), ('uncertain', 'timeout'))
            self.assertEqual(self.calls, ['started'])
        finally:
            release.set()
            self.assertTrue(stopped.wait(5))
        self.assertEqual(self.ledger.steps(self.task.task_id)[0].state, 'uncertain')

    def test_oversized_structured_and_string_outputs(self):
        for payload in ({'ok': True, 'secret_output': 'x' * 1000}, json.dumps({'ok': True, 'data': 'x' * 1000})):
            self.register(lambda: payload, output_limit_bytes=32)
            result = self.invoke(idempotency_key=str(type(payload)))
            self.assertEqual(result.status, ToolStatus.OUTPUT_LIMIT)
            self.assertIsNone(result.data)
            step = self.ledger.get_step(self.task.task_id, result.step_id)
            self.assertIsNone(step.result)
            self.assertEqual(step.state, 'uncertain')
            self.assertNotIn('secret_output', step.model_dump_json())

    def test_execution_failure_typed_and_persisted(self):
        def fail():
            self.calls.append('effect')
            raise RuntimeError('sensitive failure content')
        self.register(fail)
        with self.assertRaises(ToolDenied) as caught:
            self.host.execute_tool_and_checkpoint(self.task.task_id, 'fixture', {})
        self.assertEqual(caught.exception.result.status, ToolStatus.EXECUTION_FAILURE)
        step = self.ledger.steps(self.task.task_id)[0]
        self.assertEqual((step.state, step.error), ('uncertain', 'RuntimeError'))
        self.assertNotIn('sensitive', step.model_dump_json())
        self.assertEqual(self.invoke().status, ToolStatus.UNCERTAIN)
        self.assertEqual(self.calls, ['effect'])

    def test_restart_reuses_verified_effect(self):
        self.register()
        self.assertEqual(self.invoke().status, ToolStatus.SUCCESS)
        self.broker.ledger = TaskLedger(self.root / 'ledger.db')
        self.assertEqual(self.invoke().status, ToolStatus.SUCCESS)
        self.assertEqual(len(self.calls), 1)

    def test_workspace_wrapper_denies_bad_args_before_write(self):
        previous = os.environ.get('AI_CLOUD_WORKSPACE')
        os.environ['AI_CLOUD_WORKSPACE'] = str(self.root)
        try:
            wrapper = workspace_method(Tools(), 'write_text_file')
        finally:
            if previous is None:
                os.environ.pop('AI_CLOUD_WORKSPACE', None)
            else:
                os.environ['AI_CLOUD_WORKSPACE'] = previous
        self.broker.register(ToolManifest(tool_id='fixture', permissions={'write_workspace'}, risk='R1',
                                         filesystem_scope=str(self.root)), wrapper)
        self.assert_denied(self.invoke({'relative_path': 'denied.txt', 'content': 'no', 'extra': 1}),
                           ToolStatus.VALIDATION_DENIED)
        self.assertFalse((self.root / 'denied.txt').exists())

    def test_approval_denial_does_not_leave_pending_effect(self):
        self.register(approval_policy='always')
        self.assert_denied(self.invoke(), ToolStatus.APPROVAL_REQUIRED)
        self.assertEqual(self.ledger.steps(self.task.task_id), [])

    def test_unsupported_schema_even_on_optional_field_denied(self):
        self.register(arguments_schema={'type': 'object', 'properties': {'unused': {'type': 'string', 'pattern': 'x'}}})
        self.assert_denied(self.invoke(), ToolStatus.POLICY_DENIED)

    def test_scope_change_invalidates_approval(self):
        self.paths()
        manifest = self.broker.manifest('fixture')
        manifest.approval_policy = 'always'
        self.broker.register(manifest, lambda **args: self.calls.append(args), required_permissions={'write_workspace'})
        approval = self.approve({'value': 'file.txt'})
        manifest.filesystem_scope = str(self.root / 'narrower')
        self.broker.register(manifest, lambda **args: self.calls.append(args), required_permissions={'write_workspace'})
        self.assert_denied(self.invoke({'value': 'file.txt'}, approval_id=approval), ToolStatus.APPROVAL_INVALID)

    def test_direct_effect_requires_persistent_ledger(self):
        self.broker.ledger = None
        self.register()
        self.assertEqual(self.invoke().status, ToolStatus.POLICY_DENIED)
        self.assertEqual(self.calls, [])

    def test_schema_structural_nested_validation(self):
        schema = {'type': 'object', 'properties': {'items': {'type': 'array', 'items':
                  {'type': 'object', 'properties': {'count': {'type': 'integer'}}, 'required': ['count']}}},
                  'required': ['items']}
        self.register(arguments_schema=schema)
        self.assert_denied(self.invoke({'items': [{'count': True}]}), ToolStatus.VALIDATION_DENIED)
        self.assertEqual(self.invoke({'items': [{'count': 1}]}).status, ToolStatus.SUCCESS)


if __name__ == '__main__':
    unittest.main()
