import unittest
import tempfile
from pathlib import Path
from dante.contracts import Task
from dante.tool_broker import ToolBroker
from dante.contracts.tools import ToolStatus
from dante.dev_worker_tools import register_coding_tools
from dante.dev_agent_loop import DevelopmentAgentLoop


class TestDevWorkerTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.broker = ToolBroker()
        register_coding_tools(self.broker, str(self.root))
        self.task = Task(goal='test tools', workspace=str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_registered_tools(self):
        self.assertEqual(tuple(self.broker._tools), ('READ_FILE', 'LIST_FILES'))

    def test_read_file_success(self):
        (self.root / 'a.txt').write_text('alpha')
        (self.root / 'b.txt').write_text('beta')
        self.assertIsNone(self.broker.preflight(self.task, 'READ_FILE', {'path': 'a.txt'}))
        result = self.broker._run(self.task, 'READ_FILE', {'path': 'a.txt'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data['content'], 'alpha')

    def test_list_files_sorted(self):
        (self.root / 'a.txt').write_text('alpha')
        (self.root / 'b.txt').write_text('beta')
        (self.root / 'subdir').mkdir()
        result = self.broker._run(self.task, 'LIST_FILES', {'path': ''})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data['path'], '')
        self.assertEqual(result.data['entries'], [
            {'name': 'a.txt', 'type': 'file'},
            {'name': 'b.txt', 'type': 'file'},
            {'name': 'subdir', 'type': 'directory'},
        ])
        self.assertFalse(result.data['truncated'])

    def test_traversal_denied(self):
        result = self.broker.preflight(self.task, 'READ_FILE', {'path': '../outside'})
        self.assertEqual(result.status, ToolStatus.POLICY_DENIED)
        self.assertEqual(result.error_type, 'WorkspacePathDenied')

    def test_missing_file_is_structured_success_not_uncertain_failure(self):
        result = self.broker._run(self.task, 'READ_FILE', {'path': 'README.md'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data, {
            'ok': True, 'path': 'README.md', 'content': None,
            'error': {'code': 'not_found', 'message': 'No file exists at this workspace path.'},
        })
        self.assertFalse(result.effect_uncertain)
        summary, _digest = DevelopmentAgentLoop._tool_result_summary(result)
        self.assertIn('not_found', summary)

    def test_directory_is_structured_read_error(self):
        (self.root / 'folder').mkdir()
        result = self.broker._run(self.task, 'READ_FILE', {'path': 'folder'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data['error']['code'], 'not_a_file')

    def test_invalid_utf8_is_structured_read_error(self):
        (self.root / 'binary.txt').write_bytes(b'\xff')
        result = self.broker._run(self.task, 'READ_FILE', {'path': 'binary.txt'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data['error']['code'], 'invalid_encoding')
        self.assertIsNone(result.data['content'])

    def test_sensitive_files_are_not_returned(self):
        (self.root / '.env').write_text('unique-secret-marker', encoding='utf-8')
        result = self.broker._run(self.task, 'LIST_FILES', {'path': '.'})
        self.assertNotIn('.env', [item['name'] for item in result.data['entries']])
        read = self.broker._run(self.task, 'READ_FILE', {'path': '.env'})
        self.assertEqual(read.status, ToolStatus.SUCCESS)
        self.assertEqual(read.data['error']['code'], 'blocked_path')

    def test_absolute_path_denied(self):
        result = self.broker.preflight(self.task, 'READ_FILE', {'path': '/etc/passwd'})
        self.assertEqual(result.status, ToolStatus.POLICY_DENIED)
        self.assertEqual(result.error_type, 'WorkspacePathDenied')

    def test_unknown_tool_denied(self):
        result = self.broker.preflight(self.task, 'RUN_TESTS', {})
        self.assertEqual(result.status, ToolStatus.POLICY_DENIED)

    def test_oversized_read_failure(self):
        big = self.root / 'big.txt'
        big.write_text('x' * (24 * 1024 + 1))
        result = self.broker._run(self.task, 'READ_FILE', {'path': 'big.txt'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data['error']['code'], 'too_large')
        self.assertIsNone(result.data['content'])


if __name__ == '__main__':
    unittest.main()
