import unittest
import tempfile
from pathlib import Path
from dante.contracts import Task
from dante.tool_broker import ToolBroker
from dante.contracts.tools import ToolStatus
from dante.dev_worker_tools import register_coding_tools


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
        result = self.broker._run(self.task, 'LIST_FILES', {'path': ''})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data['entries'], ['a.txt', 'b.txt'])

    def test_traversal_denied(self):
        result = self.broker.preflight(self.task, 'READ_FILE', {'path': '../outside'})
        self.assertEqual(result.status, ToolStatus.POLICY_DENIED)

    def test_sensitive_files_are_not_returned(self):
        (self.root / '.env').write_text('unique-secret-marker', encoding='utf-8')
        result = self.broker._run(self.task, 'LIST_FILES', {'path': '.'})
        self.assertNotIn('.env', result.data['entries'])
        read = self.broker._run(self.task, 'READ_FILE', {'path': '.env'})
        self.assertEqual(read.status, ToolStatus.EXECUTION_FAILURE)

    def test_absolute_path_denied(self):
        result = self.broker.preflight(self.task, 'READ_FILE', {'path': '/etc/passwd'})
        self.assertEqual(result.status, ToolStatus.POLICY_DENIED)

    def test_unknown_tool_denied(self):
        result = self.broker.preflight(self.task, 'RUN_TESTS', {})
        self.assertEqual(result.status, ToolStatus.POLICY_DENIED)

    def test_oversized_read_failure(self):
        big = self.root / 'big.txt'
        big.write_text('x' * (24 * 1024 + 1))
        result = self.broker._run(self.task, 'READ_FILE', {'path': 'big.txt'})
        self.assertEqual(result.status, ToolStatus.EXECUTION_FAILURE)
        self.assertIsNone(result.data)


if __name__ == '__main__':
    unittest.main()
