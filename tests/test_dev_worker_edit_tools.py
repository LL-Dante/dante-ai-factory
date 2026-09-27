import unittest
import tempfile
from pathlib import Path
from dante.contracts import Task
from dante.contracts.tools import ToolStatus
from dante.tool_broker import ToolBroker
from dante.dev_worker_edit_tools import register_edit_tools

class TestDevWorkerEditTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.broker = ToolBroker()
        self.registered = register_edit_tools(self.broker, str(self.root))
        self.task = Task(goal='edit tools', workspace=str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_registered_tools(self):
        self.assertEqual(self.registered, ('SEARCH_TEXT', 'WRITE_FILE'))

    def test_search_text(self):
        (self.root / 'a.txt').write_text('Alpha target\nother\ntarget again\n')
        result = self.broker._run(self.task, 'SEARCH_TEXT', {'path': '.', 'query': 'TARGET'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(len(result.data['matches']), 2)
        self.assertEqual(result.data['matches'][0]['line'], 1)

    def test_write_file(self):
        result = self.broker._run(self.task, 'WRITE_FILE', {'path': 'nested/new.txt', 'content': 'héllo'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data['bytes_written'], 6)
        self.assertEqual((self.root / 'nested' / 'new.txt').read_text(encoding='utf-8'), 'h\u00e9llo')

    def test_traversal_denied(self):
        r1 = self.broker.preflight(self.task, 'SEARCH_TEXT', {'path': '../escape'})
        self.assertIn(r1.status, {ToolStatus.POLICY_DENIED, ToolStatus.VALIDATION_DENIED})
        r2 = self.broker.preflight(self.task, 'WRITE_FILE', {'path': '..\\escape'})
        self.assertIn(r2.status, {ToolStatus.POLICY_DENIED, ToolStatus.VALIDATION_DENIED})

    def test_oversize_write(self):
        content = 'x' * (24 * 1024 + 1)
        result = self.broker._run(self.task, 'WRITE_FILE', {'path': 'big.txt', 'content': content})
        self.assertEqual(result.status, ToolStatus.EXECUTION_FAILURE)

    def test_no_run_tests_and_unknown_tool(self):
        self.assertNotIn('RUN_TESTS', self.registered)
        r = self.broker.preflight(self.task, 'UNKNOWN_TOOL', {})
        self.assertEqual(r.status, ToolStatus.POLICY_DENIED)

    def test_search_and_write_hide_sensitive_paths(self):
        (self.root / '.env').write_text('secret-sentinel', encoding='utf-8')
        searched = self.broker._run(self.task, 'SEARCH_TEXT', {'path': '.', 'query': 'sentinel'})
        self.assertEqual(searched.data['matches'], [])
        denied = self.broker._run(self.task, 'WRITE_FILE', {'path': '.env', 'content': 'overwrite'})
        self.assertEqual(denied.status, ToolStatus.EXECUTION_FAILURE)
        self.assertEqual((self.root / '.env').read_text(encoding='utf-8'), 'secret-sentinel')

if __name__ == '__main__':
    unittest.main()

