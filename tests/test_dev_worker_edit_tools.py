import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
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
        self.assertEqual(self.registered, ('SEARCH_TEXT', 'READ_FILE_RANGE', 'PATCH_FILE', 'WRITE_FILE'))

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

    def test_read_file_range_is_line_numbered_and_bounded(self):
        (self.root / 'src.py').write_text('one\ntwo\nthree\n', encoding='utf-8')
        result = self.broker._run(self.task, 'READ_FILE_RANGE', {'path': 'src.py', 'start_line': 2, 'end_line': 3})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual([row['line'] for row in result.data['lines']], [2, 3])
        self.assertEqual(result.data['lines'][0]['text'], 'two')
        self.assertRegex(result.data['result_sha256'], r'^[0-9a-f]{64}$')

    def test_patch_file_requires_unique_expected_and_reports_hashes(self):
        target = self.root / 'src.py'
        target.write_text('def add(a, b):\n    return a - b\n', encoding='utf-8')
        result = self.broker._run(self.task, 'PATCH_FILE', {'path': 'src.py',
            'expected': 'return a - b', 'replacement': 'return a + b'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertTrue(result.data['ok'])
        self.assertNotEqual(result.data['pre_sha256'], result.data['post_sha256'])
        self.assertIn('return a + b', target.read_text(encoding='utf-8'))
        mismatch = self.broker._run(self.task, 'PATCH_FILE', {'path': 'src.py',
            'expected': 'missing', 'replacement': 'x'})
        self.assertFalse(mismatch.data['operation_ok'])
        self.assertEqual(mismatch.data['error']['code'], 'expected_text_mismatch')

    def test_patch_ambiguous_and_range_validation_do_not_mutate(self):
        target = self.root / 'same.txt'
        target.write_text('x x', encoding='utf-8')
        result = self.broker._run(self.task, 'PATCH_FILE', {'path': 'same.txt', 'expected': 'x', 'replacement': 'y'})
        self.assertEqual(result.data['error']['code'], 'expected_text_ambiguous')
        bad_range = self.broker._run(self.task, 'READ_FILE_RANGE', {'path': 'same.txt', 'start_line': 1, 'end_line': 101})
        self.assertEqual(bad_range.data['error']['code'], 'invalid_line_range')
        self.assertEqual(target.read_text(encoding='utf-8'), 'x x')

    def test_patch_atomic_failure_preserves_original_and_cleans_temp(self):
        target = self.root / 'atomic.py'
        original = 'def add(a, b):\n    return a - b\n'
        target.write_text(original, encoding='utf-8')
        with patch('dante.dev_worker_edit_tools.os.replace', side_effect=OSError('fixture')):
            result = self.broker._run(self.task, 'PATCH_FILE', {'path': 'atomic.py',
                'expected': 'return a - b', 'replacement': 'return a + b'})
        self.assertEqual(result.data['error']['code'], 'patch_failed')
        self.assertEqual(target.read_text(encoding='utf-8'), original)
        self.assertEqual(list(self.root.glob('.dante-patch-*.tmp')), [])

    def test_patch_encoding_size_and_sensitive_path_fail_without_mutation(self):
        invalid = self.root / 'invalid.py'
        invalid.write_bytes(b'\xff\xfe')
        result = self.broker._run(self.task, 'PATCH_FILE', {'path': 'invalid.py',
            'expected': 'x', 'replacement': 'y'})
        self.assertEqual(result.data['error']['code'], 'invalid_encoding')
        target = self.root / 'large.txt'
        target.write_text('x', encoding='utf-8')
        oversized = self.broker._run(self.task, 'PATCH_FILE', {'path': 'large.txt',
            'expected': 'x', 'replacement': 'y' * 8192})
        self.assertEqual(oversized.data['error']['code'], 'patch_too_large')
        secret = self.root / '.env'
        secret.write_text('keep', encoding='utf-8')
        denied = self.broker._run(self.task, 'PATCH_FILE', {'path': '.env',
            'expected': 'keep', 'replacement': 'changed'})
        self.assertEqual(denied.status, ToolStatus.EXECUTION_FAILURE)
        self.assertEqual(secret.read_text(encoding='utf-8'), 'keep')

    def test_range_encoding_error_is_structured(self):
        invalid = self.root / 'binary.txt'
        invalid.write_bytes(b'\xff')
        result = self.broker._run(self.task, 'READ_FILE_RANGE', {'path': 'binary.txt',
            'start_line': 1, 'end_line': 1})
        self.assertEqual(result.data['error']['code'], 'invalid_encoding')

if __name__ == '__main__':
    unittest.main()

