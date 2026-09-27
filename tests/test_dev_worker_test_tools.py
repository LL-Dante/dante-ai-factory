import tempfile
import unittest
from pathlib import Path

from dante.contracts import Task
from dante.contracts.tools import ToolStatus
from dante.dev_worker_test_tools import register_test_tool
from dante.tool_broker import ToolBroker


class DevelopmentTestToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'tests').mkdir()
        self.broker = ToolBroker()
        self.tools = register_test_tool(self.broker, str(self.root))
        self.task = Task(goal='run bounded tests', workspace=str(self.root))

    def tearDown(self):
        self.temp.cleanup()

    def test_runs_only_selected_unittest_file(self):
        (self.root / 'tests' / 'test_math.py').write_text(
            'import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+2,4)\n',
            encoding='utf-8')
        result = self.broker._run(self.task, 'RUN_TESTS', {'target': 'tests/test_math.py'})
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertTrue(result.data['ok'])
        self.assertIn('Ran 1 test', result.data['output'])
        self.assertIn('not an OS sandbox', result.data['isolation'])

    def test_rejects_arbitrary_paths_and_targets(self):
        for target in ('../outside.py', 'C:\\temp\\test_x.py', 'tests/evil.py',
                       'tests/test_x.py --help', 'tests/test_x.py/../x.py'):
            result = self.broker.preflight(self.task, 'RUN_TESTS', {'target': target})
            self.assertIn(result.status, {ToolStatus.POLICY_DENIED, ToolStatus.VALIDATION_DENIED})

    def test_rejects_linked_test_target(self):
        source = self.root / 'outside.py'
        source.write_text('pass\n', encoding='utf-8')
        link = self.root / 'tests' / 'test_link.py'
        try:
            link.symlink_to(source)
        except OSError:
            self.skipTest('Symlinks unavailable')
        with self.assertRaises(ValueError):
            from dante.dev_worker_test_tools import _run_tests
            _run_tests(self.root, 'tests/test_link.py')

    def test_service_registers_only_fixed_test_runner(self):
        self.assertEqual(self.tools, ('RUN_TESTS',))


if __name__ == '__main__':
    unittest.main()
