"""CLI failure-boundary regressions; no external process or provider is used."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dante.cli import main


class CliHardeningTests(unittest.TestCase):
    def test_database_open_failure_returns_sanitized_cli_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            database_is_directory = Path(temporary)
            with patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(main(['--db', str(database_is_directory), 'list']), 1)
            self.assertEqual(json.loads(output.getvalue()), {'error': 'local_command_failed'})


if __name__ == '__main__':
    unittest.main()
