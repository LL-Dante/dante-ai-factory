"""Node bootstrap is local inventory only; these tests do not qualify hardware."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

from dante.cli import main
from dante.contracts.qualification import MachineProfile, QualificationState
from dante.ledger import TaskLedger
from dante.node_bootstrap import bootstrap_node, inspect_node, persistent_node_id
from dante.qualification import QualificationStore


NODE = UUID('00000000-0000-4000-8000-000000000001')


def fixture_profile(node_id):
    return MachineProfile(node_id=node_id, os='FixtureOS', os_version='1',
        architecture='fixture64', cpu='fixture-cpu', logical_cpus=4, ram_bytes=8192)


class NodeBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = QualificationStore(TaskLedger(self.root / 'node.db'))
        self.node_id_file = self.root / 'state' / 'node-id.txt'

    def test_creates_then_reuses_persistent_uuid(self):
        first = bootstrap_node(self.store, self.node_id_file, probe=fixture_profile)
        second = bootstrap_node(self.store, self.node_id_file, probe=fixture_profile)
        self.assertTrue(first.node_id_created)
        self.assertFalse(second.node_id_created)
        self.assertEqual(first.node_id, second.node_id)
        self.assertEqual(self.node_id_file.read_text(encoding='ascii').strip(), str(first.node_id))
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(len(self.store.machines(first.node_id)), 2)

    def test_concurrent_initialization_selects_one_complete_uuid(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: persistent_node_id(self.node_id_file), range(16)))
        self.assertEqual(len({node_id for node_id, _ in results}), 1)
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertEqual(UUID(self.node_id_file.read_text(encoding='ascii').strip()).version, 4)

    def test_invalid_existing_identity_is_not_replaced(self):
        self.node_id_file.parent.mkdir(parents=True)
        self.node_id_file.write_text('not-a-node-id\n', encoding='ascii')
        with self.assertRaises(ValueError):
            persistent_node_id(self.node_id_file)
        self.assertEqual(self.node_id_file.read_text(encoding='ascii'), 'not-a-node-id\n')

    def test_noncanonical_or_non_v4_identity_is_rejected(self):
        self.node_id_file.parent.mkdir(parents=True)
        for value in ('00000000-0000-0000-0000-000000000000',
                      '123E4567-E89B-42D3-A456-426614174000'):
            with self.subTest(value=value):
                self.node_id_file.write_text(value, encoding='ascii')
                with self.assertRaises(ValueError):
                    persistent_node_id(self.node_id_file)

    def test_inspection_rejects_wrong_node(self):
        with self.assertRaises(ValueError):
            inspect_node(self.store, NODE, lambda _: fixture_profile(UUID('00000000-0000-4000-8000-000000000002')))
        self.assertEqual(self.store.machines(NODE), [])

    def test_report_is_explicitly_not_yet_qualified(self):
        report = bootstrap_node(self.store, self.node_id_file, probe=fixture_profile)
        self.assertEqual(report.qualification_state, QualificationState.UNKNOWN)
        self.assertEqual(report.qualification_status, 'NOT_YET_QUALIFIED')
        self.assertEqual(report.reasons, ('hardware_evidence_missing', 'runtime_evidence_missing'))
        self.assertIsNone(report.machine.gpus)
        self.assertNotIn('qualified', report.model_dump_json().lower().replace('not_yet_qualified', ''))

    def test_cli_bootstrap_reuses_id_and_persists_profile(self):
        database = self.root / 'cli.db'
        arguments = ['--db', str(database), 'node-bootstrap', '--node-id-file', str(self.node_id_file)]
        with patch('dante.node_bootstrap.probe_machine', fixture_profile), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(arguments), 0)
            first = json.loads(output.getvalue())
        with patch('dante.node_bootstrap.probe_machine', fixture_profile), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(arguments), 0)
            second = json.loads(output.getvalue())
        self.assertTrue(first['node_id_created'])
        self.assertFalse(second['node_id_created'])
        self.assertEqual(first['node_id'], second['node_id'])
        self.assertEqual(second['qualification_status'], 'NOT_YET_QUALIFIED')
        profiles = QualificationStore(TaskLedger(database)).machines(UUID(first['node_id']))
        self.assertEqual(len(profiles), 2)

    def test_node_inspect_uses_same_status_vocabulary(self):
        database = self.root / 'inspect.db'
        with patch('dante.node_bootstrap.probe_machine', fixture_profile), patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--db', str(database), 'node-inspect', '--node-id', str(NODE)]), 0)
            result = json.loads(output.getvalue())
        self.assertEqual(result['qualification_state'], 'unknown')
        self.assertEqual(result['qualification_status'], 'NOT_YET_QUALIFIED')
        self.assertEqual(result['machine']['node_id'], str(NODE))

    def test_cli_machine_probe_injection_applies_to_inspect_and_bootstrap(self):
        database = self.root / 'injected.db'
        commands = (
            ['--db', str(database), 'node-inspect', '--node-id', str(NODE)],
            ['--db', str(database), 'node-bootstrap', '--node-id-file', str(self.node_id_file)],
        )
        for arguments in commands:
            with self.subTest(command=arguments[2]), patch('sys.stdout', new_callable=io.StringIO) as output:
                self.assertEqual(main(arguments, node_machine_probe=fixture_profile), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result['machine']['os'], 'FixtureOS')


if __name__ == '__main__':
    unittest.main()
