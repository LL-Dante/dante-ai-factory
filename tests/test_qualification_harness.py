"""Qualification harness tests use injected fixtures; no GPU or runtime is contacted."""
from contextlib import closing
from pathlib import Path
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from dante.cli import main
from dante.contracts.qualification import Check, CheckEvidence, PerformanceEvidence, QualificationState
from dante.ledger import TaskLedger
from dante.node_qualification import NodeQualificationHarness
from dante.qualification import QualificationStore
from dante.recovery import digest
from test_qualification import changed, identity, record


def complete_profile(node_id):
    return changed(identity().machine, node_id=node_id)


class FixtureProbe:
    def __init__(self, current, *, source='hardware', failed=False, stale=False):
        self.current, self.source, self.failed, self.stale = current, source, failed, stale

    def observe(self):
        if self.stale and hasattr(self, '_observed'):
            return changed(self.current, configuration_sha256='f' * 64)
        self._observed = True
        return self.current

    def run(self, check):
        if self.failed and check == Check.LOAD:
            return CheckEvidence(check=check, outcome='failed', failure='unavailable')
        return CheckEvidence(check=check, outcome='passed', evidence_sha256=digest(['harness', check]))

    def performance(self):
        return PerformanceEvidence()


class QualificationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = TaskLedger(self.root / 'node.db')
        self.store = QualificationStore(self.ledger)
        self.node_id_file = self.root / 'node-id.txt'
        self.node_id_file.write_text(str(identity().machine.node_id) + '\n', encoding='ascii')

    def harness(self, factory=None):
        return NodeQualificationHarness(self.store, self.node_id_file,
            machine_probe=complete_profile, probe_factory=factory)

    def test_default_attempt_is_unknown_without_fake_evidence(self):
        result = self.harness().qualify(identity())
        self.assertEqual(result.qualification_state, QualificationState.UNKNOWN)
        self.assertEqual(result.qualification_status, 'UNKNOWN')
        self.assertIn('synthetic_evidence_only', result.reasons)
        history = self.store.history(result.node_id)
        self.assertEqual([item.phase for item in history], ['intent', 'completed'])
        self.assertEqual({item.attempt_id for item in history}, {result.attempt_id})
        self.assertTrue(all(check.outcome == 'unknown' for check in history[-1].checks))
        self.assertTrue(all(check.evidence_sha256 is None for check in history[-1].checks))

    def test_injected_hardware_probe_can_qualify(self):
        result = self.harness(lambda current: FixtureProbe(current)).qualify(identity())
        self.assertEqual(result.qualification_state, QualificationState.QUALIFIED)
        self.assertEqual(result.qualification_status, 'QUALIFIED')
        self.assertEqual(result.reasons, ())

    def test_injected_failure_is_failed_and_persisted(self):
        result = self.harness(lambda current: FixtureProbe(current, failed=True)).qualify(identity())
        self.assertEqual(result.qualification_status, 'FAILED')
        restored = QualificationStore(TaskLedger(self.ledger.path)).get(result.qualification_id)
        self.assertEqual(restored.attempt_id, result.attempt_id)
        self.assertEqual(restored.checks[0].failure, 'unavailable')

    def test_identity_change_during_attempt_is_failed(self):
        result = self.harness(lambda current: FixtureProbe(current, stale=True)).qualify(identity())
        self.assertEqual(result.qualification_status, 'FAILED')
        self.assertIn('probe_interrupted', result.reasons)

    def test_probe_for_different_runtime_identity_is_stale(self):
        result = self.harness(lambda current: FixtureProbe(
            changed(current, runtime=changed(current.runtime, version='fixture-2')))).qualify(identity())
        self.assertEqual(result.qualification_state, QualificationState.STALE)
        self.assertEqual(result.qualification_status, 'STALE')
        self.assertIn('changed:runtime.version', result.reasons)

    def test_stale_previous_state_is_reported_before_new_attempt(self):
        old = identity()
        self.store.append(record(old, source='hardware'))
        current = changed(old, runtime=changed(old.runtime, version='fixture-2'))
        result = self.harness().qualify(current)
        self.assertEqual(result.previous_state, QualificationState.STALE)
        self.assertEqual(result.qualification_status, 'UNKNOWN')

    def test_different_node_identity_fails_before_attempt(self):
        other = changed(identity(), machine=changed(identity().machine,
            node_id='00000000-0000-4000-8000-000000000002'))
        with self.assertRaises(ValueError):
            self.harness().qualify(other)
        self.assertEqual(self.store.history(identity().machine.node_id), [])

    def test_cli_default_flow_persists_unknown_result(self):
        current = self.root / 'identity.json'
        current.write_text(identity().model_dump_json(), encoding='utf-8')
        arguments = ['--db', str(self.ledger.path), 'node-qualify',
                     '--node-id-file', str(self.node_id_file), '--identity', str(current)]
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(arguments, node_machine_probe=complete_profile), 0)
            result = json.loads(output.getvalue())
        self.assertEqual(result['qualification_status'], 'UNKNOWN')
        self.assertEqual(len(self.store.history(identity().machine.node_id)), 2)

    def test_cli_accepts_injected_probe_factory(self):
        current = self.root / 'identity.json'
        current.write_text(identity().model_dump_json(), encoding='utf-8')
        arguments = ['--db', str(self.ledger.path), 'node-qualify',
                     '--node-id-file', str(self.node_id_file), '--identity', str(current)]
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(arguments, node_machine_probe=complete_profile,
                qualification_probe_factory=lambda observed: FixtureProbe(observed)), 0)
            result = json.loads(output.getvalue())
        self.assertEqual(result['qualification_status'], 'QUALIFIED')

    def test_uncaught_process_interrupt_leaves_intent_unknown(self):
        class InterruptProbe(FixtureProbe):
            def run(self, check):
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.harness(lambda current: InterruptProbe(current)).qualify(identity())
        history = QualificationStore(TaskLedger(self.ledger.path)).history(identity().machine.node_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].phase, 'intent')
        self.assertIsNone(history[0].completed_at)

    def test_legacy_record_without_attempt_fields_still_verifies(self):
        legacy = record()
        payload = json.loads(legacy.model_dump_json())
        payload.pop('attempt_id')
        payload.pop('phase')
        with closing(self.ledger._connect()) as database, database:
            database.execute('''INSERT INTO model_qualifications
                (qualification_id,node_id,scope,payload,payload_digest) VALUES(?,?,?,?,?)''',
                (str(legacy.qualification_id), str(legacy.identity.machine.node_id), legacy.identity.scope,
                 json.dumps(payload), digest(payload)))
        restored = self.store.get(legacy.qualification_id)
        self.assertIsNone(restored.attempt_id)
        self.assertEqual(restored.phase, 'completed')


if __name__ == '__main__':
    unittest.main()
