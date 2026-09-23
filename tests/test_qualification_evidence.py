"""Offline integrity and deadline tests; no hardware qualification occurs."""
from pathlib import Path
import tempfile
import threading
import unittest

from dante.contracts import utc_now
from dante.contracts.qualification import Check, CheckEvidence, QualificationState
from dante.ledger import TaskLedger
from dante.qualification import QualificationRunner, QualificationStore
from dante.qualification_evidence import EvidenceDocument, EvidenceStore
from test_qualification import FakeProbe, identity, record


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = EvidenceStore(self.root / 'evidence')
        self.store = QualificationStore(TaskLedger(self.root / 'node.db'), self.evidence)
        self.current = identity()

    def document(self, check=Check.LOAD):
        now = utc_now()
        return EvidenceDocument(identity_fingerprint=self.current.fingerprint,
            node_id=str(self.current.machine.node_id), check=check, outcome='passed',
            started_at=now, completed_at=now, facts={'observed': True})

    def test_atomic_roundtrip_and_tampering(self):
        key = self.evidence.put(self.document())
        self.assertTrue(self.evidence.verify(key, self.current.fingerprint,
            str(self.current.machine.node_id), Check.LOAD, 'passed'))
        self.assertFalse(self.evidence.verify(key, self.current.fingerprint,
            str(self.current.machine.node_id), Check.GENERATION, 'passed'))
        (self.evidence.root / (key + '.json')).write_text('{}', encoding='utf-8')
        self.assertFalse(self.evidence.verify(key, self.current.fingerprint,
            str(self.current.machine.node_id), Check.LOAD, 'passed'))

    def test_missing_or_corrupt_retained_evidence_denies_qualified(self):
        checks = tuple(CheckEvidence(check=check, outcome='passed',
            evidence_sha256=self.evidence.put(self.document(check))) for check in Check)
        self.store.append(record(current=self.current, source='hardware', checks=checks))
        self.assertEqual(self.store.status(self.current).state, QualificationState.QUALIFIED)
        key = checks[0].evidence_sha256
        (self.evidence.root / (key + '.json')).unlink()
        self.assertEqual(self.store.status(self.current).state, QualificationState.UNKNOWN)
        with self.assertRaises(ValueError):
            self.store.require_qualified(self.current)

    def test_deadline_fails_closed_and_records_attempt(self):
        probe = FakeProbe()
        release = threading.Event()
        self.addCleanup(release.set)
        probe.run = lambda check: (release.wait(2), CheckEvidence(
            check=check, outcome='unknown'))[1]
        result = QualificationRunner(self.store, check_deadline_s=0.01).run(probe)
        self.assertTrue(result.interrupted)
        self.assertEqual(result.checks[0].failure, 'timeout')
        self.assertEqual(self.store.status(probe.current).state, QualificationState.FAILED)


if __name__ == '__main__':
    unittest.main()
