"""Identity and evidence contract fixtures, not physical hardware qualification."""
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from dante.contracts import utc_now
from dante.contracts.qualification import Check, CheckEvidence, QualificationState
from dante.ledger import TaskLedger
from dante.nvidia_probe import CommandOutput, discover_gpu_uuids
from dante.ollama_identity import OllamaIdentityObserver
from dante.qualification import QualificationRunner, QualificationStore, changed_inputs
from dante.qualification_evidence import EvidenceDocument, EvidenceStore
from test_ollama_qualification import OllamaQualificationTests
from test_qualification import changed, identity, record


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = EvidenceStore(self.root / 'evidence', strict=True)
        self.store = QualificationStore(TaskLedger(self.root / 'node.db'), self.evidence)
        fixture = OllamaQualificationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.fixture.runtime.start()
        self.machine = changed(identity().machine, gpus=(
            changed(identity().machine.gpus[0], uuid='GPU-fixture'),))
        self.observer = OllamaIdentityObserver(self.fixture.current, fixture.config,
            fixture.runtime, fixture.http, fixture.nvidia,
            machine_probe=lambda _: self.machine)
        self.current = self.observer()

    def qualified_record(self):
        attempt = uuid4()
        checks = []
        for check in Check:
            now = utc_now()
            document = EvidenceDocument(identity_fingerprint=self.current.fingerprint,
                node_id=str(self.current.machine.node_id), attempt_id=attempt,
                check=check, outcome='passed', started_at=now, completed_at=now,
                deadline_s=60, gpu_uuid='GPU-fixture', gpu_name=self.machine.gpus[0].name,
                gpu_driver=self.machine.gpus[0].driver_version,
                runtime_id=self.current.runtime.runtime_id,
                runtime_version=self.current.runtime.version,
                runtime_configuration_sha256=self.current.runtime.configuration_sha256,
                model_reference='fixture:1', model_digest=self.current.artifact_sha256,
                quantization=self.current.quantization,
                context_tokens=self.current.context_tokens,
                qualification_configuration_sha256=self.current.configuration_sha256)
            checks.append(CheckEvidence(check=check, outcome='passed',
                                        evidence_sha256=self.evidence.put(document)))
        # Contract fixture only: no offline probe emits hardware source.
        return record(self.current, source='hardware', attempt_id=attempt, checks=tuple(checks))

    def test_observer_derives_cuda_only_from_resident_model_and_process_telemetry(self):
        self.assertEqual(self.current.runtime.backend, 'cuda')
        self.assertEqual(self.current.artifact_sha256, 'a' * 64)
        self.fixture.nvidia.result = False
        self.assertIsNone(self.observer().runtime.backend)
        self.fixture.nvidia.result = True
        self.fixture.http.digest = 'b' * 64
        self.assertNotEqual(self.observer().artifact_sha256, self.current.artifact_sha256)

    def test_exact_and_material_drift(self):
        self.store.append(self.qualified_record())
        self.assertEqual(self.store.status(self.current).state, QualificationState.QUALIFIED)
        variants = [
            changed(self.current, machine=changed(self.machine, node_id=uuid4())),
            changed(self.current, machine=changed(self.machine, gpus=(
                changed(self.machine.gpus[0], uuid='GPU-other'),))),
            changed(self.current, machine=changed(self.machine, gpus=(
                changed(self.machine.gpus[0], driver_version='other'),))),
            changed(self.current, runtime=changed(self.current.runtime, version='other')),
            changed(self.current, runtime=changed(self.current.runtime,
                configuration_sha256='b' * 64)),
            changed(self.current, artifact_sha256='b' * 64),
            changed(self.current, quantization='Q8_0'),
            changed(self.current, context_tokens=2048),
            changed(self.current, configuration_sha256='b' * 64),
        ]
        for variant in variants:
            with self.subTest(changed=changed_inputs(self.current, variant)):
                self.assertIn(self.store.status(variant).state,
                              {QualificationState.STALE, QualificationState.UNKNOWN})
                with self.assertRaises(ValueError):
                    self.store.require_qualified(variant)
        noisy = changed(self.current, machine=changed(self.machine,
            storage_free_bytes=1234, observed_at=utc_now()))
        self.assertEqual(self.store.status(noisy).state, QualificationState.QUALIFIED)

    def test_missing_corrupt_and_mismatched_evidence(self):
        item = self.qualified_record()
        self.store.append(item)
        path = self.evidence.root / (item.checks[0].evidence_sha256 + '.json')
        path.write_text('{}', encoding='utf-8')
        self.assertEqual(self.store.status(self.current).state, QualificationState.UNKNOWN)
        path.unlink()
        self.assertEqual(self.store.status(self.current).state, QualificationState.UNKNOWN)

    def test_partial_write_and_wrong_identity_document_never_authorize(self):
        item = self.qualified_record()
        self.store.append(item)
        original = self.evidence.root / (item.checks[0].evidence_sha256 + '.json')
        original.unlink()
        (self.evidence.root / '.evidence-interrupted.tmp').write_text('{', encoding='utf-8')
        self.assertEqual(self.store.status(self.current).state, QualificationState.UNKNOWN)
        now = utc_now()
        wrong = EvidenceDocument(identity_fingerprint=self.current.fingerprint,
            node_id=str(self.current.machine.node_id), attempt_id=item.attempt_id,
            check=Check.LOAD, outcome='passed', started_at=now, completed_at=now,
            deadline_s=60, gpu_uuid='GPU-fixture', gpu_name=self.machine.gpus[0].name,
            gpu_driver='wrong', runtime_id=self.current.runtime.runtime_id,
            runtime_version=self.current.runtime.version,
            runtime_configuration_sha256=self.current.runtime.configuration_sha256,
            model_reference='fixture:1', model_digest=self.current.artifact_sha256,
            quantization=self.current.quantization, context_tokens=self.current.context_tokens,
            qualification_configuration_sha256=self.current.configuration_sha256)
        key = self.evidence.put(wrong)
        changed_checks = (CheckEvidence(check=Check.LOAD, outcome='passed',
            evidence_sha256=key),) + item.checks[1:]
        self.store.append(record(self.current, source='hardware',
            attempt_id=item.attempt_id, checks=changed_checks))
        self.assertEqual(self.store.status(self.current).state, QualificationState.UNKNOWN)

    def test_failed_rerun_and_interrupted_intent_supersede_pass(self):
        self.store.append(self.qualified_record())
        class BadObservation:
            source = 'synthetic'
            def observe(self):
                raise RuntimeError('fixture unavailable')
        attempt = QualificationRunner(self.store).run(BadObservation(), requested_identity=self.current)
        self.assertTrue(attempt.interrupted)
        self.assertEqual(self.store.status(self.current).state, QualificationState.FAILED)

    def test_gpu_uuid_query_fails_closed(self):
        query = lambda _: CommandOutput(0, '0, GPU-fixture\n')
        self.assertEqual(discover_gpu_uuids(query), {'0': 'GPU-fixture'})
        self.assertEqual(discover_gpu_uuids(lambda _: CommandOutput(0, '0, bad/uuid\n')), {})
        self.assertEqual(discover_gpu_uuids(lambda _: CommandOutput(1, '')), {})


if __name__ == '__main__':
    unittest.main()
