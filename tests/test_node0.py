"""Composition contract fixtures; no hardware or runtime is qualified here."""
from pathlib import Path
import tempfile
import unittest

from dante.contracts.qualification import QualificationState
from dante.contracts.runtime import RuntimeProfile
from dante.ollama_qualification import OllamaQualificationConfig
from dante.node0 import build_node0
from dante.routing import RouteDenied
from dante.privacy import PrivacyGate
from dante.contracts import Task, PrivacyClass
from test_qualification import identity
from test_local_runtime import model


class Node0CompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        seed = identity()
        self.node = build_node0(ledger_path=root / 'node.db', evidence_path=root / 'evidence',
            audit_path=root / 'audit.jsonl', seed_identity=seed,
            qualification=OllamaQualificationConfig(profile=RuntimeProfile(runtime='ollama',
                base_url='http://127.0.0.1:11434'), model_reference='fixture:1',
                model_digest='a' * 64, quantization='fixture', context_tokens=4096,
                gpu_uuid='GPU-fixture'), models=[model()], machine_profile='old-pc')

    def test_one_registry_is_used_throughout_and_unknown_denies(self):
        node = self.node
        self.assertIs(node.router.registry, node.registry)
        self.assertIs(node.gateway.registry, node.registry)
        self.assertIs(node.adapter.registry, node.registry)
        self.assertIs(node.gate.store, node.store)
        self.assertIs(node.host.router, node.router)
        self.assertIs(node.host.gateway, node.gateway)
        self.assertIs(node.continuity.router, node.router)
        self.assertIs(node.continuity.gateway, node.gateway)
        self.assertEqual(node.store.status(node.requested_identity).state, QualificationState.UNKNOWN)
        privacy = PrivacyGate().classify('hello', PrivacyClass.PUBLIC)
        with self.assertRaises(RouteDenied):
            node.router.decide(Task(goal='test', workspace='.', privacy_class=PrivacyClass.PUBLIC), privacy, set())

    def test_failed_preflight_supersedes_prior_scope_and_leaves_no_route(self):
        node = self.node
        class FailedObserver:
            def prepare(self):
                raise RuntimeError('fixture preflight failure')
            def __call__(self):
                return node.requested_identity
        node.probe.observer = FailedObserver()
        with self.assertRaises(RuntimeError):
            node.qualify()
        self.assertEqual(node.store.status(node.requested_identity).state, QualificationState.FAILED)
        with self.assertRaises(ValueError):
            node.registry.require_automatic(node.registry.get('local-fixture'))

    def test_observer_failure_is_deny_not_uncaught_runtime_error(self):
        self.node.probe.observer = lambda: (_ for _ in ()).throw(RuntimeError('offline'))
        with self.assertRaises(ValueError):
            self.node.registry.require_automatic(self.node.registry.get('local-fixture'))


if __name__ == '__main__':
    unittest.main()
