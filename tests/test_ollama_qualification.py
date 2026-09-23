"""Deterministic qualification fixtures; these never claim physical hardware."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from dante.contracts.qualification import Check, QualificationState
from dante.contracts.runtime import RuntimeProfile
from dante.ledger import TaskLedger
from dante.ollama_qualification import (
    ExternalOllamaRuntime, OllamaQualificationConfig, OllamaQualificationHTTP,
    OllamaQualificationProbeFactory, ProbeFailure,
)
from dante.qualification import QualificationRunner, QualificationStore
from dante.qualification_evidence import EvidenceStore
from test_qualification import changed, identity


PIN = 'a' * 64


class FakeRuntime:
    owned = True
    def __init__(self):
        self.running = False
        self.stops = 0
    def start(self):
        self.running = True
    def stop(self):
        self.stops += 1
        self.running = False
        return True
    def pids(self):
        return frozenset({123}) if self.running else frozenset()


class FakeHTTP:
    def __init__(self, runtime, *, digest=PIN):
        self.runtime, self.digest = runtime, digest
        self.fail_path = None
        self.exception = None
        self.calls = []
        self.cancelled = False
        self.timed_out = False

    def get(self, path):
        self.calls.append(path)
        if not self.runtime.running:
            raise ProbeFailure('unavailable')
        if self.exception and path == self.fail_path:
            raise self.exception
        if path == '/api/version':
            return {'version': 'fixture-1'}
        if path == '/api/tags':
            return {'models': [{'name': 'fixture:1', 'digest': self.digest,
                'details': {'format': 'gguf', 'quantization_level': 'Q4_K_M'}}]}
        if path == '/api/ps':
            return {'models': [{'name': 'fixture:1', 'digest': self.digest,
                'size_vram': 2_000_000, 'context_length': 4096}]}
        raise AssertionError(path)

    def post(self, path, body, *, timeout_s=None):
        self.calls.append(path)
        if self.exception and path == self.fail_path:
            raise self.exception
        if path == '/api/generate':
            return {'done': True}
        if path == '/api/chat':
            return {'done': True, 'model': 'fixture:1',
                'message': {'role': 'assistant', 'content': 'fixture reply'},
                'prompt_eval_count': 90, 'eval_count': 2}
        raise AssertionError(path)

    def cancel_stream(self, path, body):
        self.cancelled = True
        return True

    def expect_timeout(self, path, body):
        self.timed_out = True
        return True


class FakeNvidia:
    def __init__(self, correlated=True):
        self.result = correlated
    def correlated(self, pids):
        return bool(pids) and self.result


class OllamaQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = EvidenceStore(self.root / 'evidence')
        self.store = QualificationStore(TaskLedger(self.root / 'tasks.db'), self.evidence)
        self.runtime = FakeRuntime()
        self.http = FakeHTTP(self.runtime)
        self.nvidia = FakeNvidia()
        self.current = changed(identity(), artifact_sha256=PIN, quantization='Q4_K_M')
        self.config = OllamaQualificationConfig(
            profile=RuntimeProfile(runtime='ollama', base_url='http://127.0.0.1:12345',
                                   timeout_s=2, max_response_bytes=1024),
            model_reference='fixture:1', model_digest=PIN, quantization='Q4_K_M',
            context_tokens=4096, gpu_uuid='GPU-fixture')

    def probe(self, *, runtime=None, http=None, nvidia=None):
        factory = OllamaQualificationProbeFactory(self.config, self.evidence,
            runtime_factory=lambda _: runtime or self.runtime,
            http_factory=lambda _: http or self.http,
            nvidia_factory=lambda _: nvidia or self.nvidia)
        return factory(self.current)

    def test_all_eight_checks_pass_but_fixture_cannot_qualify(self):
        probe = self.probe()
        result = QualificationRunner(self.store).run(probe)
        self.assertEqual(probe.source, 'synthetic')
        self.assertFalse(result.interrupted)
        self.assertEqual({item.check for item in result.checks if item.outcome == 'passed'},
                         set(Check) - {Check.STRUCTURED_OUTPUT, Check.TOOL_CALLING})
        self.assertTrue(self.evidence.verify_record(result))
        self.assertEqual(self.store.status(self.current).state, QualificationState.UNKNOWN)
        self.assertEqual(self.runtime.stops, 1)
        self.assertTrue(self.http.cancelled and self.http.timed_out)

    def test_digest_mismatch_and_ambiguous_gpu_fail_closed(self):
        self.http.digest = 'b' * 64
        self.assertEqual(self.probe().run(Check.LOAD).failure, 'assertion_failed')
        self.http.digest = PIN
        self.nvidia.result = False
        self.assertEqual(self.probe().run(Check.LOAD).failure, 'assertion_failed')

    def test_external_runtime_is_never_stopped(self):
        external = ExternalOllamaRuntime()
        probe = self.probe(runtime=external)
        self.assertEqual(probe.run(Check.RUNTIME_FAILURE).failure, 'unavailable')
        self.assertEqual(probe.run(Check.RECOVERY).failure, 'unavailable')
        self.assertFalse(external.stop())

    def test_typed_timeout_unavailable_and_sanitized_error(self):
        for failure in (ProbeFailure('timeout'), ProbeFailure('unavailable'),
                        RuntimeError('private prompt and token marker')):
            with self.subTest(failure=type(failure).__name__):
                self.http.exception = failure
                self.http.fail_path = '/api/chat'
                self.runtime.start()
                result = self.probe().run(Check.GENERATION)
                self.assertEqual(result.outcome, 'failed')
                payload = (self.evidence.root / (result.evidence_sha256 + '.json')).read_text()
                self.assertNotIn('private prompt', payload)
                self.assertNotIn('token marker', payload)
        self.assertEqual(result.failure, 'probe_error')

    def test_runtime_unavailable_load(self):
        self.http.exception = ProbeFailure('unavailable')
        self.http.fail_path = '/api/version'
        self.assertEqual(self.probe().run(Check.LOAD).failure, 'unavailable')

    def test_identity_request_must_match_configuration(self):
        factory = OllamaQualificationProbeFactory(self.config, self.evidence,
            runtime_factory=lambda _: self.runtime)
        with self.assertRaises(ValueError):
            factory(changed(self.current, artifact_sha256='b' * 64))

    def test_endpoint_and_http_bounds(self):
        with self.assertRaises(ValueError):
            OllamaQualificationConfig(profile=RuntimeProfile(runtime='ollama',
                base_url='https://example.com', allow_remote=True),
                model_reference='fixture:1', model_digest=PIN, quantization='Q4_K_M',
                context_tokens=4096, gpu_uuid='GPU-fixture')
        oversized = OllamaQualificationHTTP(self.config,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b'x' * 2048)))
        with self.assertRaises(ProbeFailure) as error:
            oversized.get('/api/version')
        self.assertEqual(error.exception.reason, 'invalid_response')
        malformed = OllamaQualificationHTTP(self.config,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b'{bad')))
        with self.assertRaises(ProbeFailure) as error:
            malformed.get('/api/version')
        self.assertEqual(error.exception.reason, 'invalid_response')

    def test_proxy_and_redirects_disabled(self):
        http = OllamaQualificationHTTP(self.config)
        with patch('dante.ollama_qualification.httpx.Client') as client:
            http._client(1)
        self.assertFalse(client.call_args.kwargs['trust_env'])
        self.assertFalse(client.call_args.kwargs['follow_redirects'])


if __name__ == '__main__':
    unittest.main()
