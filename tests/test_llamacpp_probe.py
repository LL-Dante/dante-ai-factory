"""llama.cpp discovery uses only deterministic HTTP fixtures."""
import unittest

import httpx

from dante.contracts.runtime import RuntimeProfile
from dante.inference import RateLimitUnavailable
from dante.llamacpp_probe import probe_llamacpp


class LlamaCppWire:
    def __init__(self, *, health='ok', models=None, properties=None, props_status=200, error=None):
        self.health = health
        self.models = [] if models is None else models
        self.properties = {} if properties is None else properties
        self.props_status = props_status
        self.error = error
        self.paths = []

    def __call__(self, request):
        self.paths.append(request.url.path)
        if self.error:
            raise self.error
        if request.url.path == '/health':
            return httpx.Response(200, json={'status': self.health})
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': self.models})
        if request.url.path == '/props':
            return httpx.Response(self.props_status, json=self.properties)
        return httpx.Response(404)


class LlamaCppProbeTests(unittest.TestCase):
    def probe(self, wire, profile=None, **kwargs):
        return probe_llamacpp(profile, transport=httpx.MockTransport(wire), **kwargs)

    def test_available_build_endpoint_state_and_inventory(self):
        wire = LlamaCppWire(models=[{'id': 'zeta.gguf'}, {'id': 'alpha.gguf'}],
            properties={'version': 'b6012', 'build_info': 'release b6012', 'backend': 'cuda'})
        result = self.probe(wire)
        self.assertEqual(result.state, 'available')
        self.assertEqual(result.endpoint, 'http://127.0.0.1:8080')
        self.assertEqual(result.server_state, 'ready')
        self.assertEqual(result.observation.version, 'b6012')
        self.assertEqual(result.build, 'release b6012')
        self.assertEqual(result.observation.backend, 'cuda')
        self.assertEqual([model.runtime_reference for model in result.models], ['alpha.gguf', 'zeta.gguf'])
        self.assertEqual(wire.paths, ['/health', '/v1/models', '/props'])

    def test_properties_endpoint_is_optional(self):
        result = self.probe(LlamaCppWire(props_status=404))
        self.assertEqual(result.state, 'available')
        self.assertIsNone(result.observation.version)
        self.assertIsNone(result.observation.backend)

    def test_explicit_endpoint_and_cpu_backend_are_recorded(self):
        profile = RuntimeProfile(runtime='llamacpp', base_url='http://localhost:22001', timeout_s=2,
                                 max_response_bytes=4096)
        result = self.probe(LlamaCppWire(props_status=404), profile, backend='cpu')
        self.assertEqual(result.endpoint, 'http://127.0.0.1:22001')
        self.assertEqual(result.observation.backend, 'cpu')
        self.assertEqual(len(result.observation.configuration_sha256), 64)

    def test_vulkan_is_represented_without_preference(self):
        result = self.probe(LlamaCppWire(properties={'backend': 'vulkan'}))
        self.assertEqual(result.observation.backend, 'vulkan')
        self.assertNotIn('cuda', result.observation.model_dump_json())

    def test_unavailable_and_timeout_are_clean(self):
        for error, failure in ((httpx.ConnectError('fixture'), 'unavailable'),
                               (httpx.ReadTimeout('fixture'), 'timeout')):
            with self.subTest(failure=failure):
                result = self.probe(LlamaCppWire(error=error))
                self.assertEqual((result.state, result.failure), ('unavailable', failure))
                self.assertIsNone(result.observation)

    def test_capacity_failure_is_not_misclassified_as_transport_unavailable(self):
        result = self.probe(LlamaCppWire(error=RateLimitUnavailable('fixture')))
        self.assertEqual((result.state, result.failure), ('error', 'capacity_unavailable'))

    def test_not_ready_is_unavailable(self):
        result = self.probe(LlamaCppWire(health='loading model'))
        self.assertEqual((result.state, result.failure), ('unavailable', 'unavailable'))

    def test_malformed_model_and_properties_are_errors(self):
        fixtures = (LlamaCppWire(models=[{'id': 'bad\nreference'}]),
                    LlamaCppWire(models=[{'id': 'valid.gguf'}, {}]),
                    LlamaCppWire(properties=[]),
                    LlamaCppWire(properties={'backend': 'metal'}),
                    LlamaCppWire(properties={'version': {'bad': True}}))
        for wire in fixtures:
            with self.subTest(paths=wire.paths):
                result = self.probe(wire)
                self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_empty_or_malformed_json_is_error(self):
        for content in (b'', b'{bad json'):
            with self.subTest(content=content):
                transport = httpx.MockTransport(
                    lambda _request, content=content: httpx.Response(200, content=content))
                result = probe_llamacpp(transport=transport)
                self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_configured_and_reported_backend_must_match(self):
        result = self.probe(LlamaCppWire(properties={'backend': 'cuda'}), backend='vulkan')
        self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_duplicate_model_reference_is_invalid(self):
        result = self.probe(LlamaCppWire(models=[{'id': 'duplicate.gguf'}, {'id': 'duplicate.gguf'}]))
        self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_profile_mismatch_rejected_before_transport(self):
        profile = RuntimeProfile(runtime='ollama', base_url='http://127.0.0.1:11434')
        with self.assertRaisesRegex(ValueError, 'llama.cpp runtime profile required'):
            self.probe(LlamaCppWire(), profile)

    def test_unknown_configured_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported llama.cpp backend'):
            self.probe(LlamaCppWire(), backend='metal')


if __name__ == '__main__':
    unittest.main()
