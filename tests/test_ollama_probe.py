"""Ollama discovery uses only deterministic HTTP fixtures."""
import unittest

import httpx

from dante.contracts.runtime import RuntimeProfile
from dante.inference import RateLimitUnavailable
from dante.ollama_probe import probe_ollama


PIN = 'a' * 64


class OllamaWire:
    def __init__(self, *, version='0.12.3', models=None, error=None):
        self.version = version
        self.models = [] if models is None else models
        self.error = error
        self.paths = []

    def __call__(self, request):
        self.paths.append(request.url.path)
        if self.error:
            raise self.error
        if request.url.path == '/api/version':
            return httpx.Response(200, json={'version': self.version})
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': self.models})
        return httpx.Response(404)


class OllamaProbeTests(unittest.TestCase):
    def probe(self, wire, profile=None):
        return probe_ollama(profile, transport=httpx.MockTransport(wire))

    def test_available_version_endpoint_and_inventory(self):
        wire = OllamaWire(models=[
            {'name': 'zeta:latest', 'digest': PIN,
             'details': {'format': 'gguf', 'quantization_level': 'Q4_K_M'}},
            {'name': 'alpha:1', 'digest': 'sha256:' + ('b' * 64), 'details': {'format': 'gguf'}},
        ])
        result = self.probe(wire)
        self.assertEqual(result.state, 'available')
        self.assertEqual(result.endpoint, 'http://127.0.0.1:11434')
        self.assertEqual(result.observation.version, '0.12.3')
        self.assertEqual([model.runtime_reference for model in result.models], ['alpha:1', 'zeta:latest'])
        self.assertEqual(result.models[1].quantization, 'Q4_K_M')
        self.assertEqual(len(result.observation.model_inventory), 2)
        self.assertEqual(wire.paths, ['/api/version', '/api/tags'])

    def test_empty_inventory_is_available_without_qualification_claim(self):
        result = self.probe(OllamaWire())
        self.assertEqual(result.state, 'available')
        self.assertEqual(result.models, ())
        self.assertEqual(result.observation.capabilities, ())
        self.assertIsNone(result.observation.backend)

    def test_explicit_endpoint_and_configuration_are_recorded(self):
        profile = RuntimeProfile(runtime='ollama', base_url='http://localhost:22000', timeout_s=2,
                                 max_response_bytes=4096)
        result = self.probe(OllamaWire(), profile)
        self.assertEqual(result.endpoint, 'http://127.0.0.1:22000')
        self.assertEqual(len(result.observation.configuration_sha256), 64)

    def test_unavailable_is_clean(self):
        result = self.probe(OllamaWire(error=httpx.ConnectError('fixture')))
        self.assertEqual((result.state, result.failure), ('unavailable', 'unavailable'))
        self.assertIsNone(result.observation)

    def test_timeout_is_clean(self):
        result = self.probe(OllamaWire(error=httpx.ReadTimeout('fixture')))
        self.assertEqual((result.state, result.failure), ('unavailable', 'timeout'))

    def test_capacity_failure_is_not_misclassified_as_transport_unavailable(self):
        result = self.probe(OllamaWire(error=RateLimitUnavailable('fixture')))
        self.assertEqual((result.state, result.failure), ('error', 'capacity_unavailable'))

    def test_malformed_version_is_error(self):
        result = self.probe(OllamaWire(version=''))
        self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_malformed_inventory_is_error(self):
        result = self.probe(OllamaWire(models=[{'name': 'bad\nreference'}]))
        self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_invalid_digest_is_error(self):
        result = self.probe(OllamaWire(models=[{'name': 'model:1', 'digest': 'not-a-digest'}]))
        self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_duplicate_model_reference_is_invalid(self):
        model = {'name': 'duplicate:1', 'digest': PIN, 'details': {'format': 'gguf'}}
        result = self.probe(OllamaWire(models=[model, model]))
        self.assertEqual((result.state, result.failure), ('error', 'invalid_response'))

    def test_profile_mismatch_is_rejected_before_transport(self):
        profile = RuntimeProfile(runtime='llamacpp', base_url='http://127.0.0.1:8080')
        with self.assertRaisesRegex(ValueError, 'Ollama runtime profile required'):
            self.probe(OllamaWire(), profile)


if __name__ == '__main__':
    unittest.main()
