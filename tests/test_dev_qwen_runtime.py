import unittest
from types import SimpleNamespace

from dante.contracts.runtime import RuntimeProfile
from dante.inference import PolicyDenied, ContextExceeded
from dante.dev_qwen_runtime import DevelopmentQwenAdapter


def model(ref='dante-qwen-agent:latest', provider='dev-qwen', local=True):
    return SimpleNamespace(
        local=local,
        runtime='ollama',
        provider_id=provider,
        local_metadata=SimpleNamespace(runtime_reference=ref, context_tokens=4096),
    )


def request(m, tokens=128):
    return SimpleNamespace(model=m, max_output_tokens=tokens)


class TestDevQwenRuntime(unittest.TestCase):
    def test_default_profile(self):
        adapter = DevelopmentQwenAdapter()
        self.assertEqual(adapter.profile.base_url, 'http://127.0.0.1:11434')
        self.assertEqual(adapter.provider_id, 'dev-qwen')
        self.assertEqual(adapter.profile.timeout_s, 120)

    def test_valid_qualified(self):
        adapter = DevelopmentQwenAdapter()
        m = model()
        self.assertEqual(adapter._qualified(request(m)), m.local_metadata.runtime_reference)

    def test_rejects_invalid(self):
        adapter = DevelopmentQwenAdapter()
        with self.assertRaises(PolicyDenied):
            adapter._qualified(request(model(ref='other:latest')))
        with self.assertRaises(PolicyDenied):
            adapter._qualified(request(model(provider='other')))
        with self.assertRaises(PolicyDenied):
            adapter._qualified(request(model(local=False)))

    def test_constructor_rejects_invalid_profile(self):
        with self.assertRaises(PolicyDenied):
            DevelopmentQwenAdapter(RuntimeProfile(runtime='ollama', base_url='http://127.0.0.1:11435', timeout_s=120))
        with self.assertRaises(PolicyDenied):
            DevelopmentQwenAdapter(RuntimeProfile(base_url='http://127.0.0.1:11434', timeout_s=120, runtime='llamacpp'))
        with self.assertRaises(PolicyDenied):
            DevelopmentQwenAdapter(RuntimeProfile(runtime='ollama', base_url='http://127.0.0.1:11434', timeout_s=121))

    def test_context_exceeded(self):
        adapter = DevelopmentQwenAdapter()
        with self.assertRaises(ContextExceeded):
            adapter._qualified(request(model(), 2049))


if __name__ == '__main__':
    unittest.main()
