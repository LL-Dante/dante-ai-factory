import unittest
from types import SimpleNamespace
import httpx
from dante.dev_qwen_runtime import DevelopmentQwenAdapter
from dante.inference import PolicyDenied

REF = 'dante-qwen-agent:latest'
ACTUAL = 'a' * 64
EXPECTED = 'b' * 64
PATHS = []

def handler(request: httpx.Request) -> httpx.Response:
    PATHS.append(request.url.path)
    if request.url.path == '/api/tags':
        return httpx.Response(200, json={'models': [{'name': REF, 'digest': ACTUAL, 'details': {'format': 'gguf'}}]})
    return httpx.Response(404)

class TestDigestEnforcement(unittest.TestCase):
    def test_rejects_digest_mismatch(self):
        model = SimpleNamespace(local=True, runtime='ollama', provider_id='dev-qwen',
            local_metadata=SimpleNamespace(runtime_reference=REF, runtime_digest=EXPECTED, context_tokens=4096))
        request = SimpleNamespace(model=model, max_output_tokens=8,
            messages=({'role': 'user', 'content': 'x'},), temperature=0,
            thinking='off', tools=(), output_schema=None)
        adapter = DevelopmentQwenAdapter(transport=httpx.MockTransport(handler))
        with self.assertRaises(PolicyDenied):
            adapter.complete(request)
        self.assertEqual(PATHS, ['/api/tags'])

if __name__ == '__main__':
    unittest.main()


