from dante.contracts.runtime import RuntimeProfile
from dante.inference import PolicyDenied, ContextExceeded
from dante.local_runtime import OllamaAdapter

ENDPOINT = 'http://127.0.0.1:11434'
REF = 'dante-qwen-agent:latest'
RUNTIME = 'ollama'


class DevelopmentQwenAdapter(OllamaAdapter):
    def __init__(self, profile=None, *, machine_profile='default', transport=None, registry=None):
        if profile is None:
            profile = RuntimeProfile(runtime=RUNTIME, base_url=ENDPOINT, timeout_s=120)
        else:
            if profile.runtime != RUNTIME or profile.base_url != ENDPOINT or profile.timeout_s > 120:
                raise PolicyDenied('invalid profile')
        super().__init__(profile, machine_profile=machine_profile, transport=transport, registry=registry, provider_id='dev-qwen')

    def _qualified(self, request):
        metadata = request.model.local_metadata
        if (not request.model.local or request.model.runtime != RUNTIME
                or request.model.provider_id != 'dev-qwen' or metadata is None):
            raise PolicyDenied('policy denied')
        if metadata.runtime_reference != REF:
            raise PolicyDenied('policy denied')
        if metadata.context_tokens is None:
            raise ContextExceeded('context is unknown')
        if request.max_output_tokens is not None and request.max_output_tokens > min(2048, metadata.context_tokens):
            raise ContextExceeded('context exceeded')
        return REF

