"""Offline adapters and crash worker shared by P6 tests."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dante.agent_host import AgentHostFoundation
from dante.contracts import ModelRef, LifecycleState, CostClass, PrivacyClass
from dante.contracts.inference import InferenceResponse, AssistantMessage
from dante.contracts.continuity import ContinuityPolicy, BackendState
from dante.continuity import ContinuityManager
from dante.inference import InferenceGateway, QuotaExhausted, AdapterUnavailable
from dante.ledger import TaskLedger
from dante.privacy import PrivacyGate
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter
from dante.worker import Worker
from test_local_runtime import model
from worker_process import host_factory as tools_host


def models():
    cloud = ModelRef(model_id='cloud', provider_id='fake-cloud', logical_alias='cloud', version='1', runtime='fixture',
        cost_class=CostClass.ZERO, cost_verification='VERIFIED_ZERO', context_tokens=4096,
        lifecycle=LifecycleState.APPROVED, privacy_eligibility=set(PrivacyClass), capabilities={'tool_calling'})
    return [cloud, model(provider_id='fake-local')]


class FakeAdapter:
    automatic_cost = 0
    def __init__(self, provider_id, error=None):
        self.provider_id, self.error, self.calls = provider_id, error, []
    def complete(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return InferenceResponse(model=request.model, assistant_message=AssistantMessage(content='fixture'), finish_reason='stop')


def host_factory(ledger, task):
    mode = os.environ.get('DANTE_CONTINUITY_FIXTURE', '')
    registry = ModelRegistry(models(), machine_profile='old-pc')
    adapters = [FakeAdapter('fake-cloud', AdapterUnavailable() if mode == 'defer' else QuotaExhausted()),
                FakeAdapter('fake-local')]
    router = RuleBasedRouter(registry)
    gateway = InferenceGateway(registry, adapters)
    manager = ContinuityManager(ledger, router, gateway, ContinuityPolicy(mode='CLOUD_FIRST_WITH_LOCAL_FALLBACK'))
    for backend in ('fake-cloud', 'fake-local'):
        if manager.observation(backend) is None or mode == 'restore':
            manager.observe(backend, BackendState.UNAVAILABLE if mode == 'defer' and backend == 'fake-local'
                            else BackendState.AVAILABLE, task=task)
    host = AgentHostFoundation(ledger, PrivacyGate(), router, gateway, tools_host(ledger, task).tools, continuity=manager)
    host.infer(task.task_id, 'private fixture prompt', {'tool_calling'}, required_context_tokens=1024)
    return host


class CrashLedger(TaskLedger):
    def checkpoint(self, *args, **kwargs):
        result = super().checkpoint(*args, **kwargs)
        if os.environ.get('DANTE_CONTINUITY_FIXTURE') == 'checkpoint':
            os._exit(73)
        return result


if __name__ == '__main__':
    Worker(CrashLedger(Path(sys.argv[1])), host_factory, lease_s=30).run_once()
