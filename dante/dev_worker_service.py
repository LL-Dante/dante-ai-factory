"""Self-hosted Qwen worker service backed by Dante's existing durable workload queue."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
import re

from dante.acceptance import AcceptanceContract
from dante.agent_runner import AgentRegistry, AgentRunner, Node0WorkloadExecutor, definition_digest
from dante.contracts import CostClass, LifecycleState, ModelRef, PrivacyClass, Task, TaskStatus
from dante.contracts.agents import AgentDefinition, AgentModelTarget, AgentRetryPolicy, AgentTaskPayload
from dante.contracts.runtime import LocalModelMetadata
from dante.dev_agent_loop import DevelopmentAgentLoop
from dante.dev_qwen_runtime import ENDPOINT, REF, DevelopmentQwenAdapter
from dante.dev_worker_edit_tools import register_edit_tools
from dante.dev_worker_tools import register_coding_tools
from dante.dev_worker_test_tools import register_test_tool
from dante.inference import InferenceCancelled, InferenceError, PolicyDenied
from dante.ledger import TaskLedger
from dante.tool_broker import ToolBroker
from dante.workload import ExecutionResult, ModelRequirement, WorkloadOrchestrator, WorkloadSpec, WorkloadStore


def discover_development_qwen(adapter: DevelopmentQwenAdapter, *, context_tokens: int = 8192) -> ModelRef:
    """Read-only runtime identity probe; qualification and license remain unknown."""
    if adapter.profile.base_url != ENDPOINT or adapter.provider_id != 'dev-qwen':
        raise PolicyDenied('Development adapter endpoint/provider is not pinned')
    if type(context_tokens) is not int or not 1024 <= context_tokens <= 8192:
        raise ValueError('Configured development context must be between 1024 and 8192 tokens')
    matches = [item for item in adapter.inventory() if item.get('id') == REF]
    if len(matches) != 1:
        raise PolicyDenied('Pinned development model is missing or ambiguous')
    entry = matches[0]
    digest = entry.get('digest')
    if (not isinstance(digest, str) or not re.fullmatch(r'(?:sha256:)?[0-9a-fA-F]{64}', digest)
            or not isinstance(entry.get('details'), dict)
            or entry['details'].get('format') != 'gguf'):
        raise PolicyDenied('Development model identity is not a local GGUF with a SHA-256 digest')
    shown = adapter.show(REF)
    details, model_info = shown.get('details'), shown.get('model_info')
    if not isinstance(details, dict) or details.get('format') != 'gguf' or not isinstance(model_info, dict):
        raise PolicyDenied('Development model metadata is incomplete')
    caps = shown.get('capabilities')
    caps = tuple(sorted(value for value in caps if isinstance(value, str))) if isinstance(caps, list) else ()
    metadata = LocalModelMetadata(runtime_reference=REF, runtime_digest=digest, format='gguf',
        quantization=details.get('quantization_level'), context_tokens=context_tokens,
        tool_use='tools' in caps, license_status='unknown', qualification='unverified',
        identity_verified=False, identity_limitations=(
            'Runtime tag digest and GGUF metadata observed; artifact file hash and production qualification unknown.',))
    return ModelRef(model_id=REF, provider_id='dev-qwen', logical_alias='dante-qwen-agent',
        version=digest.removeprefix('sha256:')[:12], runtime='ollama', local=True,
        capabilities={'tool_calling'} if 'tools' in caps else set(), cost_class=CostClass.LOCAL_COMPUTE,
        lifecycle=LifecycleState.CANDIDATE,
        privacy_eligibility={PrivacyClass.PUBLIC, PrivacyClass.INTERNAL, PrivacyClass.CONFIDENTIAL},
        local_metadata=metadata)


class DevelopmentLimitReached(InferenceError):
    failure_code = 'LIMIT_REACHED'


class DevelopmentAgentStuck(InferenceError):
    failure_code = 'AGENT_STUCK'


class _CodingAgent:
    def __init__(self, *, adapter, broker, ledger, model, workspace: Path, tool_ids):
        self.adapter, self.broker, self.ledger = adapter, broker, ledger
        self.model, self.workspace, self.tool_ids = model, workspace, tool_ids

    def execute(self, _inference, job, _definition, payload: AgentTaskPayload,
                spec: WorkloadSpec, cancellation, emit) -> ExecutionResult:
        root = Path(spec.metadata.get('workspace', '')).resolve(strict=True)
        if root != self.workspace or root.is_symlink():
            raise PolicyDenied('Job workspace differs from the configured development sandbox')
        task = Task(task_id=str(job.job_id), goal=payload.objective, workspace=str(root))
        self.ledger.create_task(task, AcceptanceContract())
        self.ledger.transition(task.task_id, TaskStatus.PLANNED)
        self.ledger.transition(task.task_id, TaskStatus.RUNNING)
        objective = payload.objective
        if payload.context:
            objective += '\n\nContext:\n' + payload.context
        try:
            result = DevelopmentAgentLoop(self.adapter, self.broker, task, self.model,
                self.tool_ids, emit=emit, max_steps=8, max_tool_calls=8,
                max_wall_s=600, max_output_tokens=384).run(objective, cancellation)
            if result.status == 'CANCELLED':
                self.ledger.transition(task.task_id, TaskStatus.CANCELLED)
                raise InferenceCancelled('Development job cancelled')
            if result.status == 'AGENT_STUCK':
                self.ledger.transition(task.task_id, TaskStatus.FAILED_TERMINAL)
                raise DevelopmentAgentStuck('Development agent repeated an unchanged read-only request')
            if result.status != 'DONE':
                self.ledger.transition(task.task_id, TaskStatus.FAILED_TERMINAL)
                raise DevelopmentLimitReached('Development agent reached a bounded limit')
            self.ledger.transition(task.task_id, TaskStatus.VERIFYING)
            completed = self.ledger.transition(task.task_id, TaskStatus.COMPLETED)
            if completed.status != TaskStatus.COMPLETED:
                raise DevelopmentLimitReached('Tool evidence did not satisfy task acceptance')
            return ExecutionResult(text=result.summary, model_id=self.model.model_id,
                provider_id=self.model.provider_id, fallback=False, runtime='ollama', metadata={
                    'worker_status': result.status, 'model_calls': result.model_calls,
                    'tool_calls': result.tool_calls, 'elapsed_wall_s': result.elapsed_s,
                    'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens,
                    'ttft_s': None, 'output_tokens_per_s': None,
                    'endpoint': ENDPOINT, 'local': True,
                })
        except Exception:
            current = self.ledger.get_task(task.task_id)
            if current.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED_TERMINAL,
                                      TaskStatus.FAILED_RETRYABLE, TaskStatus.CANCELLED}:
                self.ledger.transition(task.task_id, TaskStatus.FAILED_TERMINAL)
            raise


class _DevelopmentExecutor:
    """Only registered AGENT_TASK jobs are accepted on this local worker."""
    def __init__(self, adapter, runner):
        self._dispatch = Node0WorkloadExecutor(adapter, runner)

    def execute(self, spec, cancellation):
        raise PolicyDenied('Development worker accepts registered agent jobs only')

    def execute_agent(self, job, spec, cancellation):
        return self._dispatch.execute_agent(job, spec, cancellation)


class DevelopmentWorkerService:
    """Single-slot local Qwen worker. All filesystem tools stay within one root."""
    AGENT_ID = 'local-coding-worker'

    def __init__(self, db_path: Path, adapter: DevelopmentQwenAdapter, model,
                 workspace_root: Path):
        self.db_path = Path(db_path).resolve()
        protected = Path('C:/DanteAI').resolve()
        if self.db_path == protected or protected in self.db_path.parents:
            raise ValueError('Development worker database cannot be inside protected B2.2 worktree')
        requested_root = Path(workspace_root).absolute()
        if any(_is_link(parent) for parent in (requested_root, *requested_root.parents)):
            raise ValueError('Worker root or one of its parents cannot be a link or junction')
        self.workspace_root = requested_root.resolve(strict=True)
        if not self.workspace_root.is_dir() or _is_link(self.workspace_root):
            raise ValueError('Worker root must be an existing non-link directory')
        if protected == self.workspace_root or protected in self.workspace_root.parents:
            raise ValueError('Development worker workspace cannot be inside protected B2.2 worktree')
        if (self.db_path == self.workspace_root or self.db_path in self.workspace_root.parents
                or self.workspace_root in self.db_path.parents):
            raise ValueError('Worker database and editable workspace must not overlap')
        self.adapter, self.model = adapter, model
        metadata = getattr(model, 'local_metadata', None)
        if (adapter.profile.base_url != ENDPOINT or adapter.provider_id != 'dev-qwen'
                or not getattr(model, 'local', False) or model.runtime != 'ollama'
                or model.provider_id != 'dev-qwen' or metadata is None
                or metadata.runtime_reference != REF or not metadata.runtime_digest
                or not metadata.context_tokens):
            raise PolicyDenied('Development model identity is incomplete or not pinned')

        self.store = WorkloadStore(self.db_path)
        self.ledger = TaskLedger(self.db_path)
        self.broker = ToolBroker(ledger=self.ledger)
        tool_ids = register_coding_tools(self.broker, str(self.workspace_root))
        tool_ids += register_edit_tools(self.broker, str(self.workspace_root))
        tool_ids += register_test_tool(self.broker, str(self.workspace_root))
        self.tool_ids = tuple(tool_ids)

        target = AgentModelTarget(model_id=model.model_id, runtime_reference=REF,
            digest_sha256=metadata.runtime_digest, context_tokens=metadata.context_tokens,
            local_only=True)
        self.definition = AgentDefinition(agent_id=self.AGENT_ID,
            name='Dante Local Coding Worker', role='Bounded local workspace coding',
            instructions=('Use only the registered workspace tools. Never request shell, network, '
                          'secrets, arbitrary commands, or changed permissions. RUN_TESTS only '
                          'accepts one workspace-relative project tests/test_*.py target and is not an OS security sandbox.'),
            capabilities=('LOCAL_INFERENCE',), model_target=target,
            output_token_budget=384, timeout_s=600,
            retry_policy=AgentRetryPolicy(maximum_attempts=1), version='1.0.0')
        registry = AgentRegistry()
        registry.register(self.definition, _CodingAgent(adapter=adapter, broker=self.broker,
            ledger=self.ledger, model=model, workspace=self.workspace_root, tool_ids=self.tool_ids))
        runner = AgentRunner(registry, adapter, self.store)
        self.orchestrator = WorkloadOrchestrator(self.store,
            _DevelopmentExecutor(adapter, runner), max_gpu_jobs=1)
        self._serve_thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

    def start(self) -> None:
        with self._thread_lock:
            if self._serve_thread is not None and self._serve_thread.is_alive():
                return
            if self.orchestrator.state != 'RUNNING':
                self.orchestrator.start()
            self._serve_thread = threading.Thread(target=self.orchestrator.serve,
                name='dante-dev-qwen-worker', daemon=True)
            self._serve_thread.start()

    def submit(self, objective: str, context: str | None = None,
               idempotency_key: str | None = None):
        if not isinstance(objective, str) or not objective.strip() or len(objective) > 4000:
            raise ValueError('Objective must contain 1 to 4000 characters')
        if context is not None and (not context.strip() or len(context) > 4000
                                    or len(objective) + len(context) + 10 > 4000):
            raise ValueError('Context exceeds the bounded task input')
        self.start()
        metadata = self.model.local_metadata
        payload = AgentTaskPayload(agent_id=self.AGENT_ID, objective=objective,
            context=context, requested_output_tokens=384, definition_version=self.definition.version,
            definition_digest=definition_digest(self.definition))
        spec = WorkloadSpec(job_type='AGENT_TASK',
            model=ModelRequirement(model_id=self.model.model_id, runtime_reference=REF,
                digest_sha256=metadata.runtime_digest, context_tokens=metadata.context_tokens,
                local_only=True),
            prompt='Run the bounded local coding task using approved workspace tools.',
            agent_task=payload, max_output_tokens=384, timeout_s=600,
            maximum_attempts=1, metadata={'workspace': str(self.workspace_root)})
        return self.store.submit(spec, idempotency_key=idempotency_key)

    def get(self, job_id: str):
        return self.store.get(job_id)

    def list_jobs(self, limit: int = 100):
        return self.store.list(limit=limit)

    def events(self, job_id: str):
        return self.store.events(job_id)

    def queue_position(self, job_id: str) -> int | None:
        job = self.store.get(job_id)
        if job.state.value != 'queued':
            return None
        queued = [item for item in self.store.list(limit=500) if item.state.value == 'queued']
        queued.sort(key=lambda item: (-item.priority, item.created_at, str(item.job_id)))
        try:
            return [item.job_id for item in queued].index(job.job_id) + 1
        except ValueError:
            return None

    def queue_telemetry(self, job_id: str, *, now: datetime | None = None) -> dict:
        """Expose queue and slot facts supported by the existing durable store."""
        job = self.store.get(job_id)
        records = self.store.list(limit=500)
        state = job.state.value
        queued = [item for item in records if item.state.value == 'queued']
        depth = len(queued) if len(records) < 500 else None
        position = self.queue_position(job_id)
        events = self.store.events(job_id)
        submitted = next((event for event in events if event.get('event') == 'submitted'), None)
        queued_at = (submitted.get('timestamp_utc') if submitted else
                     (job.created_at.isoformat() if job.created_at else None))
        queued_source = 'submitted event' if submitted else ('JobRecord.created_at' if queued_at else None)

        attempt_id = str(job.current_attempt_id) if job.current_attempt_id else None
        acquired = next((event for event in reversed(events)
            if event.get('event') == 'attempt_started' and event.get('attempt_id') == attempt_id), None)
        slot_acquired_at = acquired.get('timestamp_utc') if acquired else None

        def has_acquired_attempt(candidate, candidate_events):
            current = str(candidate.current_attempt_id) if candidate.current_attempt_id else None
            return bool(current and any(event.get('event') == 'attempt_started'
                and event.get('attempt_id') == current for event in candidate_events))

        owners = []
        for candidate in records:
            if candidate.state.value not in {'running', 'cancel_requested'}:
                continue
            candidate_events = self.store.events(str(candidate.job_id))
            if has_acquired_attempt(candidate, candidate_events):
                claimed = next((event for event in reversed(candidate_events)
                    if event.get('event') == 'claimed'
                    and event.get('attempt_id') == str(candidate.current_attempt_id)), None)
                worker_id = ((claimed or {}).get('metadata') or {}).get('worker_id')
                owners.append((str(candidate.job_id), worker_id))
        owner_reason = None
        if len(owners) == 1:
            owner_job_id, owner_worker_id = owners[0]
        else:
            owner_job_id = owner_worker_id = None
            owner_reason = ('multiple active slot owners found in durable state' if owners else
                            'no active attempt_started event found')

        def parse_stamp(value):
            try:
                stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
            except (TypeError, ValueError):
                return None

        queued_dt = parse_stamp(queued_at)
        acquired_dt = parse_stamp(slot_acquired_at)
        as_of = now or datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            raise ValueError('now must be timezone-aware')
        if queued_dt and acquired_dt:
            queue_wait_ms = max(0, round((acquired_dt - queued_dt).total_seconds() * 1000))
            wait_reason = None
        elif state == 'queued' and queued_dt:
            queue_wait_ms = max(0, round((as_of.astimezone(timezone.utc) - queued_dt).total_seconds() * 1000))
            wait_reason = None
        else:
            queue_wait_ms = None
            wait_reason = 'slot acquisition or queued state is not evidenced'

        unknown = {}
        if depth is None:
            unknown['queue_depth'] = 'store listing reached its 500-record limit'
        if state == 'queued' and position is None:
            unknown['position'] = 'queued job is outside the available queue listing'
        if slot_acquired_at is None:
            unknown['slot_acquired_at'] = 'no attempt_started event for the current attempt'
        if owner_reason:
            unknown['slot_owner'] = owner_reason
        if wait_reason:
            unknown['queue_wait_ms'] = wait_reason
        unknown['slot_released_at'] = 'no durable slot-release event is recorded'
        return {
            'capacity': self.orchestrator.resource_gate.max_gpu_jobs,
            'queue_depth': depth, 'position': position, 'priority': job.priority,
            'queued_at': queued_at, 'queued_at_source': queued_source,
            'slot_acquired_at': slot_acquired_at,
            'slot_released_at': None,
            'slot_released_reason': 'no durable slot-release event is recorded',
            'queue_wait_ms': queue_wait_ms,
            'slot_owner_job_id': owner_job_id, 'slot_owner_worker_id': owner_worker_id,
            'unknown_reasons': unknown,
        }

    def cancel(self, job_id: str):
        if self.orchestrator.state == 'RUNNING':
            return self.orchestrator.cancel(job_id)
        return self.store.request_cancel(job_id)

    def close(self, drain_timeout_s: float = 65) -> None:
        self.orchestrator.request_stop()
        thread = self._serve_thread
        if thread is not None:
            thread.join(timeout=drain_timeout_s)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())
