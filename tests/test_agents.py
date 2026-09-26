from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

from dante.agent_runner import (AgentRegistry, AgentRunner,
    AgentUnavailable, DanteResearchAgent, Node0WorkloadExecutor, build_agent_registry,
    definition_digest)
from dante.contracts.agents import (AgentDefinition, AgentModelTarget, AgentTaskPayload,
                                    ResearchDraft)
from dante.inference import InferenceCancelled, InferenceTimeout
from dante.node0_control import (ControlError, Node0ControlClient, Node0ControlServer,
    Node0ControlService)
from dante.node0_cli import main as node0_cli_main
from dante.telemetry import JsonlAudit
from dante.workload import (ExecutionResult, JobState, ModelRequirement,
    WorkloadOrchestrator, WorkloadSpec, WorkloadStore)
from pydantic import ValidationError


DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'
MODEL_ID = 'qwen3-4b-node0'
RUNTIME = 'qwen3:4b'
GPU = 'GPU-fixture'
QUALIFICATION = 'qualification-fixture'
VALID_DRAFT = json.dumps({
    'summary': 'Three priorities are clear.',
    'findings': ['Keep qualification authoritative.'],
    'recommended_next_actions': ['Add agent history.'],
    'limitations': ['No external research was performed.'],
})


def model_target():
    return AgentModelTarget(model_id=MODEL_ID, runtime_reference=RUNTIME,
        digest_sha256=DIGEST, context_tokens=4096)


def agent_spec(definition, *, objective='Analyze the local architecture.', context=None,
               requested_tokens=128, **changes):
    payload = AgentTaskPayload(agent_id=definition.agent_id, objective=objective, context=context,
        requested_output_tokens=requested_tokens, definition_version=definition.version,
        definition_digest=definition_digest(definition))
    data = dict(job_type='AGENT_TASK', model=ModelRequirement(model_id=MODEL_ID,
        runtime_reference=RUNTIME, digest_sha256=DIGEST, context_tokens=4096),
        prompt='registered-agent-task:' + definition.agent_id, agent_task=payload,
        max_output_tokens=requested_tokens, thinking=definition.thinking,
        timeout_s=definition.timeout_s, maximum_attempts=definition.retry_policy.maximum_attempts,
        retry_base_s=definition.retry_policy.retry_base_s,
        retry_max_s=definition.retry_policy.retry_max_s)
    data.update(changes)
    return WorkloadSpec(**data)


class FakeLocalInference:
    def __init__(self, *, response=VALID_DRAFT, error=None, cancellation_wait=False):
        self.response, self.error, self.cancellation_wait = response, error, cancellation_wait
        self.calls = []
        self.started = threading.Event()

    def execute(self, spec, cancellation):
        self.calls.append(spec)
        self.started.set()
        if self.cancellation_wait:
            cancellation.wait(2)
            raise InferenceCancelled('cancelled')
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return ExecutionResult(self.response, MODEL_ID, 'ollama', False, 'ollama', {
            'qualification_id': QUALIFICATION, 'gpu_identity': GPU, 'gpu_vram_bytes': 1024,
            'runtime_version': '0.34.4', 'context_tokens': 4096,
        })


class AgentFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'workloads.db'
        self.store = WorkloadStore(self.db)
        self.registry = build_agent_registry(model_target())
        self.definition = self.registry.describe('dante-research')
        self.local = FakeLocalInference()
        self.runner = AgentRunner(self.registry, self.local, self.store)
        self.executor = Node0WorkloadExecutor(self.local, self.runner)

    def tearDown(self):
        self.temp.cleanup()

    def orchestrator(self, *, executor=None):
        return WorkloadOrchestrator(self.store, executor or self.executor, poll_s=0.05,
                                    max_gpu_jobs=1, worker_id='agent-test')

    def submit(self, **changes):
        spec = agent_spec(self.definition, **changes)
        return self.store.submit(spec)


class AgentContractTests(AgentFixture):
    def test_definition_serializes_and_exposes_only_local_inference(self):
        restored = AgentDefinition.model_validate_json(self.definition.model_dump_json())
        self.assertEqual(restored, self.definition)
        self.assertEqual(restored.capabilities, ('LOCAL_INFERENCE',))
        self.assertEqual(restored.thinking.value, 'off')
        with self.assertRaises(ValueError):
            AgentDefinition(**{**self.definition.model_dump(), 'shell_command': 'whoami'})

    def test_registry_lists_describes_and_rejects_unknown_or_duplicate_agents(self):
        self.assertEqual([item.agent_id for item in self.registry.list()],
                         ['dante-hardware', 'dante-research'])
        self.assertEqual(self.registry.describe('dante-research').version, '1.0.0')
        with self.assertRaises(AgentUnavailable):
            self.registry.require('not-registered')
        with self.assertRaises(ValueError):
            self.registry.register(self.definition, DanteResearchAgent())

    def test_definition_digest_is_stable_across_registry_reconstruction(self):
        other = build_agent_registry(model_target()).describe('dante-research')
        self.assertEqual(definition_digest(self.definition), definition_digest(other))

    def test_payload_validation_bounds_objective_context_and_budget(self):
        with self.assertRaises(ValueError):
            agent_spec(self.definition, objective='   ')
        with self.assertRaises(ValueError):
            agent_spec(self.definition, objective='x' * 4001)
        with self.assertRaises(ValueError):
            agent_spec(self.definition, context='x' * 4001)
        with self.assertRaises(ValueError):
            agent_spec(self.definition, requested_tokens=513)

    def test_agent_task_requires_payload_and_cannot_be_mislabeled(self):
        requirement = ModelRequirement(model_id=MODEL_ID, runtime_reference=RUNTIME,
            digest_sha256=DIGEST, context_tokens=4096)
        with self.assertRaises(ValueError):
            WorkloadSpec(job_type='AGENT_TASK', model=requirement, prompt='agent')
        with self.assertRaises(ValueError):
            WorkloadSpec(job_type='LOCAL_INFERENCE', model=requirement, prompt='agent',
                         agent_task=agent_spec(self.definition).agent_task)

    def test_job_payload_survives_store_reopen(self):
        record = self.submit(objective='Inspect durable recovery.')
        reopened = WorkloadStore(self.db)
        saved = reopened.spec(str(record.job_id))
        self.assertEqual(saved.job_type, 'AGENT_TASK')
        self.assertEqual(saved.agent_task.objective, 'Inspect durable recovery.')
        self.assertEqual(saved.agent_task.definition_digest, definition_digest(self.definition))

    def test_restart_recovers_an_interrupted_agent_attempt_durably(self):
        record = self.submit()
        claimed, attempt = self.store.claim('interrupted-agent')
        self.assertTrue(self.store.start_attempt(str(record.job_id), str(attempt)))
        restarted = WorkloadOrchestrator(WorkloadStore(self.db), self.executor,
            poll_s=0.05, worker_id='recovered-agent')
        try:
            self.assertEqual(restarted.start(), [str(record.job_id)])
            self.assertEqual(restarted.store.get(str(record.job_id)).state, JobState.RETRY_WAIT)
            self.assertEqual(restarted.store.spec(str(record.job_id)).agent_task.objective,
                             'Analyze the local architecture.')
        finally:
            restarted.stop()


class AgentExecutionTests(AgentFixture):
    def test_success_persists_bounded_structured_result_and_audit_chain(self):
        audit_path = Path(self.temp.name) / 'agent-audit.jsonl'
        self.store.audit = JsonlAudit(audit_path)
        record = self.submit()
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            self.assertTrue(orchestrator.run_once())
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.SUCCEEDED)
            result = saved.result['structured_result']
            self.assertEqual(set(result), {'summary', 'findings', 'recommended_next_actions',
                'limitations', 'agent_id', 'model', 'qualification_id', 'started_at', 'completed_at'})
            self.assertEqual(result['agent_id'], 'dante-research')
            self.assertEqual(result['model'], MODEL_ID)
            self.assertEqual(result['qualification_id'], QUALIFICATION)
            self.assertEqual(saved.result['fallback'], False)
            self.assertEqual(self.local.calls[0].thinking.value, 'off')
            self.assertEqual(self.local.calls[0].max_output_tokens, 128)
            events = [event['event'] for event in self.store.events(str(record.job_id))]
            for required in ('agent.job.submitted', 'agent.execution.started',
                             'agent.inference.started', 'agent.inference.completed',
                             'agent.execution.completed'):
                self.assertIn(required, events)
            self.assertNotIn('Analyze the local architecture.', json.dumps(self.store.events(str(record.job_id))))
            audit_text = audit_path.read_text(encoding='utf-8')
            self.assertIn('workload.agent.execution.completed', audit_text)
            self.assertNotIn('Analyze the local architecture.', audit_text)
        finally:
            orchestrator.stop()

    def test_inference_failure_is_durable_and_never_stored_as_success(self):
        self.local.error = InferenceTimeout('bounded local timeout')
        record = self.submit()
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            orchestrator.run_once()
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.RETRY_WAIT)
            self.assertIsNone(saved.result)
            self.assertIn('agent.execution.failed', [e['event'] for e in self.store.events(str(record.job_id))])
        finally:
            orchestrator.stop()

    def test_invalid_model_json_fails_closed(self):
        self.local.response = 'not-json'
        record = self.submit(maximum_attempts=1)
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            orchestrator.run_once()
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.FAILED)
            self.assertIsNone(saved.result)
            self.assertIn('agent.execution.failed', [e['event'] for e in self.store.events(str(record.job_id))])
        finally:
            orchestrator.stop()

    def test_oversized_structured_report_fails_without_persisting_output(self):
        self.local.response = json.dumps({'summary': 'x' * 1201, 'findings': ['one'],
            'recommended_next_actions': ['next'], 'limitations': []})
        record = self.submit(maximum_attempts=1)
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            orchestrator.run_once()
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.FAILED)
            self.assertIsNone(saved.result)
        finally:
            orchestrator.stop()

    def test_timeout_is_enforced_by_existing_orchestrator_and_cleans_workers(self):
        self.local.cancellation_wait = True
        record = self.submit(timeout_s=0.05)
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            orchestrator.run_once()
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.RETRY_WAIT)
            self.assertIsNone(saved.result)
            self.assertFalse(orchestrator._active)
            self.assertIn('agent.execution.failed', [e['event'] for e in self.store.events(str(record.job_id))])
        finally:
            orchestrator.stop()

    def test_expired_deadline_is_terminal_without_inference(self):
        record = self.submit(deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            self.assertFalse(orchestrator.run_once())
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.TIMED_OUT)
            self.assertEqual(self.local.calls, [])
        finally:
            orchestrator.stop()

    def test_gpu_slot_serializes_concurrent_agent_dispatch(self):
        class SlotProbe(FakeLocalInference):
            def __init__(inner):
                super().__init__()
                inner.release = threading.Event()
                inner.lock = threading.Lock()
                inner.active = 0
                inner.peak = 0

            def execute(inner, spec, cancellation):
                with inner.lock:
                    inner.active += 1
                    inner.peak = max(inner.peak, inner.active)
                inner.started.set()
                inner.release.wait(2)
                with inner.lock:
                    inner.active -= 1
                return ExecutionResult(VALID_DRAFT, MODEL_ID, 'ollama', False, 'ollama', {
                    'qualification_id': QUALIFICATION, 'gpu_identity': GPU,
                    'gpu_vram_bytes': 1024, 'runtime_version': '0.34.4',
                })

        probe = SlotProbe()
        runner = AgentRunner(self.registry, probe, self.store)
        orchestrator = self.orchestrator(executor=Node0WorkloadExecutor(probe, runner))
        first, second = self.submit(), self.submit(objective='Second queued report.')
        orchestrator.start()
        finished = threading.Event()
        dispatch = threading.Thread(target=lambda: (orchestrator.run_once(), finished.set()), daemon=True)
        try:
            dispatch.start()
            self.assertTrue(probe.started.wait(1))
            orchestrator.run_once()
            self.assertEqual(self.store.get(str(second.job_id)).state, JobState.RETRY_WAIT)
            self.assertEqual(probe.peak, 1)
            probe.release.set()
            self.assertTrue(finished.wait(2))
            dispatch.join(1)
            self.assertEqual(self.store.get(str(first.job_id)).state, JobState.SUCCEEDED)
            self.assertEqual(probe.peak, 1)
        finally:
            probe.release.set()
            orchestrator.stop()

    def test_cancellation_reaches_active_agent_and_never_succeeds(self):
        self.local.cancellation_wait = True
        record = self.submit()
        orchestrator = self.orchestrator()
        orchestrator.start()
        finished = threading.Event()
        runner = threading.Thread(target=lambda: (orchestrator.run_once(), finished.set()), daemon=True)
        try:
            runner.start()
            self.assertTrue(self.local.started.wait(1))
            orchestrator.cancel(str(record.job_id))
            self.assertTrue(finished.wait(2))
            runner.join(1)
            self.assertEqual(self.store.get(str(record.job_id)).state, JobState.CANCELLED)
            self.assertFalse(orchestrator._active)
            events = [e['event'] for e in self.store.events(str(record.job_id))]
            self.assertIn('agent.execution.cancelled', events)
            self.assertNotIn('agent.execution.completed', events)
        finally:
            orchestrator.stop()

    def test_retry_uses_definition_policy_then_succeeds(self):
        self.local.error = InferenceTimeout('transient')
        record = self.submit(maximum_attempts=2, retry_base_s=0.1, retry_max_s=0.2)
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            orchestrator.run_once()
            self.assertEqual(self.store.get(str(record.job_id)).state, JobState.RETRY_WAIT)
            with closing(self.store._connect()) as db, db:
                db.execute('UPDATE workload_jobs SET eligible_at=0 WHERE job_id=?', (str(record.job_id),))
            orchestrator.run_once()
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.SUCCEEDED)
            self.assertEqual(saved.attempt_count, 2)
        finally:
            orchestrator.stop()

    def test_qualification_rejection_fails_closed(self):
        record = self.submit(maximum_attempts=1)
        from dante.workload import Node0InferenceExecutor
        supervisor = Mock()
        supervisor.operator_status.return_value = {'ready_for_local_routing': False, 'gate_accepted': False}
        inference = Node0InferenceExecutor(supervisor, expected_model_id=MODEL_ID,
            expected_runtime_reference=RUNTIME, expected_digest=DIGEST)
        runner = AgentRunner(self.registry, inference, self.store)
        executor = Node0WorkloadExecutor(inference, runner)
        orchestrator = self.orchestrator()
        orchestrator.executor = executor
        try:
            orchestrator.start()
            orchestrator.run_once()
            saved = self.store.get(str(record.job_id))
            self.assertEqual(saved.state, JobState.FAILED)
            self.assertIsNone(saved.result)
            supervisor.infer.assert_not_called()
        finally:
            orchestrator.stop()

    def test_registered_agent_cannot_execute_arbitrary_job_types(self):
        with self.assertRaises(ValueError):
            WorkloadSpec(job_type='SHELL_COMMAND', model=ModelRequirement(model_id=MODEL_ID,
                runtime_reference=RUNTIME, digest_sha256=DIGEST, context_tokens=4096), prompt='whoami')
        with self.assertRaises(ValueError):
            self.executor.execute(agent_spec(self.definition), threading.Event())


class SequencedInference(FakeLocalInference):
    def __init__(self, *responses):
        super().__init__()
        self.responses = list(responses)
    def execute(self, spec, cancellation):
        self.calls.append(spec)
        self.started.set()
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return ExecutionResult(self.responses[index], MODEL_ID, 'ollama', False, 'ollama', {
            'qualification_id': QUALIFICATION, 'gpu_identity': GPU, 'gpu_vram_bytes': 1024,
            'runtime_version': '0.34.4', 'context_tokens': 4096})


class AgentStructuredOutputTests(AgentFixture):
    def setUp(self):
        super().setUp()
        self.local = SequencedInference(VALID_DRAFT)
        self.runner = AgentRunner(self.registry, self.local, self.store)
        self.executor = Node0WorkloadExecutor(self.local, self.runner)

    def run_with(self, *responses):
        self.local.responses = list(responses)
        record = self.submit(maximum_attempts=1)
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            orchestrator.run_once()
        finally:
            orchestrator.stop()
        saved = self.store.get(str(record.job_id))
        return saved, [event['event'] for event in self.store.events(str(record.job_id))]

    def test_direct_valid_json_is_accepted_without_repair(self):
        saved, events = self.run_with(VALID_DRAFT)
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        self.assertEqual(saved.result['structured_result']['summary'], 'Three priorities are clear.')
        self.assertEqual(len(self.local.calls), 1)
        self.assertNotIn('agent.output.repair.started', events)

    def test_fenced_valid_json_is_accepted_without_repair(self):
        saved, events = self.run_with('```json\n' + VALID_DRAFT + '\n```')
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        self.assertEqual(len(self.local.calls), 1)
        self.assertNotIn('agent.output.invalid', events)

    def test_bare_fence_and_surrounding_whitespace_are_accepted(self):
        saved, _ = self.run_with('\n  ```\n' + VALID_DRAFT + '\n```  \n')
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        self.assertEqual(len(self.local.calls), 1)

    def test_prose_wrapped_json_is_rejected_then_repaired_once(self):
        saved, events = self.run_with('Here is the report:\n' + VALID_DRAFT, VALID_DRAFT)
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        self.assertEqual(len(self.local.calls), 2)
        self.assertIn('agent.output.invalid', events)
        self.assertIn('agent.output.repair.started', events)
        self.assertIn('agent.output.repair.completed', events)

    def test_multiple_fences_are_not_accepted(self):
        doubled = '```json\n' + VALID_DRAFT + '\n```\nand also\n```json\n' + VALID_DRAFT + '\n```'
        saved, _ = self.run_with(doubled, VALID_DRAFT)
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        self.assertEqual(len(self.local.calls), 2)

    def test_malformed_json_fails_closed_after_one_repair(self):
        saved, events = self.run_with('not-json', 'still not json')
        self.assertEqual(saved.state, JobState.FAILED)
        self.assertIsNone(saved.result)
        self.assertIn('agent.output.repair.failed', events)
        self.assertEqual(len(self.local.calls), 2)

    def test_schema_invalid_json_fails_closed_after_one_repair(self):
        bad = json.dumps({'summary': 'ok', 'findings': [], 'recommended_next_actions': ['a'],
            'limitations': []})
        saved, _ = self.run_with(bad, bad)
        self.assertEqual(saved.state, JobState.FAILED)
        self.assertIsNone(saved.result)
        self.assertEqual(len(self.local.calls), 2)

    def test_schema_rejects_unknown_fields(self):
        bad = json.dumps({'summary': 'ok', 'findings': ['a'], 'recommended_next_actions': ['b'],
            'limitations': [], 'shell_command': 'whoami'})
        saved, _ = self.run_with(bad, bad)
        self.assertEqual(saved.state, JobState.FAILED)
        self.assertIsNone(saved.result)

    def test_repair_never_runs_more_than_once(self):
        saved, _ = self.run_with('nope', 'nope', VALID_DRAFT)
        self.assertEqual(saved.state, JobState.FAILED)
        self.assertIsNone(saved.result)
        self.assertEqual(len(self.local.calls), 2)

    def test_repair_uses_qualified_local_route_and_states_the_schema(self):
        saved, _ = self.run_with('not-json', VALID_DRAFT)
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        repair = self.local.calls[1]
        self.assertEqual(repair.model.model_id, MODEL_ID)
        self.assertEqual(repair.model.digest_sha256, DIGEST)
        self.assertEqual(repair.model.runtime_reference, RUNTIME)
        self.assertEqual(repair.max_output_tokens, self.definition.output_token_budget)
        self.assertEqual(repair.thinking.value, 'off')
        self.assertIn('"recommended_next_actions"', repair.prompt)
        self.assertIn('no markdown', repair.prompt)
        self.assertIn('untrusted data', repair.prompt)
        self.assertIn('not-json', repair.prompt)
        result = saved.result['structured_result']
        self.assertEqual(result['qualification_id'], QUALIFICATION)
        self.assertEqual(result['model'], MODEL_ID)
        self.assertFalse(saved.result['fallback'])


class AgentSchemaConstrainedDecodingTests(AgentFixture):
    """The agent path must carry a decoding constraint, not just prompt wording."""

    def setUp(self):
        super().setUp()
        self.local = SequencedInference(VALID_DRAFT)
        self.runner = AgentRunner(self.registry, self.local, self.store)
        self.executor = Node0WorkloadExecutor(self.local, self.runner)

    def run_with(self, *responses):
        self.local.responses = list(responses)
        record = self.submit(maximum_attempts=1)
        orchestrator = self.orchestrator()
        try:
            orchestrator.start()
            orchestrator.run_once()
        finally:
            orchestrator.stop()
        saved = self.store.get(str(record.job_id))
        return saved, [event['event'] for event in self.store.events(str(record.job_id))]

    def test_agent_request_carries_the_research_draft_schema(self):
        self.run_with(VALID_DRAFT)
        self.assertEqual(len(self.local.calls), 1)
        schema = self.local.calls[0].output_schema
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema, DanteResearchAgent.draft_json_schema())
        self.assertEqual(schema['type'], 'object')

    def test_repair_request_carries_the_identical_schema(self):
        self.run_with('not-json', VALID_DRAFT)
        self.assertEqual(len(self.local.calls), 2)
        first, repair = self.local.calls
        self.assertIsNotNone(repair.output_schema)
        self.assertEqual(repair.output_schema, first.output_schema)

    def test_one_repair_maximum_still_holds_under_the_schema(self):
        saved, events = self.run_with('nope', 'nope', VALID_DRAFT)
        self.assertEqual(saved.state, JobState.FAILED)
        self.assertIsNone(saved.result)
        self.assertEqual(len(self.local.calls), 2)
        self.assertTrue(all(call.output_schema is not None for call in self.local.calls))
        self.assertEqual(events.count('agent.output.repair.started'), 1)
        self.assertIn('agent.output.repair.failed', events)

    def test_ordinary_workload_spec_carries_no_schema(self):
        spec = agent_spec(self.definition)
        self.assertIsNone(spec.output_schema)
        self.local.execute(spec, threading.Event())
        self.assertIsNone(self.local.calls[-1].output_schema)

    def test_schema_is_exactly_compatible_with_research_draft(self):
        schema = DanteResearchAgent.draft_json_schema()
        fields = set(ResearchDraft.model_fields)
        self.assertEqual(set(schema['properties']), fields)
        required = {name for name, f in ResearchDraft.model_fields.items() if f.is_required()}
        self.assertEqual(set(schema['required']), required)
        self.assertIs(schema['additionalProperties'], False)
        for name, fragment in schema['properties'].items():
            field = ResearchDraft.model_fields[name]
            self.assertEqual(fragment['type'], 'array' if field.annotation is not str else 'string')
        self.assertNotIn('$defs', json.dumps(schema))
        self.assertNotIn('title', json.dumps(schema))
        valid = {'summary': 's', 'findings': ['f'], 'recommended_next_actions': ['a']}
        # limitations is optional in the schema, so it defaults exactly as the
        # contract declares.
        self.assertEqual(ResearchDraft.model_validate(valid).model_dump(mode='json'),
                         {**valid, 'limitations': []})
        self.assertNotIn('limitations', schema['required'])
        with self.assertRaises(ValidationError):
            ResearchDraft.model_validate({**valid, 'unexpected': 1})

    def test_constrained_response_validates_and_persists(self):
        payload = json.dumps({'summary': 'Fail-closed admission is enforced.',
                              'findings': ['Admission is denied on stale or unknown backend state.'],
                              'recommended_next_actions': ['Keep the gate closed by default.'],
                              'limitations': ['Single qualified runtime observed.']})
        saved, events = self.run_with(payload)
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        self.assertEqual(saved.result['structured_result']['summary'], 'Fail-closed admission is enforced.')
        self.assertNotIn('agent.output.invalid', events)
        self.assertNotIn('agent.output.repair.started', events)
        self.assertEqual(len(self.local.calls), 1)

    def test_schema_constrained_reply_still_fails_closed_when_invalid(self):
        empty_findings = json.dumps({'summary': 's', 'findings': [],
                                     'recommended_next_actions': ['a']})
        saved, events = self.run_with(empty_findings, empty_findings)
        self.assertEqual(saved.state, JobState.FAILED)
        self.assertIsNone(saved.result)
        self.assertIn('agent.output.invalid', events)
        self.assertIn('agent.output.repair.failed', events)

    def test_qualification_and_local_only_route_unchanged_by_the_schema(self):
        saved, _ = self.run_with(VALID_DRAFT)
        self.assertEqual(saved.state, JobState.SUCCEEDED)
        for call in self.local.calls:
            self.assertEqual(call.model.model_id, MODEL_ID)
            self.assertEqual(call.model.digest_sha256, DIGEST)
            self.assertEqual(call.model.runtime_reference, RUNTIME)
            self.assertTrue(call.model.local_only)
        result = saved.result['structured_result']
        self.assertEqual(result['qualification_id'], QUALIFICATION)
        self.assertEqual(result['model'], MODEL_ID)
        self.assertEqual(result['agent_id'], 'dante-research')
        self.assertEqual(saved.result['fallback'], False)
        self.assertEqual(saved.result['metadata']['gpu_identity'], GPU)


class AgentControlPlaneTests(AgentFixture):
    def setUp(self):
        super().setUp()
        self.supervisor = Mock()
        self.supervisor.config = type('Config', (), {
            'model': type('Model', (), {'model_id': MODEL_ID, 'local_metadata': object()})(),
            'qualification': type('Qualification', (), {'model_reference': RUNTIME,
                'model_digest': DIGEST, 'context_tokens': 4096})(),
        })()
        self.supervisor.operator_status.return_value = {
            'ready_for_local_routing': True, 'gate_accepted': True,
        }
        self.service = Node0ControlService(self.supervisor, workload_store=self.store,
            agent_registry=self.registry)

    def test_control_plane_lists_describes_and_rejects_unknown_agent(self):
        listed = self.service.dispatch({'op': 'agent_list'})['result']
        self.assertEqual([item['agent_id'] for item in listed],
                         ['dante-hardware', 'dante-research'])
        described = self.service.dispatch({'op': 'agent_describe', 'agent_id': 'dante-research'})['result']
        self.assertEqual(described['capabilities'], ['LOCAL_INFERENCE'])
        with self.assertRaisesRegex(ControlError, 'unknown_agent'):
            self.service.dispatch({'op': 'agent_describe', 'agent_id': 'shell'})

    def test_control_plane_submit_durable_job_status_cancel_and_result(self):
        submitted = self.service.dispatch({'op': 'agent_submit', 'agent_id': 'dante-research',
            'objective': 'Create three engineering priorities.', 'requested_output_tokens': 128})['result']
        job_id = submitted['job_id']
        self.assertEqual(submitted['state'], 'queued')
        self.assertEqual(self.store.spec(job_id).job_type, 'AGENT_TASK')
        self.assertEqual(self.service.dispatch({'op': 'agent_job', 'job_id': job_id})['result']['job_id'], job_id)
        with self.assertRaisesRegex(ControlError, 'agent_result_unavailable'):
            self.service.dispatch({'op': 'agent_result', 'job_id': job_id})
        cancelled = self.service.dispatch({'op': 'agent_job_cancel', 'job_id': job_id})['result']
        self.assertEqual(cancelled['state'], 'cancelled')
        self.assertIn('agent.job.cancel_requested', [e['event'] for e in self.store.events(job_id)])

    def test_control_plane_returns_only_the_persisted_structured_agent_result(self):
        submitted = self.service.dispatch({'op': 'agent_submit', 'agent_id': 'dante-research',
            'objective': 'Return the structured architecture report.'})['result']
        orchestrator = self.orchestrator()
        self.service.orchestrator = orchestrator
        try:
            orchestrator.start()
            orchestrator.run_once()
            response = self.service.dispatch({'op': 'agent_result', 'job_id': submitted['job_id']})['result']
            self.assertEqual(response['state'], 'succeeded')
            self.assertEqual(response['result']['agent_id'], 'dante-research')
            self.assertEqual(response['result']['summary'], 'Three priorities are clear.')
            self.assertFalse(response['execution']['fallback'])
        finally:
            orchestrator.stop()

    def test_control_plane_rejects_invalid_objective_budget_and_unqualified_node(self):
        with self.assertRaisesRegex(ControlError, 'invalid_agent_request'):
            self.service.dispatch({'op': 'agent_submit', 'agent_id': 'dante-research', 'objective': ' '})
        with self.assertRaisesRegex(ControlError, 'invalid_output_budget'):
            self.service.dispatch({'op': 'agent_submit', 'agent_id': 'dante-research',
                'objective': 'analysis', 'requested_output_tokens': 512})
        self.supervisor.operator_status.return_value = {'ready_for_local_routing': False, 'gate_accepted': False}
        with self.assertRaisesRegex(ControlError, 'qualification_or_runtime_not_ready'):
            self.service.dispatch({'op': 'agent_submit', 'agent_id': 'dante-research', 'objective': 'analysis'})

    def test_generic_workload_operation_cannot_submit_agent_task(self):
        spec = agent_spec(self.definition).model_dump(mode='json')
        with self.assertRaisesRegex(ControlError, 'agent_task_requires_agent_submit'):
            self.service.dispatch({'op': 'workload_submit', 'spec': spec, 'idempotency_key': None})

    @unittest.skipUnless(os.name == 'nt', 'Windows named-pipe integration test')
    def test_agent_operations_round_trip_over_local_control_pipe(self):
        pipe_name = rf'\\.\pipe\DanteNode0.AgentTest.{uuid4().hex}'
        server = Node0ControlServer(self.service, pipe_name=pipe_name)
        client = Node0ControlClient(pipe_name=pipe_name, timeout_s=3)
        try:
            server.start()
            listed = client.request({'op': 'agent_list'})
            self.assertEqual([agent['agent_id'] for agent in listed],
                             ['dante-hardware', 'dante-research'])
            submitted = client.request({'op': 'agent_submit', 'agent_id': 'dante-research',
                'objective': 'Round-trip through the local pipe.'})
            self.assertEqual(submitted['state'], 'queued')
            self.assertEqual(self.store.spec(submitted['job_id']).agent_task.objective,
                             'Round-trip through the local pipe.')
        finally:
            server.stop(timeout_s=3)


class AgentCliTests(unittest.TestCase):
    def test_cli_maps_registered_agent_commands_only_to_control_operations(self):
        cases = [
            (['agent', 'list'], {'op': 'agent_list'}),
            (['agent', 'describe', 'dante-research'],
             {'op': 'agent_describe', 'agent_id': 'dante-research'}),
            (['agent', 'submit', 'dante-research', '--objective', 'Analyze the architecture.',
              '--context', 'Node 0 is local.', '--output-tokens', '128'],
             {'op': 'agent_submit', 'agent_id': 'dante-research', 'objective': 'Analyze the architecture.',
              'context': 'Node 0 is local.', 'requested_output_tokens': 128}),
            (['agent', 'job', 'job_1'], {'op': 'agent_job', 'job_id': 'job_1'}),
            (['agent', 'cancel', 'job_1'], {'op': 'agent_job_cancel', 'job_id': 'job_1'}),
            (['agent', 'result', 'job_1'], {'op': 'agent_result', 'job_id': 'job_1'}),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                client = Mock()
                client.request.return_value = {'ok': True}
                output = []
                with patch('dante.node0_cli.Node0ControlClient', return_value=client), \
                     patch('dante.node0_cli._print', side_effect=output.append):
                    self.assertEqual(node0_cli_main(arguments), 0)
                client.request.assert_called_once_with(expected)
                self.assertEqual(output, [{'ok': True}])

    def test_invalid_agent_cli_arguments_fail_before_pipe_client(self):
        with patch('dante.node0_cli.Node0ControlClient') as client:
            with self.assertRaises(SystemExit) as raised:
                node0_cli_main(['agent', 'submit', 'dante-research'])
        self.assertEqual(raised.exception.code, 2)
        client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
