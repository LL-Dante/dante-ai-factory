from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dante.acceptance import AcceptanceContract, AcceptanceResult, ObservedAcceptance, continuation_prompt
from dante.contracts import PrivacyClass, Task, TaskStatus
from dante.inference import InferenceGateway, InferenceResult
from dante.contracts.inference import InferenceRequest, ToolDefinition
from dante.ledger import TaskLedger
from dante.recovery import ReconciliationRequired
from dante.privacy import PrivacyGate
from dante.routing import RuleBasedRouter
from dante.telemetry import TraceContext, reset_trace, use_trace
from dante.tool_broker import ToolBroker, ToolDenied
from dante.contracts.tools import ToolStatus


class AgentHostFoundation:
    """Persistent orchestration boundary; cptr remains the operational agent loop."""

    def __init__(self, ledger: TaskLedger, privacy: PrivacyGate, router: RuleBasedRouter, gateway: InferenceGateway, tools: ToolBroker, *, continuity=None) -> None:
        self.ledger, self.privacy, self.router, self.gateway, self.tools = ledger, privacy, router, gateway, tools
        self.continuity = continuity
        if tools is not None:
            if tools.ledger is not None and tools.ledger is not ledger:
                raise ValueError('Tool Broker belongs to another ledger')
            tools.ledger = ledger

    def start(self, goal: str, workspace: str, privacy_class: PrivacyClass, *, acceptance: AcceptanceContract | None = None) -> Task:
        task = self.ledger.create_task(Task(goal=goal, workspace=workspace, privacy_class=privacy_class), acceptance)
        self.ledger.transition(task.task_id, TaskStatus.PLANNED, current_step="planned")
        return self.ledger.transition(task.task_id, TaskStatus.RUNNING, current_step="routing")

    def infer(self, task_id: str, content: str, capabilities: set[str], *, tools: tuple[ToolDefinition, ...] = (), required_context_tokens: int | None = None) -> InferenceResult:
        task = self.ledger.get_task(task_id)
        trace = TraceContext.create(task.task_id, task.trace_id)
        token = use_trace(trace)
        try:
            privacy_content = content + "\n" + "\n".join(tool.model_dump_json() for tool in tools)
            privacy = self.privacy.classify(privacy_content, task.privacy_class)
            if self.gateway.audit:
                self.gateway.audit.write("privacy.decided", classification=privacy.classification.value, cloud_allowed=privacy.cloud_allowed, policy_version=privacy.policy_version)
            if self.continuity is not None:
                from dante.continuity import ContinuitySignal, Disposition
                outcome = self.continuity.infer(task, privacy, [{'role': 'user', 'content': content}],
                    capabilities, tools=tools, context_tokens=required_context_tokens)
                if outcome.disposition != Disposition.CONTINUE_NOW:
                    raise ContinuitySignal(outcome)
                return outcome.response
            decision = self.router.decide(task, privacy, capabilities | ({"tool_calling"} if tools else set()))
            self.ledger.set_route(decision)
            return self.gateway.infer(decision, InferenceRequest(
                model=decision.selected_model, messages=[{"role": "user", "content": content}],
                tools=tools, task_id=task.task_id, trace_id=task.trace_id))
        finally:
            reset_trace(token)

    def execute_tool_and_checkpoint(self, task_id: str, tool_id: str, arguments: dict[str, Any], *,
                                    stop_after: bool = False, idempotency_key: str | None = None,
                                    approval_id: str | None = None, plan_hash: str = '') -> dict[str, Any]:
        task = self.ledger.get_task(task_id)
        token = use_trace(TraceContext.create(task.task_id, task.trace_id))
        try:
            if task.status == TaskStatus.FAILED_RETRYABLE:
                task = self.resume(task_id)
                if task.status == TaskStatus.FAILED_RETRYABLE:
                    raise ReconciliationRequired('Unresolved task effects require reconciliation')
            task = self.ledger.transition(task_id, TaskStatus.WAITING_TOOL, current_step=f"tool:{tool_id}")
            outcome = self.tools.invoke(task, tool_id, arguments, approval_id=approval_id,
                                        plan_hash=plan_hash, idempotency_key=idempotency_key)
            if outcome.status == ToolStatus.UNCERTAIN:
                raise ReconciliationRequired('Prior tool invocation requires reconciliation')
            if outcome.status != ToolStatus.SUCCESS:
                raise ToolDenied(outcome)
            result = self.ledger.verified_step_result(task_id, outcome.step_id)
            self.ledger.checkpoint(task_id, f"tool:{tool_id}:complete", {'step_id': outcome.step_id})
            if not stop_after:
                self.ledger.transition(task_id, TaskStatus.RUNNING, current_step="post-tool")
            return result
        except Exception:
            current = self.ledger.get_task(task_id)
            if current.status in {TaskStatus.RUNNING, TaskStatus.WAITING_TOOL}:
                self.ledger.transition(task_id, TaskStatus.FAILED_RETRYABLE, current_step='reconcile-or-retry')
            raise
        finally:
            reset_trace(token)

    def resume(self, task_id: str) -> Task:
        task = self.ledger.get_task(task_id)
        for step in self.ledger.steps(task_id):
            if step.state == 'intent':
                continue
            try:
                self.ledger.verified_step_result(task_id, step.step_id)
            except ReconciliationRequired:
                if task.status in {TaskStatus.RUNNING, TaskStatus.WAITING_TOOL, TaskStatus.VERIFYING}:
                    return self.ledger.transition(task_id, TaskStatus.FAILED_RETRYABLE,
                                                  current_step='reconcile:' + step.step_id)
                return task
        if task.status in {TaskStatus.WAITING_TOOL, TaskStatus.FAILED_RETRYABLE}:
            return self.ledger.transition(task_id, TaskStatus.RUNNING, current_step="resumed")
        return task

    def verify_acceptance(self, task_id: str, contract: AcceptanceContract | None = None,
                          observed: ObservedAcceptance | None = None) -> AcceptanceResult:
        if contract is not None:
            self.ledger.bind_acceptance(task_id, contract)
        if observed is not None:
            self.ledger.record_observations(task_id, observed)
        task = self.ledger.get_task(task_id)
        if task.status != TaskStatus.VERIFYING:
            self.ledger.transition(task_id, TaskStatus.VERIFYING, current_step="verifying-acceptance")
        result = self.ledger.verify_persistent_acceptance(task_id)
        self.ledger.checkpoint(task_id, "acceptance:verified", result.model_dump(mode="json"))
        return result

    def complete(self, task_id: str, contract: AcceptanceContract | None = None, observed: ObservedAcceptance | None = None) -> Task:
        if self.ledger.get_task(task_id).status == TaskStatus.COMPLETED:
            return self.ledger.get_task(task_id)
        result = self.verify_acceptance(task_id, contract, observed)
        if not result.acceptance_complete:
            return self.ledger.transition(task_id, TaskStatus.FAILED_RETRYABLE, current_step="acceptance-incomplete",
                                          metadata=result.model_dump(mode="json"))
        return self.ledger.transition(task_id, TaskStatus.COMPLETED, current_step="completed")

    def complete_with_one_continuation(
        self,
        task_id: str,
        contract: AcceptanceContract,
        observed: ObservedAcceptance,
        continue_missing: Callable[[str], ObservedAcceptance],
        *,
        known_final_fields: dict[str, str] | None = None,
    ) -> tuple[Task, AcceptanceResult]:
        first = self.verify_acceptance(task_id, contract, observed)
        if first.acceptance_complete:
            return self.ledger.transition(task_id, TaskStatus.COMPLETED, current_step="completed"), first

        self.ledger.transition(task_id, TaskStatus.FAILED_RETRYABLE, current_step="acceptance-incomplete", metadata=first.model_dump(mode="json"))
        self.resume(task_id)
        combined = observed.merged(continue_missing(continuation_prompt(first, known_final_fields, contract.instructions)))
        second = self.verify_acceptance(task_id, contract, combined)
        target = TaskStatus.COMPLETED if second.acceptance_complete else TaskStatus.FAILED_RETRYABLE
        step = "completed" if second.acceptance_complete else "acceptance-incomplete"
        return self.ledger.transition(task_id, target, current_step=step, metadata=second.model_dump(mode="json")), second
