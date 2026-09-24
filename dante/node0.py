"""Explicit Node 0 composition. Construction never starts a runtime or qualifies a model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from dante.agent_host import AgentHostFoundation
from dante.continuity import ContinuityManager
from dante.contracts import ModelRef, utc_now
from dante.contracts.continuity import ContinuityPolicy
from dante.contracts.qualification import ModelQualification, QualificationIdentity
from dante.inference import InferenceGateway
from dante.ledger import TaskLedger
from dante.local_runtime import OllamaAdapter
from dante.ollama_qualification import OllamaQualificationConfig, OllamaQualificationProbeFactory
from dante.privacy import PrivacyGate
from dante.qualification import NodeQualificationGate, QualificationRunner, QualificationStore
from dante.qualification_evidence import EvidenceStore
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter
from dante.telemetry import JsonlAudit
from dante.tool_broker import ToolBroker


class Node0Router(RuleBasedRouter):
    def __init__(self, registry, audit):
        super().__init__(registry)
        self.audit = audit

    def decide(self, task, privacy, capabilities):
        try:
            decision = super().decide(task, privacy, capabilities)
        except Exception as exc:
            self.audit.write('route.denied', reason_type=type(exc).__name__)
            raise
        self.audit.write('route.selected', model=decision.selected_model.model_id,
                         provider=decision.selected_model.provider_id)
        return decision

    def continuity_candidates(self, privacy, capabilities, policy, *, context_tokens=None):
        eligible, denied = super().continuity_candidates(
            privacy, capabilities, policy, context_tokens=context_tokens)
        self.audit.write('route.candidates', eligible=[model.model_id for model in eligible], denied=denied)
        return eligible, denied


class Node0Gateway(InferenceGateway):
    def infer(self, decision, request):
        self.audit.write('inference.started', model=decision.selected_model.model_id,
                         provider=decision.selected_model.provider_id)
        try:
            return super().infer(decision, request)
        except Exception as exc:
            self.audit.write('inference.denied_or_failed', error_type=type(exc).__name__)
            raise


@dataclass
class Node0Runtime:
    ledger: TaskLedger
    evidence: EvidenceStore
    store: QualificationStore
    gate: NodeQualificationGate
    registry: ModelRegistry
    router: Node0Router
    gateway: Node0Gateway
    adapter: OllamaAdapter
    continuity: ContinuityManager
    host: AgentHostFoundation
    probe: object
    audit: JsonlAudit
    requested_identity: QualificationIdentity

    def qualify(self):
        """Supersede prior PASS before preflight, including when preflight raises."""
        attempt = uuid4()
        started = utc_now()
        # The intended CUDA scope must supersede an earlier CUDA PASS even if
        # the runtime is currently down and the fresh observer cannot run.
        scope_identity = self.requested_identity.model_copy(update={
            'runtime': self.requested_identity.runtime.model_copy(update={'backend': 'cuda'})})
        self.store.append(ModelQualification(qualification_id=uuid4(), attempt_id=attempt,
            phase='intent', identity=scope_identity, source=self.probe.source,
            started_at=started))
        self.audit.write('qualification.started', attempt_id=str(attempt),
                         model=self.requested_identity.model_id)
        try:
            prepare = getattr(self.probe.observer, 'prepare', None)
            if prepare is None:
                raise ValueError('Fresh identity preparation unavailable')
            current = prepare()
            self.probe.identity = current
            self.probe.observer.requested = current
            result = QualificationRunner(self.store).run(self.probe, requested_identity=current)
        except Exception:
            self.store.append(ModelQualification(qualification_id=uuid4(), attempt_id=attempt,
                phase='completed', identity=scope_identity, source=self.probe.source,
                started_at=started, completed_at=utc_now(), interrupted=True))
            self.audit.write('qualification.failed', attempt_id=str(attempt), reason='preflight_or_runner_error')
            raise
        self.audit.write('qualification.completed', attempt_id=str(result.attempt_id),
                         state=self.store.status(current).state.value)
        return result


def build_node0(*, ledger_path: Path, evidence_path: Path, audit_path: Path,
                seed_identity: QualificationIdentity, qualification: OllamaQualificationConfig,
                models: list[ModelRef], machine_profile: str,
                policy: ContinuityPolicy | None = None,
                probe_factory: OllamaQualificationProbeFactory | None = None,
                adapters: list | None = None, audit=None) -> Node0Runtime:
    """One ledger, evidence verifier, gate, registry and execution graph."""
    audit = audit or JsonlAudit(audit_path)
    ledger = TaskLedger(ledger_path, audit)
    evidence = EvidenceStore(evidence_path, strict=True, model_reference=qualification.model_reference)
    store = QualificationStore(ledger, evidence)
    probe = (probe_factory or OllamaQualificationProbeFactory(qualification, evidence))(seed_identity)
    gate = NodeQualificationGate(store, lambda model: probe.observer())
    registry = ModelRegistry(models, machine_profile=machine_profile, qualification_gate=gate)
    router = Node0Router(registry, audit)
    adapter = OllamaAdapter(qualification.profile, machine_profile=machine_profile, registry=registry)
    gateway = Node0Gateway(registry, [adapter, *(adapters or [])], audit)
    continuity = ContinuityManager(ledger, router, gateway, policy or ContinuityPolicy())
    host = AgentHostFoundation(ledger, PrivacyGate(), router, gateway, ToolBroker(audit, ledger=ledger),
                               continuity=continuity)
    return Node0Runtime(ledger, evidence, store, gate, registry, router, gateway, adapter,
                        continuity, host, probe, audit, seed_identity)
