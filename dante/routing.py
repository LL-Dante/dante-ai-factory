from __future__ import annotations

from dante.contracts import CostClass, LifecycleState, RouteDecision, Task
from dante.privacy import PrivacyDecision
from dante.registry import ModelRegistry


class RouteDenied(RuntimeError):
    pass


class RuleBasedRouter:
    def __init__(self, registry: ModelRegistry, *, policy_version: str = "routing-rules-v1") -> None:
        self.registry = registry
        self.policy_version = policy_version

    def decide(self, task: Task, privacy: PrivacyDecision, capabilities: set[str]) -> RouteDecision:
        if not privacy.model_allowed:
            raise RouteDenied("Privacy policy denies model access")
        eligible = []
        for model in self.registry.candidates():
            if not model.available or model.lifecycle not in {LifecycleState.APPROVED, LifecycleState.PRODUCTION}:
                continue
            if model.cost_class not in {CostClass.ZERO, CostClass.LOCAL_COMPUTE} or (model.cost_class == CostClass.LOCAL_COMPUTE and not model.local):
                continue
            if not capabilities.issubset(model.capabilities):
                continue
            if privacy.classification not in model.privacy_eligibility:
                continue
            if not privacy.cloud_allowed and not model.local:
                continue
            try:
                self.registry.require_automatic(model, tool_use='tool_calling' in capabilities)
            except ValueError:
                continue
            eligible.append(model)
        if not eligible:
            raise RouteDenied("No zero-cost privacy-eligible model is available")
        eligible.sort(key=lambda model: (not model.local, model.logical_alias, model.model_id))
        selected = eligible[0]
        reasons = list(privacy.reasons)
        reasons.extend(("automatic_cost_zero", "local_selected" if selected.local else "cloud_free_selected"))
        return RouteDecision(
            task_id=task.task_id,
            trace_id=task.trace_id,
            selected_model=selected,
            candidates=tuple(model.model_id for model in eligible if model.local == selected.local),
            reasons=tuple(reasons),
            policy_version=self.policy_version,
            automatic_cost=0,
        )


    def continuity_candidates(self, privacy, capabilities, policy, *, context_tokens=None):
        """Static gates only. Operational health is evaluated independently each attempt."""
        eligible, denied = [], {}
        for model in self.registry.candidates():
            reason = None
            if (not privacy.model_allowed or privacy.classification not in model.privacy_eligibility
                    or (not privacy.cloud_allowed and not model.local)):
                reason = 'privacy'
            elif ((policy.mode == 'LOCAL_ONLY' and not model.local)
                    or (policy.allowed_backends and model.provider_id not in policy.allowed_backends)
                    or (policy.mode == 'SPECIFIC_ALLOWED_BACKENDS' and not policy.allowed_backends)):
                reason = 'user_policy'
            elif (model.cost_class not in {CostClass.ZERO, CostClass.LOCAL_COMPUTE}
                    or (not model.local and (model.cost_class != CostClass.ZERO
                                             or model.cost_verification != 'VERIFIED_ZERO'))):
                reason = 'cost_unverified_or_paid'
            elif model.lifecycle not in {LifecycleState.APPROVED, LifecycleState.PRODUCTION}:
                reason = 'qualification'
            elif not capabilities.issubset(model.capabilities):
                reason = 'capability'
            else:
                capacity = model.local_metadata.context_tokens if model.local_metadata else model.context_tokens
                if context_tokens is not None and (capacity is None or capacity < context_tokens):
                    reason = 'context_capacity'
                else:
                    try:
                        # Model qualification is separate from administrative availability.
                        self.registry.require_automatic(model.model_copy(update={'available': True}),
                                                        tool_use='tool_calling' in capabilities)
                    except ValueError:
                        reason = 'qualification_or_machine'
            if reason:
                denied[model.model_id] = reason
            else:
                eligible.append(model)
        cloud_first = policy.mode == 'CLOUD_FIRST_WITH_LOCAL_FALLBACK'
        eligible.sort(key=lambda m: (m.local if cloud_first else not m.local, m.logical_alias, m.model_id))
        return eligible, denied
