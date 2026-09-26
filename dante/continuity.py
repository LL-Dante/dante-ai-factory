"""Persistent, backend-neutral route reevaluation. No transports or tool execution."""
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time

from dante.contracts import CostClass, LifecycleState, PrivacyClass, RouteDecision
from dante.contracts.continuity import (BackendState as State, ContinuityDisposition as Disposition,
    ContinuityPolicy, ExecutionPolicy)
from dante.contracts.inference import (DEFAULT_MAX_OUTPUT_TOKENS, InferenceRequest,
    InferenceResponse, ThinkingPolicy)
from dante.inference import (InferenceError, InferenceTimeout, AdapterUnavailable, RateQuotaUnavailable,
    QuotaExhausted, ContextExceeded, InvalidResponse, PolicyDenied, QualificationDenied,
    ConfigurationDenied, InvalidRequest)
from dante.task_queue import assert_lease


@dataclass(frozen=True)
class ContinuityOutcome:
    disposition: Disposition
    reason: str
    response: InferenceResponse | None = None
    next_attempt_at: float | None = None
    diagnostic: dict | None = None


class ContinuitySignal(Exception):
    def __init__(self, outcome):
        self.outcome = outcome
        super().__init__(outcome.reason)


def classify(error):
    """Reason, operational state (None means request/route denial), retryable."""
    for cls, reason, state, retry in (
        (QuotaExhausted, 'quota_exhausted', State.QUOTA_EXHAUSTED, True),
        (RateQuotaUnavailable, 'rate_limit', State.RATE_LIMITED, True),
        (InferenceTimeout, 'timeout', State.DEGRADED, True),
        (AdapterUnavailable, 'unavailable', State.UNAVAILABLE, True),
        (ContextExceeded, 'context_exceeded', None, False),
        (InvalidRequest, 'invalid_request', None, False),
        (InvalidResponse, 'invalid_response', State.DEGRADED, True),
        (QualificationDenied, 'qualification_denied', None, False),
        (ConfigurationDenied, 'configuration_denied', None, False),
        (PolicyDenied, 'policy_denied', None, False),
    ):
        if isinstance(error, cls):
            return reason, state, retry
    return 'unclassified_inference_error', None, False


class ContinuityManager:
    def __init__(self, ledger, router, gateway, policy: ContinuityPolicy, *, clock=time.time):
        self.ledger, self.router, self.gateway, self.policy, self.clock = ledger, router, gateway, policy, clock

    def observation(self, backend_id):
        with closing(self.ledger._connect()) as db:
            row = db.execute('SELECT * FROM backend_observations WHERE backend_id=?', (backend_id,)).fetchone()
        return dict(row) if row else None

    def read_only_status(self, *, context_tokens=None):
        """Return bounded operational state for configured providers without writing state."""
        now = self.clock()
        models = self.router.registry.candidates()
        providers = sorted({model.provider_id for model in models if model.provider_id})
        result = []
        for provider in providers:
            provider_models = [model for model in models if model.provider_id == provider]
            observation = self.observation(provider)
            state = observation['state'] if observation else State.UNKNOWN.value
            reason = observation['reason'] if observation else None
            failures = observation['consecutive_failures'] if observation else 0
            observed_at = observation['observed_at'] if observation else None
            last_success = observation['last_success'] if observation else None
            cooldown_until = observation['cooldown_until'] if observation else None
            cooldown_remaining = max(0.0, cooldown_until - now) if cooldown_until is not None else None
            probation = cooldown_until is not None and cooldown_until <= now
            retry_at = None
            denial = None

            if not provider_models:
                denial = 'provider_not_configured'
            elif self.policy.allowed_backends and provider not in self.policy.allowed_backends:
                denial = 'user_policy'
            else:
                eligible = []
                static_denials = []
                for model in provider_models:
                    capacity = (model.local_metadata.context_tokens if model.local_metadata
                                else model.context_tokens)
                    if ((self.policy.mode == ExecutionPolicy.LOCAL_ONLY and not model.local)
                            or (self.policy.mode == ExecutionPolicy.CLOUD_ONLY and model.local)
                            or (self.policy.mode == ExecutionPolicy.SPECIFIC_ALLOWED_BACKENDS
                                and not self.policy.allowed_backends)):
                        static_denials.append('user_policy')
                    elif (not model.available or model.lifecycle not in
                          {LifecycleState.APPROVED, LifecycleState.PRODUCTION}):
                        static_denials.append('administratively_unavailable' if not model.available
                                              else 'qualification')
                    elif (model.cost_class not in {CostClass.ZERO, CostClass.LOCAL_COMPUTE}
                            or (not model.local and model.cost_verification != 'VERIFIED_ZERO')):
                        static_denials.append('cost_unverified_or_paid')
                    elif PrivacyClass.INTERNAL not in model.privacy_eligibility or (not model.local
                                                                                   and self.policy.mode == ExecutionPolicy.LOCAL_ONLY):
                        static_denials.append('privacy')
                    elif context_tokens is not None and (capacity is None or capacity < context_tokens):
                        static_denials.append('context_capacity')
                    else:
                        eligible.append(model)
                if not eligible:
                    denial = static_denials[0] if static_denials else 'no_eligible_route'
                adapter = self.gateway.adapters.get(provider)
                if denial is None and (adapter is None or adapter.automatic_cost != 0):
                    denial = 'adapter_configuration_or_cost'
                elif denial is None and (observation is None or state == State.UNKNOWN.value):
                    denial = 'health_unknown'
                    retry_at = now + self.policy.base_backoff_s
                elif denial is None and cooldown_until is not None and cooldown_until > now:
                    denial = 'cooldown'
                    retry_at = cooldown_until
                elif (denial is None and cooldown_until is None
                      and state not in {State.AVAILABLE.value, State.DEGRADED.value}):
                    denial = 'health_unavailable_or_stale'
                    retry_at = now + self.policy.base_backoff_s
                elif (denial is None and cooldown_until is None
                      and now - observed_at > self.policy.observation_ttl_s):
                    denial = 'health_unavailable_or_stale'
                    retry_at = now + self.policy.base_backoff_s

            result.append({
                'provider': provider,
                'state': state,
                'reason': reason,
                'consecutive_errors': failures,
                'last_observed_at': self._status_timestamp(observed_at),
                'last_failure_at': self._status_timestamp(observed_at) if failures else None,
                'last_success_at': self._status_timestamp(last_success),
                'cooldown_until': self._status_timestamp(cooldown_until),
                'cooldown_remaining_s': round(cooldown_remaining, 3) if cooldown_remaining is not None else None,
                'probation': bool(probation),
                'retry_at': self._status_timestamp(retry_at),
                'retry_in_s': round(max(0.0, retry_at - now), 3) if retry_at is not None else None,
                'route_admissible': denial is None,
                'denial_reason': denial,
            })
        return result

    @staticmethod
    def _status_timestamp(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None

    def observe(self, backend_id, state: State, *, cooldown_until=None, reason=None, task=None):
        """Trusted health input; free-form reasons and response bodies are never stored."""
        state = State(state)
        now = self.clock()
        if cooldown_until is not None and not math.isfinite(cooldown_until):
            raise ValueError('Invalid cooldown')
        reason = reason or state.value
        if reason not in {'available', 'degraded', 'rate_limit', 'rate_limited', 'quota_exhausted',
                          'unavailable', 'cooldown', 'unknown', 'timeout', 'invalid_response'}:
            raise ValueError('Invalid backend reason')
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if task is not None:
                assert_lease(db, task.task_id)
            row = db.execute('SELECT * FROM backend_observations WHERE backend_id=?', (backend_id,)).fetchone()
            failures = 0 if state == State.AVAILABLE else (row['consecutive_failures'] if row else 0) + 1
            last_success = now if state == State.AVAILABLE else (row['last_success'] if row else None)
            db.execute('INSERT OR REPLACE INTO backend_observations VALUES(?,?,?,?,?,?,?)',
                (backend_id, state.value, reason, now, cooldown_until, failures, last_success))

    def _event(self, task, event, metadata):
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            assert_lease(db, task.task_id)
            self.ledger._event(db, task, 'continuity.' + event, task.status, metadata)

    def _reserve(self, task):
        now = self.clock()
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            assert_lease(db, task.task_id, before_effect=True)
            db.execute('INSERT OR IGNORE INTO task_continuity(task_id,window_start) VALUES(?,?)', (task.task_id, now))
            row = db.execute('SELECT * FROM task_continuity WHERE task_id=?', (task.task_id,)).fetchone()
            if now >= row['window_start'] + self.policy.window_s:
                db.execute('UPDATE task_continuity SET window_start=?,attempts=0 WHERE task_id=?', (now, task.task_id))
            elif row['attempts'] >= self.policy.max_route_attempts:
                return row['window_start'] + self.policy.window_s
            db.execute('UPDATE task_continuity SET attempts=attempts+1 WHERE task_id=?', (task.task_id,))
        return None

    def _outcome(self, task, disposition, reason, *, response=None, retry_at=None, diagnostic=None):
        with closing(self.ledger._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            assert_lease(db, task.task_id)
            db.execute('INSERT OR IGNORE INTO task_continuity(task_id,window_start) VALUES(?,?)', (task.task_id, self.clock()))
            db.execute('UPDATE task_continuity SET next_attempt_at=?,last_reason=?,disposition=? WHERE task_id=?',
                       (retry_at, reason, disposition.value, task.task_id))
            if disposition == Disposition.CONTINUE_NOW:
                db.execute('UPDATE task_continuity SET window_start=?,attempts=0 WHERE task_id=?',
                           (self.clock(), task.task_id))
            metadata = {'disposition': disposition.value, 'reason': reason, 'next_attempt_at': retry_at}
            if diagnostic is not None:
                metadata['diagnostic'] = diagnostic
            self.ledger._event(db, task, 'continuity.outcome', task.status, metadata)
        return ContinuityOutcome(disposition, reason, response, retry_at, diagnostic)

    def infer(self, task, privacy, messages, capabilities, *, tools=(), context_tokens=None,
              max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, thinking=ThinkingPolicy.OFF):
        if context_tokens is not None and (isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens <= 0):
            return self._outcome(task, Disposition.TERMINAL, 'invalid_request')
        excluded = {}
        previous = None
        last_diagnostic = None
        while True:
            now = self.clock()
            candidates, denied = self.router.continuity_candidates(privacy,
                capabilities | ({'tool_calling'} if tools else set()), self.policy, context_tokens=context_tokens)
            usable, retry_dates, cooldown_reasons = [], [], []
            for model in candidates:
                if model.model_id in excluded:
                    denied[model.model_id] = excluded[model.model_id]
                    continue
                adapter = self.gateway.adapters.get(model.provider_id)
                if adapter is None or adapter.automatic_cost != 0:
                    denied[model.model_id] = 'adapter_configuration_or_cost'
                    continue
                observation = self.observation(model.provider_id)
                if not model.available:
                    denied[model.model_id] = 'administratively_unavailable'
                    retry_dates.append(now + self.policy.base_backoff_s)
                elif observation is None or observation['state'] == State.UNKNOWN:
                    denied[model.model_id] = 'health_unknown'
                    retry_dates.append(now + self.policy.base_backoff_s)
                elif observation['cooldown_until'] is not None and observation['cooldown_until'] > now:
                    denied[model.model_id] = 'cooldown'
                    retry_dates.append(observation['cooldown_until'])
                    cooldown_reasons.append(observation['reason'])
                elif observation['cooldown_until'] is not None:
                    usable.append(model)  # Bounded probation after explicit cooldown.
                elif (observation['state'] not in {State.AVAILABLE, State.DEGRADED}
                      or now - observation['observed_at'] > self.policy.observation_ttl_s):
                    denied[model.model_id] = 'health_unavailable_or_stale'
                    retry_dates.append(now + self.policy.base_backoff_s)
                else:
                    usable.append(model)
            self._event(task, 'candidates', {'eligible': [m.model_id for m in usable], 'denied': denied})
            if not usable:
                if retry_dates:
                    reason = ('provider_quota_cooldown' if set(cooldown_reasons) == {'quota_exhausted'}
                              else 'provider_rate_limit_cooldown' if set(cooldown_reasons) == {'rate_limit'}
                              else 'routes_temporarily_unavailable')
                    return self._outcome(task, Disposition.RETRY_LATER, reason,
                                         retry_at=max(now + self.policy.base_backoff_s, min(retry_dates)),
                                         diagnostic=last_diagnostic)
                return self._outcome(task, Disposition.TERMINAL, 'no_eligible_route')
            retry_at = self._reserve(task)
            if retry_at is not None:
                return self._outcome(task, Disposition.RETRY_LATER, 'route_attempt_budget', retry_at=retry_at)
            selected = usable[0]
            self._event(task, 'selected', {'model_id': selected.model_id, 'backend_id': selected.provider_id,
                                         'previous_model_id': previous})
            decision = RouteDecision(task_id=task.task_id, trace_id=task.trace_id, selected_model=selected,
                candidates=(selected.model_id,), reasons=('continuity_eligible',),
                policy_version=self.router.policy_version, automatic_cost=0)
            self.ledger.set_route(decision)
            try:
                response = self.gateway.infer_once(decision, InferenceRequest(model=selected, messages=messages,
                    tools=tools, task_id=task.task_id, trace_id=task.trace_id,
                    max_output_tokens=max_output_tokens, thinking=thinking))
            except InferenceError as exc:
                reason, state, retry = classify(exc)
                last_diagnostic = exc.diagnostic if isinstance(exc.diagnostic, dict) else last_diagnostic
                self._event(task, 'failure', {'model_id': selected.model_id, 'backend_id': selected.provider_id,
                                            'classification': reason,
                                            **({'diagnostic': last_diagnostic} if last_diagnostic is not None else {})})
                if retry:
                    observation = self.observation(selected.provider_id)
                    failures = observation['consecutive_failures'] if observation else 0
                    delay = min(self.policy.max_backoff_s, self.policy.base_backoff_s * 2 ** min(failures, 30))
                    delay = max(delay, exc.retry_after or 0)
                    cooldown_until = self.clock() + delay
                    if exc.retry_at is not None:
                        cooldown_until = max(cooldown_until, exc.retry_at)
                    self.observe(selected.provider_id, state, cooldown_until=cooldown_until,
                                 reason=reason, task=task)
                elif reason in {'invalid_request', 'unclassified_inference_error'}:
                    return self._outcome(task, Disposition.TERMINAL, reason)
                else:
                    excluded[selected.model_id] = reason
                previous = selected.model_id
                continue
            self.observe(selected.provider_id, State.AVAILABLE, task=task)
            response = response.model_copy(update={'fallback': previous is not None})
            return self._outcome(task, Disposition.CONTINUE_NOW, 'inference_succeeded', response=response)
