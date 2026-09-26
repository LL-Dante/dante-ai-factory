"""Operator commands for the already-running production Node 0 supervisor."""
from __future__ import annotations

import argparse
import json
import sys

from dante.node0_control import ControlError, Node0ControlClient
from dante.contracts.inference import DEFAULT_MAX_OUTPUT_TOKENS


def _print(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _workload_parser(sub):
    workload = sub.add_parser("workload", help="Durable LOCAL_INFERENCE jobs")
    actions = workload.add_subparsers(dest="workload_action", required=True)
    submit = actions.add_parser("submit")
    submit.add_argument("--prompt", required=True)
    submit.add_argument("--model-id", required=True)
    submit.add_argument("--runtime-reference", required=True)
    submit.add_argument("--digest", required=True)
    submit.add_argument("--context", type=int, required=True)
    submit.add_argument("--max-output-tokens", type=int, default=128)
    submit.add_argument("--timeout", type=float, default=120)
    submit.add_argument("--deadline")
    submit.add_argument("--max-attempts", type=int, default=2)
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--idempotency-key")
    for name in ("status", "result", "cancel", "events"):
        command = actions.add_parser(name)
        command.add_argument("job_id")
    listing = actions.add_parser("list")
    listing.add_argument("--limit", type=int, default=100)
    actions.add_parser("orchestrator-status")
    return workload


def _agent_parser(sub):
    agent = sub.add_parser('agent', help='Run registered local-only agents')
    actions = agent.add_subparsers(dest='agent_action', required=True)
    actions.add_parser('list')
    describe = actions.add_parser('describe')
    describe.add_argument('agent_id')
    submit = actions.add_parser('submit')
    submit.add_argument('agent_id')
    submit.add_argument('--objective', required=True)
    submit.add_argument('--context')
    submit.add_argument('--output-tokens', type=int)
    submit.add_argument('--idempotency-key')
    for name in ('job', 'result', 'cancel'):
        command = actions.add_parser(name)
        command.add_argument('job_id')
    return agent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Current in-memory supervisor and qualification state")
    sub.add_parser("health", help="Live runtime, model, and GPU residency health")
    infer = sub.add_parser("infer", help="Run one LOCAL_ONLY inference through Node 0")
    infer.add_argument("--model", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    infer.add_argument("--thinking", choices=("off", "on"), default="off")
    _workload_parser(sub)
    _agent_parser(sub)
    args = parser.parse_args(argv)
    try:
        client = Node0ControlClient()
        if args.command == "status":
            result = client.request({"op": "status"})
        elif args.command == "health":
            result = client.request({"op": "health"})
        elif args.command == "infer":
            result = client.request({"op": "infer", "model": args.model, "prompt": args.prompt,
                                     "max_output_tokens": args.max_output_tokens,
                                     "thinking": args.thinking})
        else:
            result = (_workload_request(client, args) if args.command == 'workload'
                      else _agent_request(client, args))
        _print(result)
        return 0
    except ControlError as exc:
        result = {"error": exc.code}
        if exc.diagnostic is not None:
            result['diagnostic'] = exc.diagnostic
        _print(result)
        return 2
    except (OSError, TimeoutError):
        _print({"error": "node0_control_unavailable"})
        return 2
    except (ValueError, TypeError):
        _print({"error": "invalid_operator_request"})
        return 2


def _workload_request(client, args):
    action = args.workload_action
    if action == "submit":
        spec = {
            "job_type": "LOCAL_INFERENCE",
            "model": {"model_id": args.model_id, "runtime_reference": args.runtime_reference,
                      "digest_sha256": args.digest, "context_tokens": args.context,
                      "local_only": True},
            "prompt": args.prompt,
            "max_output_tokens": args.max_output_tokens,
            "timeout_s": args.timeout,
            "maximum_attempts": args.max_attempts,
            "priority": args.priority,
        }
        if args.deadline:
            spec["deadline_at"] = args.deadline
        return client.request({"op": "workload_submit", "spec": spec,
                               "idempotency_key": args.idempotency_key})
    if action == "list":
        return client.request({"op": "workload_list", "limit": args.limit})
    if action == "orchestrator-status":
        return client.request({"op": "orchestrator_status"})
    operation = {"status": "workload_status", "result": "workload_result",
                 "cancel": "workload_cancel", "events": "workload_events"}[action]
    return client.request({"op": operation, "job_id": args.job_id})


def _agent_request(client, args):
    action = args.agent_action
    if action == 'list':
        return client.request({'op': 'agent_list'})
    if action == 'describe':
        return client.request({'op': 'agent_describe', 'agent_id': args.agent_id})
    if action == 'submit':
        request = {'op': 'agent_submit', 'agent_id': args.agent_id,
                   'objective': args.objective}
        if args.context is not None:
            request['context'] = args.context
        if args.output_tokens is not None:
            request['requested_output_tokens'] = args.output_tokens
        if args.idempotency_key is not None:
            request['idempotency_key'] = args.idempotency_key
        return client.request(request)
    operation = {'job': 'agent_job', 'cancel': 'agent_job_cancel',
                 'result': 'agent_result'}[action]
    return client.request({'op': operation, 'job_id': args.job_id})


if __name__ == "__main__":
    raise SystemExit(main())
