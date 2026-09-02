"""Offline local CLI: python -m dante --db PATH <command>."""
import argparse
import json
from pathlib import Path
import signal

from dante.agent_host import AgentHostFoundation
from dante.contracts import Task, ToolManifest
from dante.ledger import TaskLedger
from dante.task_queue import ExecutionPlan, TaskQueue
from dante.tool_broker import ToolBroker, workspace_method
from dante.worker import Worker


def offline_host(ledger, task):
    from tools.ai_cloud_workspace import Tools
    tools = Tools()
    if Path(task.workspace).resolve() != tools._root:
        raise ValueError('Task workspace differs from configured workspace')
    broker = ToolBroker(ledger=ledger)
    for method in ('ensure_workspace', 'create_directory', 'write_text_file', 'file_exists', 'list_directory'):
        wrapper = workspace_method(tools, method)
        broker.register(ToolManifest(tool_id='workspace.' + method, permissions={wrapper.permission},
            risk='R0' if wrapper.permission == 'read_workspace' else 'R1', filesystem_scope=str(tools._root)), wrapper)
    return AgentHostFoundation(ledger, None, None, None, broker)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    submit = sub.add_parser('submit')
    submit.add_argument('--goal', required=True)
    submit.add_argument('--plan', type=Path, required=True, help='JSON ExecutionPlan, local tool actions only')
    status = sub.add_parser('status')
    status.add_argument('task_id')
    listing = sub.add_parser('list')
    listing.add_argument('--runnable', action='store_true', help='Include ready and expired recoverable leases')
    cancel = sub.add_parser('cancel')
    cancel.add_argument('task_id')
    for name in ('retry', 'resume'):
        retry = sub.add_parser(name)
        retry.add_argument('task_id')
        retry.add_argument('--delay', type=float, default=1)
    worker = sub.add_parser('worker')
    worker.add_argument('--poll', type=float, default=1)
    worker.add_argument('--lease', type=float, default=30)
    worker.add_argument('--once', action='store_true', help='Process at most one task then exit')
    args = parser.parse_args(argv)
    ledger = TaskLedger(args.db)
    queue = TaskQueue(ledger)
    try:
        if args.command == 'submit':
            from tools.ai_cloud_workspace import Tools
            task = Task(goal=args.goal, workspace=str(Tools()._root))
            plan = ExecutionPlan.model_validate_json(args.plan.read_text(encoding='utf-8-sig'))
            host = offline_host(ledger, task)
            for action in plan.actions:
                if (host.tools.manifest(action.tool_id).version != action.version
                        or host.tools.preflight(task, action.tool_id, action.arguments) is not None):
                    raise ValueError('Plan contains an unsupported local action')
            queue.submit(task, plan)
            result = queue.status(task.task_id)
        elif args.command == 'status':
            result = queue.status(args.task_id)
        elif args.command == 'list':
            result = queue.list(runnable=args.runnable)
        elif args.command == 'cancel':
            queue.cancel(args.task_id)
            result = queue.status(args.task_id)
        elif args.command in {'retry', 'resume'}:
            queue.retry(args.task_id, delay_s=args.delay)
            result = queue.status(args.task_id)
        else:
            instance = Worker(ledger, offline_host, poll_s=args.poll, lease_s=args.lease)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: instance.request_stop())
            if args.once:
                instance.run_once()
            else:
                instance.run()
            result = {'worker_id': instance.worker_id, 'stopped': True}
        print(json.dumps(result))
        return 0
    except (ValueError, KeyError, RuntimeError):
        # Do not echo submitted plans, tool arguments or exception messages.
        print(json.dumps({'error': 'local_command_failed'}))
        return 1
