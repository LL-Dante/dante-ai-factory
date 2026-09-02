"""Deterministic local worker crash fixture; no providers."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dante.agent_host import AgentHostFoundation
from dante.contracts import ToolManifest
from dante.ledger import TaskLedger
from dante.tool_broker import ToolBroker
from dante.worker import Worker


def host_factory(ledger, task):
    root = Path(task.workspace)
    def effect():
        with (root / 'counter.txt').open('a', encoding='utf-8') as stream:
            stream.write('effect\n')
            stream.flush()
            os.fsync(stream.fileno())
        if os.environ.get('DANTE_WORKER_FIXTURE_CRASH') == 'effect':
            os._exit(73)
        return {'ok': True}
    broker = ToolBroker(ledger=ledger)
    broker.register(ToolManifest(tool_id='counter', permissions={'write_workspace'}, risk='R1',
        filesystem_scope=str(root), arguments_schema={'type': 'object'}), effect,
        required_permissions={'write_workspace'})
    return AgentHostFoundation(ledger, None, None, None, broker)


class CrashLedger(TaskLedger):
    def finish_step(self, *args, **kwargs):
        result = super().finish_step(*args, **kwargs)
        if os.environ.get('DANTE_WORKER_FIXTURE_CRASH') == 'verified':
            os._exit(73)
        return result


if __name__ == '__main__':
    ledger = CrashLedger(Path(sys.argv[1]))
    worker = Worker(ledger, host_factory, lease_s=1, poll_s=.05)
    if len(sys.argv) > 2 and sys.argv[2] == 'claim':
        worker.queue.claim(worker.worker_id, 1)
        os._exit(73)
    worker.run_once()
