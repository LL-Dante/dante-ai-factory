"""Local process-crash fixture. Never imports provider adapters."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dante.agent_host import AgentHostFoundation
from dante.ledger import TaskLedger
from dante.tool_broker import ToolBroker
from dante.contracts import ToolManifest
from dante.recovery import ReconciliationRequired

root, task_id, window = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


class CrashLedger(TaskLedger):
    def prepare_step(self, *args, **kwargs):
        result = super().prepare_step(*args, **kwargs)
        if window == 'A':
            os._exit(71)
        return result

    def claim_step(self, *args, **kwargs):
        super().claim_step(*args, **kwargs)
        if window == 'B':
            os._exit(71)

    def finish_step(self, *args, **kwargs):
        result = super().finish_step(*args, **kwargs)
        if window == 'D':
            os._exit(71)
        return result


def effect(value):
    with (root / 'counter.txt').open('a', encoding='utf-8') as stream:
        stream.write(value + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if window == 'C':
        os._exit(71)
    return {'ok': True}


ledger = CrashLedger(root / 'tasks.db')
if window == 'inspect':
    print(ledger.get_acceptance(task_id).model_dump_json())
    sys.exit(0)
broker = ToolBroker()
broker.register(ToolManifest(tool_id='counter', permissions={'write_workspace'}, risk='R1', filesystem_scope=str(root), arguments_schema={'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']}), effect, required_permissions={'write_workspace'})
host = AgentHostFoundation(ledger, None, None, None, broker)
host.resume(task_id)
try:
    result = host.execute_tool_and_checkpoint(task_id, 'counter', {'value': 'effect'}, idempotency_key='once')
    print(json.dumps(result))
except ReconciliationRequired:
    print('RECONCILIATION_REQUIRED')
    sys.exit(3)
