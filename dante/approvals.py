from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from dante.contracts import Approval
from dante.ledger import TaskLedger


class ApprovalManager:
    def __init__(self, ledger: TaskLedger) -> None:
        self.ledger = ledger

    def request(self, task_id: str, action: str, target: str, plan_hash: str) -> tuple[Approval, str]:
        approval = Approval(task_id=task_id, action=action, target=target, plan_hash=plan_hash)
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with closing(self.ledger._connect()) as connection, connection:
            connection.execute("INSERT INTO approvals(approval_id,task_id,payload,token_hash) VALUES(?,?,?,?)", (approval.approval_id, task_id, approval.model_dump_json(), token_hash))
        return approval, token

    def decide(self, approval_id: str, token: str, plan_hash: str, actor: str, *, approve: bool) -> Approval:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with closing(self.ledger._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload,token_hash,consumed FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
            if not row or row["consumed"] or not secrets.compare_digest(row["token_hash"], token_hash):
                raise PermissionError("Approval token is invalid or consumed")
            approval = Approval.model_validate_json(row["payload"])
            if approval.plan_hash != plan_hash:
                raise PermissionError("Approval plan hash changed")
            approval.status = "approved" if approve else "rejected"
            approval.decided_at = datetime.now(timezone.utc)
            approval.decided_by = actor
            connection.execute("UPDATE approvals SET payload=?,consumed=1 WHERE approval_id=? AND consumed=0", (approval.model_dump_json(), approval_id))
        return approval


    def request_tool(self, broker, task, tool_id: str, arguments: dict, *, plan_hash: str = ''):
        if broker.preflight(task, tool_id, arguments) is not None:
            raise PermissionError('Action is not admissible')
        binding = broker.action_hash(task, tool_id, arguments, plan_hash)
        return self.request(task.task_id, 'tool.execute', binding, binding)

    def consume_action(self, approval_id: str, task_id: str, action_hash: str) -> None:
        """Decision token and execution grant are separate one-use transitions."""
        with closing(self.ledger._connect()) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT payload,consumed FROM approvals WHERE approval_id=?',
                                     (approval_id,)).fetchone()
            if row is None or not row['consumed']:
                raise PermissionError('Approval is not decided')
            approval = Approval.model_validate_json(row['payload'])
            if (approval.status != 'approved' or approval.task_id != task_id
                    or approval.action != 'tool.execute' or approval.target != action_hash
                    or approval.plan_hash != action_hash):
                raise PermissionError('Approval does not authorize this action')
            approval.status = 'executed'
            connection.execute('UPDATE approvals SET payload=? WHERE approval_id=?',
                               (approval.model_dump_json(), approval_id))
