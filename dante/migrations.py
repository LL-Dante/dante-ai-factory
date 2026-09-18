"""P2 additive migration for the unversioned foundation database."""
import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute('BEGIN IMMEDIATE')
    try:
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version > 4:
            raise RuntimeError('Unsupported DANTE database schema')
        if version == 0:
            connection.execute('''CREATE TABLE task_acceptance (
                task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
                contract TEXT NOT NULL, strict INTEGER NOT NULL, locked INTEGER NOT NULL,
                attestations TEXT NOT NULL DEFAULT '{}', final_evidence TEXT NOT NULL DEFAULT '{}'
            )''')
            connection.execute('''INSERT INTO task_acceptance(task_id,contract,strict,locked)
                SELECT task_id, '{}', 0, 0 FROM tasks''')
            connection.execute('''CREATE TABLE execution_steps (
                step_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
                tool_id TEXT NOT NULL, tool_version TEXT NOT NULL, arguments_digest TEXT NOT NULL,
                idempotency_key TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL CHECK(state IN ('intent','started','succeeded','uncertain')),
                created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
                result TEXT, result_digest TEXT, evidence_ref TEXT, error TEXT,
                UNIQUE(task_id,idempotency_key)
            )''')
            connection.execute('PRAGMA user_version=1')
        if version < 2:
            connection.execute('''CREATE TABLE task_queue (
                task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
                plan TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'ready'
                    CHECK(state IN ('ready','leased','blocked','done')),
                next_action INTEGER NOT NULL DEFAULT 0,
                worker_id TEXT, generation INTEGER NOT NULL DEFAULT 0,
                claimed_at REAL, lease_expires_at REAL, heartbeat_at REAL,
                attempt INTEGER NOT NULL DEFAULT 0, retry_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0, last_failure TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            )''')
            connection.execute('CREATE INDEX task_queue_runnable ON task_queue(state,next_attempt_at)')
            connection.execute('PRAGMA user_version=2')
        if version < 3:
            connection.execute('''CREATE TABLE backend_observations (
                backend_id TEXT PRIMARY KEY, state TEXT NOT NULL, reason TEXT NOT NULL,
                observed_at REAL NOT NULL, cooldown_until REAL,
                consecutive_failures INTEGER NOT NULL DEFAULT 0, last_success REAL
            )''')
            connection.execute('''CREATE TABLE task_continuity (
                task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), window_start REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL, last_reason TEXT,
                disposition TEXT
            )''')
            connection.execute('PRAGMA user_version=3')
        if version < 4:
            connection.execute('''CREATE TABLE IF NOT EXISTS node_profiles (
                snapshot_id TEXT PRIMARY KEY, node_id TEXT NOT NULL, payload TEXT NOT NULL
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS model_qualifications (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                qualification_id TEXT NOT NULL UNIQUE, node_id TEXT NOT NULL,
                scope TEXT NOT NULL, payload TEXT NOT NULL, payload_digest TEXT NOT NULL
            )''')
            connection.execute('CREATE INDEX IF NOT EXISTS qualification_scope ON model_qualifications(scope,sequence)')
            connection.execute('PRAGMA user_version=4')
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
