# C:\DanteAI — Dante AI Factory workspace

Repo root and permanent local workspace for `dante-ai-factory`.
Remote: `https://github.com/LL-Dante/dante-ai-factory.git`
Branch of record: `codex/workload-orchestrator`

## CAVEMAN MODE (permanent operating rules)

- Think short. Act directly.
- No repeated analysis. State a finding once.
- Read only the files needed for the current task.
- Before editing: plan of at most 5 lines.
- Prefer minimal targeted edits. Never rewrite whole files unless necessary.
- If an edit or test fails, inspect the exact failure before retrying. Never blind-retry.
- Run focused tests after meaningful changes.
- Run the full suite before commit when appropriate.
- Never bypass qualification, safety, or control-plane boundaries.
- Never start, stop, or restart production runtime without explicit permission.
- Never install software or download models without explicit permission.
- Never commit or push without explicit permission.
- Preserve context. Avoid unnecessary output.

## Final report format (always end with these five blocks)

```
CHANGED
TESTS
PASS/FAIL
PROBLEMS
NEXT
```

## Layout

- `dante/` — core package: control plane, workload orchestrator, node0 supervisor/CLI, contracts, agent runner
- `tests/` — stdlib `unittest` suite (438 tests, 1 skipped)
- `docs/` — operational docs, including `node0-supervisor.md`
- `config/` — tracked configuration
- `monitoring/` — local monitoring subsystem (source is tracked, runtime is not)
- `scripts/`, `tools/` — operator scripts and tooling
- `migration-backup/` — LOCAL ONLY, never commit. Holds the Codex sandbox snapshot and backups.

## Build, test, verify

There is no `python` on PATH on this machine. Use a virtualenv interpreter explicitly.

```powershell
# one-time, requires explicit permission (installs packages)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dante.in

# compile / syntax check
.\.venv\Scripts\python.exe -m compileall -q dante tests

# focused tests (note -t tests: several tests import sibling modules by bare name)
.\.venv\Scripts\python.exe -m unittest discover -s tests -t tests -p "test_workload.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -t tests -p "test_agents.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -t tests -p "test_node0_control.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -t tests -p "test_node0_supervisor.py"

# full suite
.\.venv\Scripts\python.exe -m unittest discover -s tests -t tests -p "test_*.py"
```

`pytest` is not installed and is not required. The suite is stdlib `unittest`.

## Git hygiene

Tracked: `monitoring/` source, config, dashboard, scripts, tests.
Ignored, local only — never stage these:

- `monitoring/logs/`, `monitoring/state/` (runtime output and state)
- `__pycache__/`, `*.pyc`
- `migration-backup/`
- `opencode.json`, `Modelfile.agent`, `agent-test.txt`

`AGENTS.md` and `.gitignore` are tracked. Local changes to the last two are expected
and intentional.

## Operational boundaries

- The production supervisor must be restarted by a human before new control-plane
  operations load into the running process. Agents never restart it.
- Qualification, safety, and control-plane gates are not bypassable for convenience.
- Uncommitted work exists in the old Codex sandbox at
  `C:\Users\dante\Documents\Codex\2026-09-23\files-pasted-by-the-user-you\dante-ai-factory`.
  It is a snapshot source, not a workspace. Never delete it.
