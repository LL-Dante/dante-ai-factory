# Dante Node 0 supervisor

The supervisor owns the native Ollama server at `http://127.0.0.1:11435` and
reuses the existing `build_node0` composition, qualification ledger, content
addressed evidence, single registry, gate, local only router, inference gateway,
and Ollama adapter. It does not create another qualification database or mark a
runtime trusted because a process is present.

## Readiness and recovery

At start and after runtime recovery the supervisor loads the configured model,
checks the exact reference, digest, quantization, runtime identity, GPU/driver
identity and retained evidence. An unchanged identity with verified hardware
evidence reuses the prior qualification. Missing, stale, failed or corrupt
evidence starts the existing full native qualification attempt while routing
remains denied. Only a fresh `QUALIFIED` result enables the existing registry
gate. The gate also checks supervisor readiness at route selection and at the
inference execution boundary.

The monitor periodically checks the owned process, loopback API, model manifest,
model residency and fresh machine/runtime identity. It has bounded exponential
restart delays and a rolling restart limit. Unknown process ownership, an
occupied endpoint, identity drift, qualification failure or restart limit keeps
local routing closed. Cloud fallback is disabled by the Node 0 `LOCAL_ONLY`
policy.

On Windows, each launched Ollama process is assigned to a kill-on-close Job
Object before the model is loaded. Its child runner is thereby contained in the
same kernel-owned process tree. Shutdown first requests runner closure using
the verified process ancestry and executable path, then reaps the owned parent
and closes the Job Object. Closing the supervisor unexpectedly also closes its
Job Object. An external Ollama process is never adopted or stopped.

The supervisor writes its structured snapshot beside the private Node 0 state
and shares the normal JSONL audit path. Audit rotation retains the active file
and one previous file, each bounded by the configured size. No prompts, model
responses, credentials or environment values are written.

## Registered local agents

The operator control plane exposes a code-registered agent allowlist. The first
entry is `dante-research`, which accepts an objective and optional context,
uses only the currently qualified LOCAL_ONLY model, and stores a validated,
bounded JSON report in the existing durable workload database. Agent jobs share
the supervisor's single workload orchestrator and one-slot GPU gate. Job
payloads retain the agent version and definition digest so an interrupted job
cannot silently resume under changed instructions or model identity. Agent
definitions are data; the control plane does not accept executable code, shell
commands, web access, or model downloads.

From the repository root, use the existing Node 0 named-pipe client:

```powershell
& .\.venv\Scripts\python.exe -m dante.node0_cli agent list
& .\.venv\Scripts\python.exe -m dante.node0_cli agent describe dante-research
& .\.venv\Scripts\python.exe -m dante.node0_cli agent submit dante-research `
  --objective "Analyze the local AI Factory architecture" --output-tokens 256
& .\.venv\Scripts\python.exe -m dante.node0_cli agent job <job_id>
& .\.venv\Scripts\python.exe -m dante.node0_cli agent result <job_id>
```

Use `agent cancel <job_id>` to request cancellation. The result operation is
available only after a successful job; job status and event history remain
available through the same existing control plane and workload store.

## Start manually

From the repository root in PowerShell:

```powershell
$py = '.\.venv\Scripts\python.exe'
$config = Join-Path $env:LOCALAPPDATA 'DanteNode0\qualification-config.json'
& $py -m dante.node0_supervisor --config $config
```

The process stays in the foreground. Ctrl+C requests shutdown. A one-shot startup
simulation uses `--once`; it verifies readiness and then cleanly stops its owned
runtime:

```powershell
& $py -m dante.node0_supervisor --config $config --once
```

## User logon startup

Install or remove the current user's Task Scheduler entry without elevation:

```powershell
& $py -m dante.node0_supervisor --config $config --install-startup
& $py -m dante.node0_supervisor --config $config --remove-startup
```

The task uses the repository `.venv` `pythonw.exe`, a direct repository entry
script, an explicit working directory, interactive current-user logon, one
instance, bounded retry after process failure, and settings that keep the task
running when the session becomes idle or power changes to battery. Arguments
contain paths only, not credentials. The OS mutex is released automatically when
a crashed process exits; a duplicate instance exits without touching Ollama. The
one-shot manual simulation validates the same entry code. A Windows reboot is
still required to prove actual logon startup and reboot persistence.

All runtime, state, ledger, model and evidence paths are local to this Node 0
configuration. Keep the private qualification config and evidence outside Git.
Changes to GPU, NVIDIA driver, Ollama executable/version/configuration, exact
model digest/quantization, context or qualification configuration invalidate
the stored identity and require the existing full qualification again.
