# DANTE AI Factory

DANTE provides persistent, evidence-based task execution with typed inference,
controlled tools and backend continuity. This is a separate source export, not a
clone of the private development repository. Original DANTE material is licensed
under Apache-2.0 following owner confirmation. See LICENSE, LICENSE-DECISION.md
and THIRD_PARTY_NOTICES.md for the scope and third-party boundary.

A private development baseline identified as
`8ce432ba57b55ffb3bb41fe3afb8b4ad125986e4` passed **171/171 tests** and qualification
gates A-K, with verdict QUALIFIED_PRE_WORKSTATION. This export does not have that
Git commit hash. Its independent test result is recorded in EXPORT_QUALIFICATION.md.
Neither result guarantees future changes, machines or real-provider behavior.

## Architecture and capabilities

```text
Persistent task / acceptance -> SQLite ledger + worker queue
                             -> Agent Host
                                -> privacy / registry / router / continuity
                                -> typed inference gateway -> adapters
                                -> Tool Broker -> trusted workspace handlers
                                -> journal / immutable artifacts / acceptance
```

- P0: isolated Python/uv dependencies and reproducible tests.
- P1: typed messages, tool calls/results, usage and inference errors.
- P2: persistent acceptance, crash journal, idempotency and immutable artifacts.
- P3: schema validation, scoped permissions, exact-action one-use approvals,
  output limits and timeout evidence.
- P4: persistent queue, leases, heartbeat, cancellation and process restart.
- P5: qualified model registry and Ollama/llama.cpp adapter contracts.
- P6: backend state, bounded retry/cooldown, route reevaluation and durable deferral.
- P7: node qualification contracts, persistent evidence, identity invalidation and
  an optional registry gate. The software layer is fixture-tested; physical Node 0
  and real local inference remain NOT YET QUALIFIED. See [Node 0 qualification](docs/node-qualification.md).

Action success is not task success: required acceptance evidence must verify.
An uncertain effect requires reconciliation; arbitrary effects are not promised
exactly-once semantics. Local-first is an explicit policy choice. Continuity is
opt-in through DanteConfig.continuity and an injected ContinuityManager; supported
policies include LOCAL_ONLY, LOCAL_FIRST, CLOUD_FIRST_WITH_LOCAL_FALLBACK and
SPECIFIC_ALLOWED_BACKENDS. Health, qualification, privacy, capability, context,
machine and cost gates still apply. Backend outage can defer the same task.

## Install from public files

Use Python 3.12 (qualified version 3.12.13) and uv 0.12.0, provisioned separately.
From this export directory, in PowerShell:

```powershell
uv venv .venv --python 3.12
$py = '.\.venv\Scripts\python.exe'
uv pip sync --python $py requirements-dante.lock
uv pip check --python $py
& $py -c "import yaml, dante; print('imports OK')"
```

On a POSIX system the interpreter path is .venv/bin/python; the current qualification
is Windows-specific. Check command exit statuses. --offline can be added to uv sync
if all artifacts are cached. No globally installed Python packages are required.
The public lock pins all 22 distributions; root requirements match pyproject.toml.
It is a version lock, not a wheel-hash lock. Different platforms/artifacts require
revalidation. Optional services and model weights are not installed by these steps.

## Offline CLI example

The CLI executes persistent local plans; it is not an inference-driven planning
loop. Use workspace and audit paths you control. A submitter may exit before a
separate worker starts.

```powershell
$py = '.\.venv\Scripts\python.exe'
$env:PYTHON_DOTENV_DISABLED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:AI_CLOUD_WORKSPACE = Join-Path $env:USERPROFILE 'Documents\AI-Cloud-Workspace'
$demo = Join-Path $env:LOCALAPPDATA 'DANTE-demo'
New-Item -ItemType Directory -Force $demo | Out-Null
$env:AI_CLOUD_AGENT_AUDIT_LOG = Join-Path $demo 'tools.jsonl'
$db = Join-Path $demo 'tasks.db'
$plan = Join-Path $demo 'plan.json'
'{"actions":[{"tool_id":"workspace.write_text_file","arguments":{"relative_path":"demo.txt","content":"DANTE local task"}}]}' | Set-Content $plan -Encoding utf8
& $py -m dante --db $db submit --goal 'Write local demo' --plan $plan
& $py -m dante --db $db worker --once
& $py -m dante --db $db list
```

Additional commands: status TASK_ID, cancel TASK_ID, retry TASK_ID --delay 5.
Retries do not bypass reconciliation or approval. Store operational data outside
source control. config/dante/base.yaml is a neutral example; configure workspace,
privacy and model eligibility explicitly before deployment. No real model is
pre-approved or shipped. .env.example contains only an optional gateway key slot.

## Tests

```powershell
$env:PYTHON_DOTENV_DISABLED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All 171 original tests are intentionally distributable synthetic/local fixtures:
foundation 16, P1 17, P2 22, P3 24, P4 21, P5 33, P6 38. No test is removed or
weakened. They include subprocess crash/restart and continuity tests and do not
require external AI calls or model downloads. They do not qualify a real model.

P7 adds 82 focused tests; the current implementation passed 253/253 offline tests.
See [P7 verification and hardware procedure](docs/node-qualification.md) for scope
and limitations. No GPU, CUDA, runtime server or model weights are required.

The Node 0 bootstrap creates or reuses a persistent UUID, records a conservative
machine profile and reports `NOT_YET_QUALIFIED` until real hardware and runtime
evidence exists. The exact local command is documented in the P7 guide.
`node-qualify` persists an attempt and defaults to `UNKNOWN` because this repository
does not ship a real hardware/runtime probe. Reviewed probes can be injected through
the typed harness interface without making a runtime authoritative.
Machine inspection can collect NVIDIA model, VRAM, driver and compute capability
through bounded `nvidia-smi` queries. Tool absence is a supported unknown state;
the probe does not benchmark or qualify CUDA, Vulkan, Ollama or llama.cpp. An
observation-only Ollama probe reads the configured endpoint's version and model
inventory through deterministic, bounded adapter calls. It never downloads or runs a
model, and availability is not qualification.

The equivalent llama.cpp probe reads health, model state and optional build metadata.
Its backend field accepts explicit CPU, CUDA or Vulkan observations/configuration
without ranking them; missing metadata remains unknown.

## Third-party boundary and limitations

DANTE does not import or require cptr/Open WebUI. The private stack's deployment,
service-client and upstream patch scripts are excluded. Optional independently
installed services remain under their own licenses; see THIRD_PARTY_NOTICES.md.
The retained LiteLLM adapter communicates over HTTP with a separately configured
gateway. No cptr source, binary, patch fragment, branding or asset is included.

- Trusted in-process Python handlers are not an OS sandbox. Thread timeout cannot
  forcibly terminate arbitrary code; uncertain effects may require operator review.
- Keep gateway/runtime services on loopback and use a trusted local account.
- Real local model inference, real llama.cpp runtime and workstation hardware have
  not been qualified. vLLM is not qualified or included.
- Unknown usage/cost stays unknown. Gemini remains COST_UNVERIFIED and automatic
  continuity routing denies unverified cloud cost and paid routes.
- Health or installed-model inventory does not establish model qualification.
- No real-hardware qualification harness, RAG, Control API, Media Factory or
  workstation-specific tuning is included. P7 inventory alone grants no approval.

See SECURITY.md, CONTRIBUTING.md and LICENSE-DECISION.md before redistribution.
The canonical public source is [LL-Dante/dante-ai-factory](https://github.com/LL-Dante/dante-ai-factory).
SOURCE_MANIFEST.json and the export/provenance reports are historical evidence for
the initial public export, not current-file integrity manifests after P7 changes.

GitHub Private Vulnerability Reporting is enabled. Use the
[private reporting form](https://github.com/LL-Dante/dante-ai-factory/security/advisories/new).
Do not disclose vulnerabilities in public issues.
