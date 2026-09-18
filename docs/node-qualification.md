# P7 — Node 0 qualification layer

Status: the qualification **software layer** is tested with local fixtures. Node 0,
RTX 5080, real Ollama/llama.cpp inference, CUDA and Vulkan remain **NOT YET QUALIFIED**.
No benchmark numbers or compatibility verdicts are supplied for real hardware.

## Architecture and scope

DANTE remains the control layer. Existing tasks, acceptance, Tool Broker, worker,
recovery, gateway and continuity retain their responsibilities. P7 adds evidence
accounting and an optional additional local-model registry gate. A healthy endpoint,
installed model or model lifecycle approval is not a node qualification.

- `MachineProfile` records a provisioned random node UUID and selected machine facts.
  GPU inventory supports multiple slots; an unprobed inventory is `null`, while an
  observed CPU-only machine uses an empty list. Hostnames, serials and usernames
  are not collected. The built-in probe collects only OS/version, architecture,
  CPU/logical CPU information and Windows physical RAM. When `nvidia-smi` is on
  PATH, the NVIDIA probe also records GPU index/model, VRAM, driver version and
  compute capability where supported. Physical cores and storage await dedicated probes.
- Existing `RuntimeProfile` remains transport configuration. `RuntimeObservation`
  records runtime identity/version/backend, configuration and optional binary
  digests, reported capabilities and sanitized inventory aliases.
- `QualificationIdentity` binds machine, runtime, model alias, artifact digest and
  kind, quantization, context, configuration, schema and suite version.
  A runtime manifest digest is explicitly distinguished from a model-file digest.
- `CheckEvidence` and `PerformanceEvidence` carry bounded typed outcomes and optional
  measurements. Missing performance fields stay `null`; no throughput or cost is
  inferred. Raw provider replies and arbitrary exception text are not persisted.
- `ModelQualification` stores the identity snapshot, source (`synthetic`/`hardware`),
  checks, timestamps and measurements. `QualificationStore` uses TaskLedger's SQLite
  connections. `QualificationRunner` accepts a trusted injected `QualificationProbe`.

No dependency, runtime installation, GPU driver, model weight or external service
is added. Discovery uses explicit registration, never PATH scanning or service startup.
`adapter_inventory` can reuse the existing Ollama and llama.cpp inventory methods
when explicitly invoked; its tests inject HTTP fixtures. It discards raw inventory
metadata and exposes only caller-selected safe aliases. Versions and backend
configuration must come from future runtime-specific observations, not guessed
from an endpoint name.

The NVIDIA inventory runs bounded local `nvidia-smi` commands without a shell. Its
structured CSV queries collect `index`, `name`, `memory.total`, `driver_version` and,
separately, `compute_cap`. Output is capped and malformed or incomplete base inventory
fails to unknown. If the executable is absent or cannot run, `gpus` remains `null`;
an explicit “no devices” response becomes an empty GPU list. Optional compute data
can remain null without discarding valid base inventory. The default display's
`CUDA Version` is recorded as `driver-supported-X.Y` in `cuda_runtime`: this is only
the driver's advertised CUDA API compatibility, not proof of an installed CUDA
runtime or toolkit. `cuda_toolkit` remains null. The probe performs no benchmark,
model execution, runtime qualification or backend preference.

The Ollama probe uses an explicit `RuntimeProfile` endpoint (defaulting to the
loopback origin `http://127.0.0.1:11434`) and the existing bounded HTTP adapter. It
requests only `/api/version` and `/api/tags`. The result distinguishes available,
unavailable and invalid/error states, retains observed model references in the
Ollama-specific evidence, and maps only opaque model aliases into `RuntimeObservation`.
It does not start Ollama, scan ports, pull models, invoke inference or create
qualification evidence.

The llama.cpp probe uses an explicit `RuntimeProfile` endpoint (defaulting to
`http://127.0.0.1:8080`). It requires the existing `/health` and `/v1/models`
observations and reads `/props` only when the server provides it. Version and build
remain unknown when absent. Backend is restricted to an explicitly reported or
configured `cpu`, `cuda` or `vulkan` value; conflicting or unknown values fail closed,
and no backend is ranked above another. The probe performs no model load, generation,
benchmark or qualification.

## Lifecycle and evidence boundary

| State | Meaning |
|---|---|
| `unknown` | No record, incomplete run/checks/identity, unsupported suite, or synthetic-only evidence |
| `qualified` | Current exact identity, hardware-labelled evidence, all mandatory checks passed, completed run |
| `failed` | A check failed, a probe raised an exception, or identity changed during execution |
| `stale` | Stored identity differs from the current observed identity |

The versioned `node0-v1` suite requires load, generation, context, cancellation,
timeout, runtime failure, recovery and stability checks. Structured output and tool
calling are separate capability checks; a tool route additionally requires a passed
tool-calling check. A recorded optional failure makes that qualification fail;
`unsupported` does not establish a capability. Passing checks require evidence digests.

The runner appends an incomplete snapshot before starting checks and a final
snapshot afterwards. A crash leaves `unknown`; a caught exception records a safe
failure without its message. A failed or incomplete newer run takes precedence over
an older successful run. History remains append-only through the application API.
Run one qualification harness at a time per tuple; P7 does not add a probe scheduler
or leases for concurrent qualification runs. Stop an interrupted harness before retrying.

Probe implementations are trusted local code, not an attestation authority or an OS
sandbox. The software verifies contracts, payload integrity and identity matching;
it cannot prove that a caller telling it `source='hardware'` actually measured hardware.
The tests construct such metadata solely to verify verdict logic. Production probes
must retain the underlying local evidence matching each digest, implement bounded
execution and cancellation, and verify the named check's assertions. P7 does not
yet ship that real-runtime measurement harness or a raw-evidence retention service.
The ledger stores the typed evidence summaries; external evidence digests are not
automatically resolved or rehashed. Do not manually turn fixture records into approvals.

## Invalidation and persistence

Semantic identity inputs include node identity, OS/CPU/RAM/GPU facts, GPU driver,
observed CUDA environment, runtime type/version/backend/binary/configuration,
model alias/artifact/hash/quantization/context/configuration, probe version and
qualification schema/test-suite version. A difference returns `stale` with field
names explaining why. Historical evidence is not rewritten. No evidence for a new
node/backend/model scope returns `unknown`; `assess(old_record, current_identity)`
can explain staleness across scopes explicitly.

Observation timestamps, current free disk space, GPU enumeration ordering, reported
capability ordering and unrelated discovered models do not invalidate evidence.
The hardware fingerprint is conservative: installed GPU or CUDA-environment changes
invalidate prior evidence even if a later harness determines they do not affect a
particular backend. There is no time-based guarantee; callers must supply fresh
observations, especially after changes. A stale input JSON cannot detect hardware drift.
Missing CUDA runtime information prevents qualifying a CUDA tuple. Unavailable
optional facts remain unknown rather than claiming a toolkit is installed.

Configuration fingerprints must hash a reviewed, complete allowlist of execution
settings (including generation/sampling, offload, threading, context and backend
options where applicable). Exclude credentials, arbitrary environment dumps and
personal paths. A hash of an incomplete configuration does not establish reproducibility.
Use `dante.recovery.digest` for canonical JSON hashing. Human-facing fields accept
bounded labels, not URLs or paths; model/provider aliases must match registry IDs.

SQLite schema 3→4 adds `node_profiles`, `model_qualifications` and a scope index in
the existing migration transaction. It preserves prior tasks, events, revisions,
approvals, journals, queue and continuity data. Repeated initialization is safe;
unknown future database/contract versions fail closed. Back up an existing database
before upgrading: older code rejects schema 4, and automatic downgrade is not provided.
The three existing migration tests change only their expected final schema number.

## Registry integration

Construct the Node 0 registry with
`qualification_gate=NodeQualificationGate(store, current_identity_for_model)`.
The callback must obtain a current `QualificationIdentity` for each requested model,
including the selected backend/configuration, rather than return historical evidence.
It must reject unsupported/unobservable setups. The gate checks model/provider/runtime,
artifact, quantization and context binding before consulting the latest evidence.
License, lifecycle, machine eligibility, file integrity, privacy and cost gates remain
required. It never promotes a model or changes model metadata.

This hook is opt-in to preserve the qualified P0–P6 APIs. The Node 0 composition must
install it on the registry shared by router and gateway. Direct adapter calls and
registries constructed without it retain P5 behavior; they do not acquire P7
qualification enforcement. The offline tool-worker CLI does not configure inference.

## First procedure on Node 0

First install the declared environment and run the offline regression from README.
Then run the minimal bootstrap. It creates a random Node UUID once, reuses it on
later runs, invokes the P7 machine inspection and stores each observed profile:

```powershell
$py = '.\.venv\Scripts\python.exe'
$state = Join-Path $env:LOCALAPPDATA 'DANTE-Node0'
New-Item -ItemType Directory -Force $state | Out-Null
$idFile = Join-Path $state 'node-id.txt'
$db = Join-Path $state 'qualification.db'
& $py -m dante --db $db node-bootstrap --node-id-file $idFile
```

Expected output includes `qualification_state: unknown` and
`qualification_status: NOT_YET_QUALIFIED`, with missing hardware/runtime evidence
listed as reasons. Bootstrap and `node-inspect` are inventory commands, not hardware
benchmarks. Keep the UUID file stable across restarts and assign a new one for a
different physical node. A malformed existing UUID file fails closed and is never
replaced automatically. Keep the UUID, database and evidence outside the repository.

After a future trusted harness collects a complete current identity and evidence,
the same CLI flow can persist a qualification attempt:

```powershell
& $py -m dante --db $db node-qualify --node-id-file $idFile `
  --identity (Join-Path $state 'current-identity.json')
& $py -m dante --db $db node-status --identity (Join-Path $state 'current-identity.json')
```

The repository intentionally ships no real runtime probe. With the default
observation-only probe, `node-qualify` records `intent` and `completed` lifecycle
entries and returns `UNKNOWN`; every check remains `unknown` with no evidence digest.
A future trusted Node 0 composition injects `QualificationProbeFactory` and a richer
machine probe through `NodeQualificationHarness`. No module loading, runtime startup
or executable discovery occurs from CLI arguments. Injected probes use the existing
P7 `QualificationProbe` contract and must return bounded evidence for the exact
observed identity. The result reports `UNKNOWN`, `QUALIFIED`, `FAILED` or `STALE`
and includes safe reason codes. Process interruption leaves the durable intent so
restart inspection cannot mistake an older pass for the interrupted attempt.

`node-history` returns durable snapshots with runtimes/models/backend/configuration,
timestamps and failed check reasons. Assess each against current observations to
identify still-qualified or stale entries. CLI exit status 0 means the query worked;
inspect the returned `state`, which may be `unknown`, `failed` or `stale`.

Before any real model is enabled, the remaining work is:

1. Observe actual GPU/VRAM/driver and applicable runtime/toolkit facts with reviewed probes.
2. Pin the installed runtime build, backend, model artifact/license and full configuration.
3. Implement and validate the bounded hardware probe harness and evidence retention.
4. Measure the suite and stability over repeated runs. Measure memory, load time and
   prompt/generation rates; thermal data is optional until a reliable probe exists.
5. Compare llama.cpp CUDA and Vulkan as independent matrix entries when available;
   measure Ollama independently. No ordering or compatibility is presumed.
6. Reobserve the identity, review evidence, wire the Node 0 registry gate and retain
   the existing model approval process. Never mark a missing measurement as zero.

vLLM, MCP, A2A, distributed execution, RAG, Control API and Media Factory remain
unimplemented. Runtime labels and the injected probe interface allow future additions
without making a provider authoritative.

## Verification of this implementation

On Windows with Python 3.12.13 and the existing declared environment, the full suite
passed **280/280** tests: original P0–P6 **171**, P7 **82**, P8 **17**, hardening
audits **10**, no skips.
This is evidence for the reviewed working changes based on public commit
`5de6869d8d4cfbf81a72b5df2eea31188d37ca23`, not qualification of physical Node 0.
P7 coverage includes validation, multiple GPUs, selective invalidation, version
changes, restart persistence, database migration, corrupt evidence, interrupted runs,
synthetic/hardware separation, runtime discovery fixtures, gateway denial before HTTP,
persistent bootstrap identity, concurrent initialization and explicit unqualified status.
The qualification harness coverage adds durable attempt lifecycle, restart evidence,
legacy-record compatibility, injected probe boundaries and all four verdict states.
NVIDIA coverage uses command fixtures for multiple GPUs, VRAM conversion, driver and
compute data, the shell-free process boundary, bounded malformed output, missing tools,
command failure and CPU-only fallback.
Ollama coverage uses HTTP fixtures for version and inventory discovery, explicit
endpoint configuration, empty inventory, unavailable and timeout states, and malformed
version, reference and digest responses. No test requires an Ollama installation.
llama.cpp coverage uses HTTP fixtures for health, inventory, optional properties,
version/build metadata, endpoint configuration, CPU/CUDA/Vulkan representation,
backend conflicts, malformed responses, timeout and absence. No test requires a
llama.cpp installation.
