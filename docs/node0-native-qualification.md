# Node 0 native Windows qualification

This path is implemented but has **not** qualified the physical RTX 5080. `qwen3:4b` is a planned first deployment model, not an installed or qualified model. The normal suite uses fixtures only.

## Architecture and safety

`build_node0` in `dante/node0.py` constructs one ledger, strict evidence verifier, qualification store, `NodeQualificationGate`, `ModelRegistry`, router, inference gateway, Ollama adapter, continuity manager, and `AgentHostFoundation`. The router, gateway, and adapter share the same registry. The gateway and adapter recheck qualification at execution, including after a route was selected. LOCAL_ONLY allows no cloud fallback. Other explicit continuity policies retain the existing provider quota behavior.

Administrative model approval and an inventory observation do not prove that the current GPU can run the exact model. A physical attempt must retain evidence for every required check and establish the exact node, GPU UUID and driver, runtime version and configuration, model digest and quantization, and context. `UNKNOWN`, `STALE`, `FAILED`, missing evidence, corrupt evidence, or any changed identity blocks automatic routing. Tool calling requires separate measured capability; the eight checks do not authorize it.

The evidence store writes content-addressed JSON documents and the SQLite ledger records each attempt, including its intent before preflight. A failed rerun supersedes an earlier pass. Model, driver, GPU, runtime, binary, context, or qualification configuration changes require another full attempt. The gate observes identity again when routing and executing.

## Physical workflow (later deployment only)

Install native Ollama and an exact local model only after separate authorization. Provision a stable random node UUID. Use a dedicated loopback Ollama endpoint and a dedicated model store. The qualification process must own the `ollama.exe serve` process that it starts; it refuses an occupied endpoint. It stops only that owned process for the failure/recovery checks. It does not kill an existing Ollama service. An external service cannot pass the destructive failure/recovery checks.

Create a private JSON config with these explicit fields. `model` is a complete `ModelRef` document whose local metadata has the exact runtime reference, digest, quantization, context, machine profile, approved lifecycle, verified license, and evaluation references. The values below are **placeholders**, not observed hardware facts:

```json
{
  "node_uuid": "REPLACE-WITH-STABLE-UUID",
  "machine_profile": "node0",
  "ledger_path": "C:/node0-private/qualification.db",
  "evidence_path": "C:/node0-private/evidence",
  "audit_path": "C:/node0-private/audit.jsonl",
  "runtime": {
    "endpoint": "http://127.0.0.1:11435",
    "model_reference": "EXACT-LOCAL-MODEL:TAG",
    "model_digest": "EXACT-64-HEX-DIGEST",
    "quantization": "EXACT-QUANTIZATION",
    "context_tokens": 4096,
    "gpu_uuid": "GPU-EXACT-UUID",
    "executable": "C:/path/to/ollama.exe",
    "model_store": "C:/path/to/dedicated/models"
  },
  "model": { "...": "complete approved ModelRef" }
}
```

Run only when explicitly ready for a physical attempt:

```powershell
$env:DANTE_NODE0_HARDWARE='1'
$env:DANTE_NODE0_CONFIG='C:\node0-private\config.json'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_node0_hardware.py -v
```

Or invoke `python -m dante.node0_hardware --config <private-config-path> --run-hardware`. Both the CLI flag and environment opt-in are required. The command reports safe machine-readable JSON and exits nonzero on failure. It never downloads a model. Keep the config and retained evidence out of Git. The harness performs qualification, then attempts a LOCAL_ONLY host inference through the router, gate, gateway, and Ollama adapter. A PASS requires both a qualified state and the local inference response. Inspect the private audit and evidence paths for detail; prompts and full responses are not logged.

The eight required checks mean: **load** proves exact model residency and GPU process attribution; **generation** proves a real reply; **context** tests configured context residency; **cancellation** cancels an active stream and verifies subsequent service; **timeout** verifies bounded failure and subsequent service; **runtime failure** stops only the owned process and observes failure; **recovery** restarts it and verifies inference; **stability** repeats bounded inference. Structured output and tool calling are separate capabilities.

If the state is UNKNOWN, inspect missing facts or evidence. STALE means identity changed; requalify from the beginning. FAILED means at least one check or preflight failed; inspect the per-check safe failure codes and retained evidence. An unavailable GPU UUID, ambiguous NVIDIA process attribution, different Ollama digest or quantization, occupied endpoint, or missing owned process prevents a PASS. Do not mark a fixture run as physical evidence.

The first native Ollama path requires Python and the repository's pinned dependencies plus the separately authorized native Ollama/model. Docker, WSL, standalone CUDA Toolkit, Node/npm, and llama.cpp are not required. The normal repository tests need none of these runtimes or an NVIDIA GPU.
