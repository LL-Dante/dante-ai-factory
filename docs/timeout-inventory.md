# Timeout inventory

Verified facts only. These timeout mechanisms are independent and must not be assumed to be the same mechanism. Do not change any timeout value. Details not verified are marked UNKNOWN.

| NAME | COMPONENT | DEFAULT / CONFIG SOURCE | APPLIES TO | ERROR TEXT / RETRY | SAFE TO CHANGE | CURRENT VALUE |
| --- | --- | --- | --- | --- | --- | --- |
| Adapter profile cap | Node0Supervisor / local adapter profile | min(profile timeout, 30 s) | local adapter profile | UNKNOWN | UNKNOWN | 30 s cap |
| ToolManifest.timeout_s | ToolBroker | ToolManifest.timeout_s default 30 s | ToolBroker wait on tool execution | UNKNOWN | UNKNOWN | 30 s (default) |
| Control-pipe client I/O | control-pipe client | client I/O default 75 s | control-pipe client I/O | UNKNOWN | UNKNOWN | 75 s (default) |
| WorkloadSpec.timeout_s | workload orchestrator | WorkloadSpec.timeout_s default 120 s; maximum_attempts default 2 | workload execution | UNKNOWN / retry: max 2 attempts | UNKNOWN | 120 s, 2 attempts (defaults) |
| OpenCode timeout | OpenCode config | config timeout 600000 ms | OpenCode | UNKNOWN | UNKNOWN | 600000 ms |
| OpenCode headerTimeout | OpenCode config | config headerTimeout 600000 ms | OpenCode provider headers | historical log: ProviderHeaderTimeoutError after 300000 ms; retry UNKNOWN | UNKNOWN | 600000 ms |
| Operator timeout 30000 | UNRESOLVED | source not located (UNRESOLVED) | UNKNOWN | UNKNOWN | UNKNOWN | UNRESOLVED (30000 observed) |
