# Operator Guide

## GUI

Launch from `C:\DanteAI-gui`:

```powershell
C:\DanteAI\.venv\Scripts\python.exe -m dante.control_center_gui
```

Close window to stop. Local only. No cloud fallback. Do not assume direct local chat works in the GUI.

## Endpoints

- Dev Ollama: `127.0.0.1:11434` — local Qwen endpoint.
- Production Node0: `127.0.0.1:11435` — strictly protected. Never touch.

## Work states

- **queued** — waiting for a slot.
- **working** — actively running.
- **timeout** — exceeded time limit.
- **error** — failed.

GUI queue counts are dashboard counts. Queue position and slot owner remain unknown. Retry is unavailable in GUI.

## Tests

Same interpreter:

```powershell
C:\DanteAI\.venv\Scripts\python.exe -m unittest discover -s tests -t tests
```

## Dev recovery (11434 only)

1. Check the 11434 listener.
2. Check Ollama `/api/tags` and `/api/ps`.
3. If the server is absent, use the normal installed Ollama executable/config only.
4. Verify `/api/tags` and `/api/ps` after recovery.

Rules:

- Never kill llama-server merely because a model stays resident.
- Never touch production 11435 or the shared model store.
