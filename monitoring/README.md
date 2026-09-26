# DanteAI local hardware monitor

Run: `C:\Users\dante\Documents\Codex\tools\uv-python\cpython-3.12.13-windows-x86_64-none\python.exe C:\DanteAI\monitoring\scripts\monitor.py`

Open `http://127.0.0.1:8765`. The HTTP server binds only to loopback. `GET /api/status` and `state/current.json` return the same normalized schema. Null values explicitly mean unavailable, never zero.

Validated sources: GPU metrics use `nvidia-smi`; CPU utilization uses Windows formatted performance counters; memory uses `Win32_OperatingSystem`; Lexar model/health uses `Get-PhysicalDisk`.

HWiNFO v8.52 is installed and was started in sensors mode, but its `Global\\HWiNFO_SENS_SM2` shared-memory mapping is not enabled, so CPU/CCD/NVMe/motherboard temperatures, CPU package power, GPU hotspot and memory-junction temperature are intentionally unavailable. Enable **Settings → Shared Memory Support** in HWiNFO later to make a supported HWiNFO adapter the next extension; no guessing is performed today.

Telemetry logs are CSV, daily rotated, sampled each 60 seconds, and retained 30 days. Alerts are daily CSVs with a 15-minute repeat cooldown and recovery events. Adjust all policy in `config/thresholds.json`.

Autostart is deliberately not enabled. To enable it, run `scripts/install-autostart.ps1` from an elevated PowerShell; it creates task `DanteAI Hardware Monitoring`. The script documents disable/removal commands.
