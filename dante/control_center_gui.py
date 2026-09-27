"""Local desktop Control Center for the existing Node 0 APIs.

The UI is a presentation layer. Agent execution, job state, qualification and
hardware facts remain owned by Dante's existing control plane and contracts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
import time
import urllib.request
import tkinter as tk
from datetime import datetime, timezone
from tkinter import messagebox, ttk

from dante.hardware_inventory import collect_snapshot
from dante.node0_control import ControlError, Node0ControlClient
from dante.config import load_config
from dante.dev_qwen_runtime import DevelopmentQwenAdapter
from dante.dev_worker_service import DevelopmentWorkerService, discover_development_qwen


LOG = logging.getLogger("dante.control_center_gui")
PAGES = ("Dashboard", "Chat", "Agents", "Jobs", "Models", "Hardware", "Logs", "Settings")
STATES = {
    "queued": ("QUEUED", "#8792a2"), "claimed": ("STARTING", "#d7a83e"),
    "running": ("RUNNING", "#39b978"), "cancel_requested": ("STOPPING", "#d7a83e"),
    "retry_wait": ("RETRY_BACKOFF", "#e29a38"), "succeeded": ("DONE", "#39b978"),
    "failed": ("ERROR", "#e05d65"), "cancelled": ("CANCELLED", "#8792a2"),
    "timed_out": ("TIMEOUT", "#e05d65"),
}
OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
OPENCODE_CONFIG = Path(__file__).resolve().parent.parent / "opencode.json"


def read_opencode_limits(config=None):
    """Return only supported local Qwen limits and timeouts; never expose URLs."""
    if config is None:
        try:
            config = json.loads(OPENCODE_CONFIG.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    providers = config.get("provider", {}) if isinstance(config, dict) else {}
    provider = providers.get("ollama", {}) if isinstance(providers, dict) else {}
    options = provider.get("options", {}) if isinstance(provider, dict) else {}
    models = provider.get("models", {}) if isinstance(provider, dict) else {}
    model = models.get("dante-qwen-agent:latest", {}) if isinstance(models, dict) else {}
    limits = model.get("limit", {}) if isinstance(model, dict) else {}

    def numeric(container, key):
        value = container.get(key) if isinstance(container, dict) else None
        return str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < float("inf") else "Unknown"

    return {
        "timeout": numeric(options, "timeout"),
        "header_timeout": numeric(options, "headerTimeout"),
        "context": numeric(limits, "context"),
        "output": numeric(limits, "output"),
    }


def format_bytes(value, unit="GiB"):
    """Render a byte count, or the explicit unknown text when absent."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return "Unknown (not reported)"
    return f"{round(value / (1024 ** 3), 1)} {unit}"


def format_count(value, unit=""):
    """Render a plain count, or the explicit unknown text when absent."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return "Unknown (not reported)"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value) + (f" {unit}" if unit else "")


def parse_ollama_tags(payload):
    """Parse GET /api/tags into display rows; missing facts stay explicit."""
    models = []
    for entry in (payload or {}).get("models", []):
        if not isinstance(entry, dict):
            continue
        details = entry.get("details") or {}
        models.append({
            "name": entry.get("name") or "Unknown (not reported)",
            "size": format_bytes(entry.get("size")),
            "quantization": details.get("quantization_level") or "Unknown (not reported)",
            "max_context": format_count(details.get("context_length")),
        })
    return models


def parse_ollama_ps(payload):
    """Parse GET /api/ps into display rows; missing facts stay explicit."""
    models = []
    for entry in (payload or {}).get("models", []):
        if not isinstance(entry, dict):
            continue
        models.append({
            "name": entry.get("name") or "Unknown (not reported)",
            "size_vram": format_bytes(entry.get("size_vram"), "GiB VRAM"),
            "context_length": format_count(entry.get("context_length")),
        })
    return models


def build_ollama_status(online, tags_payload, ps_payload, errors=()):
    """Assemble the read-only status state shown on the Settings page."""
    return {
        "online": bool(online),
        "endpoint": OLLAMA_ENDPOINT,
        "installed": parse_ollama_tags(tags_payload),
        "resident": parse_ollama_ps(ps_payload),
        "errors": list(errors),
    }


def build_model_rows(snapshot, ollama_state, opencode_limits=None):
    """Assemble Models-page rows; metadata max context and resident runtime
    context are separate columns and missing facts stay explicitly unknown."""
    limits = read_opencode_limits() if opencode_limits is None else opencode_limits
    configured_context = limits.get("context", "Unknown")
    rows, seen = [], set()
    for runtime in (snapshot.runtimes if snapshot else ()):
        for model in runtime.installed_models:
            model_context = configured_context if model.runtime_reference == "dante-qwen-agent:latest" else "Not applicable"
            rows.append((model.runtime_reference, runtime.endpoint, "Unknown (not reported)", "Unknown (not reported)",
                         display_fact(model.quantization), "Unknown (not reported)", "Unknown (not reported)",
                         "Unknown (not reported)",
                         display_fact(model.size_bytes, scale=1 / (1024 ** 3), suffix=" GiB"), "Unknown (not reported)",
                         model_context))
            seen.add(model.runtime_reference)
    online = bool(ollama_state.get("online"))
    resident_by_name = {m.get("name"): m for m in ollama_state.get("resident", [])}
    for model in ollama_state.get("installed", []):
        name = model.get("name", "Unknown (not reported)")
        if name in seen:
            continue
        resident = resident_by_name.get(name)
        if resident:
            runtime_context = resident.get("context_length", "Unknown (not reported)")
        elif online:
            runtime_context = "NOT LOADED (unknown)"
        else:
            runtime_context = "Unknown (not reported)"
        model_context = configured_context if name == "dante-qwen-agent:latest" else "Not applicable"
        rows.append((name, OLLAMA_ENDPOINT, "AVAILABLE" if online else "OFFLINE",
                     "LOADED" if resident else ("NOT LOADED" if online else "Unknown (not reported)"),
                     model.get("quantization", "Unknown (not reported)"),
                     model.get("max_context", "Unknown (not reported)"),
                     runtime_context,
                     resident.get("size_vram", "Unknown (not reported)") if resident else "Unknown (not reported)",
                     model.get("size", "Unknown (not reported)"), "Unknown (not reported)", model_context))
    return rows


def fetch_ollama(path, timeout=3.0):
    """Bounded read-only GET against the local Ollama endpoint."""
    with urllib.request.urlopen(OLLAMA_ENDPOINT.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def display_fact(fact, scale=1.0, suffix=""):
    """Render a measured value or its explicit unknown reason."""
    if fact is None:
        return "Unknown (not reported)"
    kind, value = getattr(fact, "kind", None), getattr(fact, "value", None)
    if kind == "unknown":
        return f"Unknown ({fact.reason})"
    if isinstance(value, (int, float)):
        value = round(value * scale, 1) if scale != 1 else value
    source = getattr(fact, "source", "")
    return f"{value}{suffix}" + (f"  ·  {source}" if source else "")


def job_label(job):
    label, color = STATES.get(job.get("state"), (str(job.get("state", "UNKNOWN")).upper(), "#8792a2"))
    failure = job.get("failure") or {}
    if failure.get("code") in {"inference_timeout", "deadline_expired", "timeout"}:
        label, color = "TIMEOUT", "#e05d65"
    return label, color


def job_counts(jobs):
    """Count queued and active jobs from existing records; no slot, ordering, or capacity inference."""
    queued = active = 0
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        state = job.get("state")
        if state == "queued":
            queued += 1
        elif state in {"running", "claimed", "retry_wait", "cancel_requested"}:
            active += 1
    return queued, active


def job_group(job):
    """Group a job record into a coarse lifecycle bucket."""
    if not isinstance(job, dict):
        return "UNKNOWN"
    state = job.get("state")
    if state in {"running", "claimed", "retry_wait", "cancel_requested"}:
        return "RUNNING"
    if state == "queued":
        return "QUEUED"
    if state == "succeeded":
        return "COMPLETED"
    if state in {"failed", "timed_out"}:
        return "FAILED"
    if state == "cancelled":
        return "CANCELLED"
    return "UNKNOWN"


def filter_log_jobs(jobs, *, job_id="", group="ALL"):
    """Filter only fields present in the recent-job summary records."""
    needle = str(job_id).strip().lower()
    return [job for job in (jobs or []) if isinstance(job, dict)
            and (not needle or needle in str(job.get("job_id", "")).lower())
            and (group == "ALL" or job_group(job) == group)]


def hardware_rows(snapshot, ollama):
    """Render only already collected hardware and dev-runtime facts."""
    rows = []

    def add(component, label, fact, *, scale=1.0, suffix=""):
        rows.append((component, label, display_fact(fact, scale=scale, suffix=suffix),
                     getattr(fact, "source", "") if fact is not None else ""))

    if snapshot is not None:
        for label, fact in (("Model", snapshot.cpu.model),
                            ("Physical cores", snapshot.cpu.physical_cores),
                            ("Logical threads", snapshot.cpu.logical_threads),
                            ("Utilization", snapshot.cpu.utilization_percent),
                            ("Temperature", snapshot.cpu.temperature_c),
                            ("Current clock", snapshot.cpu.current_clock_mhz)):
            add("CPU", label, fact, suffix="%" if label == "Utilization" else " MHz" if label == "Current clock" else " °C" if label == "Temperature" else "")
        for label, fact in (("Installed", snapshot.memory.installed_bytes),
                            ("Available", snapshot.memory.available_bytes)):
            add("RAM", label, fact, scale=1 / (1024 ** 3), suffix=" GiB")
        for gpu in snapshot.gpus:
            for label, fact, kwargs in (
                ("Utilization", gpu.utilization_percent, {"suffix": "%"}),
                ("VRAM used", gpu.vram_used_bytes, {"scale": 1 / (1024 ** 3), "suffix": " GiB"}),
                ("VRAM free", gpu.vram_free_bytes, {"scale": 1 / (1024 ** 3), "suffix": " GiB"}),
                ("VRAM total", gpu.vram_total_bytes, {"scale": 1 / (1024 ** 3), "suffix": " GiB"}),
                ("Temperature", gpu.temperature_c, {"suffix": " °C"}),
                ("Power", gpu.power_draw_w, {"suffix": " W"}),
                ("Power limit", gpu.power_limit_w, {"suffix": " W"}),
                ("SM clock", gpu.sm_clock_mhz, {"suffix": " MHz"}),
                ("Memory clock", gpu.memory_clock_mhz, {"suffix": " MHz"}),
            ):
                add(gpu.name, label, fact, **kwargs)
        for volume in snapshot.volumes:
            add(volume.mount_point, "Free space", volume.free_bytes,
                scale=1 / (1024 ** 3), suffix=" GiB")
            add(volume.mount_point, "Total space", volume.size_bytes,
                scale=1 / (1024 ** 3), suffix=" GiB")
        for store in snapshot.model_stores:
            add(store.name, "Exists", store.exists)
            add(store.name, "Size", store.size_bytes, scale=1 / (1024 ** 3), suffix=" GiB")
            add(store.name, "File count", store.file_count)

    state = ("ONLINE" if ollama.get("online") is True else
             "OFFLINE" if ollama.get("online") is False else "Unknown (not reported)")
    rows.append(("Development Ollama", "Status", state,
                 ollama.get("endpoint", OLLAMA_ENDPOINT)))
    return rows


def job_phase(job, events=None):
    """Show a detailed phase only when durable state/event evidence supports it."""
    state = job.get("state")
    label, _ = job_label(job)
    if state != "running":
        return label
    rows = events if events is not None else (job.get("events") or [])
    if rows:
        name = str(rows[-1].get("event", "")).lower()
        if name == "agent.inference.started":
            return "MODEL_GENERATING"
        if name == "retry_scheduled" or "retry_backoff" in name:
            return "RETRY_BACKOFF"
        if "tool" in name and any(word in name for word in ("start", "running", "execut")):
            return "TOOL_EXECUTING"
        if "waiting" in name or name.endswith(".wait"):
            return "WAITING"
    return "RUNNING"


def _known(value):
    if value is None or isinstance(value, bool) or value == "":
        return "Unknown (not reported)"
    return str(value)


def elapsed_wall(job, now=None):
    """Elapsed wall time since submission; terminal records end at updated_at."""
    try:
        start = datetime.fromisoformat(str(job["created_at"]).replace("Z", "+00:00"))
        if start.tzinfo is None:
            return "Unknown (timestamp has no timezone)"
        terminal = job.get("state") in {"succeeded", "failed", "cancelled", "timed_out"}
        end_value = job.get("updated_at") if terminal else None
        end = datetime.fromisoformat(str(end_value).replace("Z", "+00:00")) if end_value else (now or datetime.now(timezone.utc))
        if end.tzinfo is None:
            return "Unknown (timestamp has no timezone)"
        seconds = max(0, int((end - start).total_seconds()))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (KeyError, TypeError, ValueError):
        return "Unknown (not reported)"


def job_submission_text(record):
    """Summarize a returned JobRecord without inventing slot or queue data."""
    record = record if isinstance(record, dict) else {}
    state = str(record.get("state", "Unknown (not reported)"))
    label, _ = job_label(record)
    lines = ["LOCAL JOB SUBMITTED", f"Job ID: {_known(record.get('job_id'))}", f"State: {label}"]
    if state == "queued":
        lines.extend(("Queue position: Unknown (not reported)",
                      "Model-slot ownership: Unknown (not reported)"))
    lines.extend((f"Priority: {_known(record.get('priority'))}",
                  f"Attempt count: {_known(record.get('attempt_count'))}"))
    return "\n".join(lines)


def job_details_text(job, events=()):
    """Human-readable job facts and event timeline with unknowns kept explicit."""
    failure = job.get("failure") or {}
    facts = [
        ("Job ID", _known(job.get("job_id"))), ("State / phase", job_phase(job, events)),
        ("Agent", "Unknown (not reported)"), ("Model", "Unknown (not reported)"),
        ("Endpoint", "Unknown (not reported)"), ("Priority", _known(job.get("priority"))),
        ("Queue position", "Unknown (not reported)"), ("Elapsed wall since submit", elapsed_wall(job)),
        ("Attempt count", _known(job.get("attempt_count"))),
        ("Context total tokens", "Unknown (not reported)"), ("New input tokens", "Unknown (not reported)"),
        ("Cached input tokens", "Unknown (not reported)"), ("Output tokens", "Unknown (not reported)"),
        ("TTFT", "Unknown (not reported)"), ("Prompt evaluation", "Unknown (not reported)"),
        ("Generation", "Unknown (not reported)"), ("Tool execution", "Unknown (not reported)"),
        ("Retry / backoff", "Unknown (not reported)"),
        ("Last error", _known(failure.get("code"))),
    ]
    lines = [f"{name}: {value}" for name, value in facts]
    lines.append("\nDURABLE EVENTS")
    if events:
        for event in events[-40:]:
            lines.append(f"{_known(event.get('timestamp_utc'))}  {_known(event.get('event'))}")
    else:
        lines.append("Unknown (no events reported)")
    return "\n".join(lines)


def development_job_state_label(state, telemetry, job_id):
    if (state == 'queued' and isinstance(telemetry, dict)
            and telemetry.get('slot_owner_job_id')
            and telemetry.get('slot_owner_job_id') != job_id):
        return 'QUEUED — WAITING FOR LOCAL MODEL SLOT'
    return str(state).upper()


def development_queue_lines(telemetry, job_id, state):
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    unknown = telemetry.get('unknown_reasons') or {}

    def value(key):
        item = telemetry.get(key)
        if item is None or item == '':
            return f"Unknown ({unknown.get(key, 'not reported')})"
        return str(item)

    position = value('position') if state == 'queued' else 'not queued'
    owner = telemetry.get('slot_owner_job_id')
    owner_state = telemetry.get('slot_owner_state')
    if owner_state == 'none':
        owner_text = 'NONE · no active slot owner'
    elif owner:
        worker = telemetry.get('slot_owner_worker_id')
        owner_text = str(owner) + (f' · {worker}' if worker else '')
    else:
        owner_text = f"Unknown ({unknown.get('slot_owner', 'not reported')})"
    wait = telemetry.get('queue_wait_ms')
    wait_text = f'{wait} ms' if isinstance(wait, int) and not isinstance(wait, bool) else value('queue_wait_ms')
    released = telemetry.get('slot_released_at')
    release_text = str(released) if released else f"Unknown ({unknown.get('slot_released_at', telemetry.get('slot_released_reason', 'not reported'))})"
    return [
        f"QUEUE DEPTH: {value('queue_depth')} · POSITION: {position} · CAPACITY: {value('capacity')}",
        f"PRIORITY: {value('priority')}", f"QUEUED AT: {value('queued_at')}",
        f"LOCAL MODEL SLOT OWNER: {owner_text}",
        f"SLOT ACQUIRED AT: {value('slot_acquired_at')}", f"{'CURRENT ' if state == 'queued' else ''}QUEUE WAIT: {wait_text}",
        f"SLOT RELEASED AT: {release_text}",
    ]


def development_job_metrics(events):
    """Summarize only measured development-worker event fields."""
    rows = [event for event in events if isinstance(event, dict)]
    model_events = [event for event in rows if event.get("event") == "model.completed"]
    tool_events = [event for event in rows if event.get("event") == "tool.completed"]

    def total(key):
        values = [(event.get("metadata") or {}).get(key) for event in model_events]
        if not values or any(type(value) is not int or value < 0 for value in values):
            return "Unknown (not reported)"
        return str(sum(values))

    return {
        "model_calls": str(len(model_events)),
        "tool_calls": str(len(tool_events)),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "current_operation": rows[-1].get("event", "Unknown (not reported)") if rows else "Unknown (not reported)",
    }


def development_test_status(events):
    """Report bounded test-runner events without claiming OS isolation."""
    runs = [event for event in events if isinstance(event, dict)
            and event.get("event") == "tool.completed"
            and (event.get("metadata") or {}).get("tool_id") == "RUN_TESTS"]
    if not runs:
        return "NOT RUN · bounded RUN_TESTS available"
    metadata = runs[-1].get("metadata") or {}
    if metadata.get("status") != "success":
        return f"RUN_TESTS execution {metadata.get('status', 'unknown')}"
    passed = metadata.get("tests_passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL · test executed"
    return "RUN_TESTS executed · outcome not reported"


def inspector_payload(kind, key, record=None, events=()):
    """Build safe, factual drill-down content for the shared inspector."""
    record = record if isinstance(record, dict) else {}
    events = [event for event in events if isinstance(event, dict)]
    metadata = [event.get("metadata") if isinstance(event.get("metadata"), dict) else {} for event in events]
    if kind == "job":
        metrics = development_job_metrics(events)
        telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
        candidate = record.get("job")
        if isinstance(candidate, dict):
            job = candidate
        elif callable(getattr(candidate, "model_dump", None)):
            job = candidate.model_dump(mode="json")
        else:
            job = record
        fields = [
            ("Job ID", key), ("State", record.get("state") or job.get("state") or "Unknown (not reported)"),
            ("Created / queued", (job.get("created_at") if isinstance(job, dict) else None) or telemetry.get("queued_at") or "Unknown (not reported)"),
            ("Started", telemetry.get("slot_acquired_at") or "Unknown (not reported)"),
            ("Ended", (job.get("updated_at") if isinstance(job, dict) else None) or telemetry.get("slot_released_at") or "Unknown (not reported)"),
            ("Priority", telemetry.get("priority", job.get("priority", "Unknown (not reported)") if isinstance(job, dict) else "Unknown (not reported)")),
            ("Queue position", telemetry.get("position", "Unknown (not reported)")),
            ("Queue wait", f"{telemetry['queue_wait_ms']} ms" if isinstance(telemetry.get("queue_wait_ms"), int) else "Unknown (not reported)"),
            ("Slot acquired", telemetry.get("slot_acquired_at") or "Unknown (not reported)"),
            ("Slot released", telemetry.get("slot_released_at") or "Unknown (not reported)"),
            ("Release reason", telemetry.get("slot_released_reason") or "Unknown (not reported)"),
            ("Model", "dante-qwen-agent:latest" if events or record.get("job") else "Unknown (not reported)"),
            ("Endpoint", OLLAMA_ENDPOINT if events or record.get("job") else "Unknown (not reported)"),
            ("Model calls", metrics["model_calls"]), ("Tool calls", metrics["tool_calls"]),
            ("Input tokens", metrics["input_tokens"]), ("Output tokens", metrics["output_tokens"]),
            ("Tests", development_test_status(events)),
            ("Current operation", metrics["current_operation"]),
            ("Result", json.dumps(job.get("result"), ensure_ascii=False, default=str) if isinstance(job, dict) and job.get("result") is not None else "Unknown (not reported)"),
            ("Failure", json.dumps(job.get("failure"), ensure_ascii=False, default=str) if isinstance(job, dict) and job.get("failure") is not None else "None reported"),
        ]
        return {"kind": kind, "key": key, "title": f"Job · {key}", "summary": str(fields[1][1]),
                "fields": fields, "events": events[-40:], "related": [("Model", "model", "dante-qwen-agent:latest"),
                    ("Runtime", "runtime", OLLAMA_ENDPOINT)]}
    if kind in {"model_calls", "tool_calls", "tests", "failure", "queue", "model", "runtime", "agent", "hardware"}:
        if kind == "model_calls":
            selected = [(event, meta) for event, meta in zip(events, metadata) if event.get("event") == "model.completed"]
            rows = []
            for sequence, (event, meta) in enumerate(selected, 1):
                rows.append((f"Call {sequence}", [
                    ("Timestamp", event.get("timestamp_utc", "Unknown (not reported)")),
                    ("Input tokens", meta.get("input_tokens", "Unknown (not reported)")),
                    ("Cached input tokens", meta.get("cached_input_tokens", "Unknown (not reported)")),
                    ("Output tokens", meta.get("output_tokens", "Unknown (not reported)")),
                    ("Elapsed model time", meta.get("model_wall_s", "Unknown (not reported)")),
                    ("TTFT", meta.get("ttft_s", "Unknown (not reported)")),
                    ("Generation time", meta.get("generation_s", "Unknown (not reported)")),
                    ("Output tokens/s", meta.get("output_tokens_per_second", "Unknown (not reported)")),
                    ("Requested action", meta.get("tool_requested", "No tool requested")),
                    ("Arguments summary", meta.get("tool_arguments_summary", "Unknown (not reported)")),
                    ("Configured context tokens", meta.get("configured_context_tokens", "Unknown (not reported)")),
                    ("Context remaining tokens", meta.get("context_remaining_tokens", "Unknown (not reported)")),
                    ("State", event.get("event", "Unknown (not reported)")),
                ]))
            return {"kind": kind, "key": key, "title": f"Model calls · {key}", "summary": f"{len(rows)} calls",
                    "fields": [(f"Call {n}", f"{len(f)} reported fields") for n, f in rows],
                    "sections": rows, "events": [], "related": [("Model", "model", "dante-qwen-agent:latest")]}
        if kind == "tool_calls":
            selected = [(event, meta) for event, meta in zip(events, metadata) if event.get("event") == "tool.completed"]
            rows = []
            for sequence, (event, meta) in enumerate(selected, 1):
                rows.append((f"Tool {sequence} · {meta.get('tool_id', 'Unknown')}", [
                    ("Timestamp", event.get("timestamp_utc", "Unknown (not reported)")),
                    ("Arguments", meta.get("normalized_arguments", "Unknown (not reported)")),
                    ("Result summary", meta.get("result_summary", "Unknown (not reported)")),
                    ("Duration", meta.get("elapsed_s", "Unknown (not reported)")),
                    ("Tool duration", meta.get("tool_wall_s", "Unknown (not reported)")),
                    ("Status", meta.get("status", "Unknown (not reported)")),
                    ("Result digest", meta.get("result_digest", "Unknown (not reported)")),
                    ("Error", meta.get("error_type", "None reported")),
                ]))
            return {"kind": kind, "key": key, "title": f"Tool calls · {key}", "summary": f"{len(rows)} calls",
                    "fields": [(n, f"{len(f)} reported fields") for n, f in rows],
                    "sections": rows, "events": [], "related": []}
        telemetry = record.get("telemetry") if isinstance(record.get("telemetry"), dict) else {}
        if kind == "queue":
            fields = [(name, telemetry.get(source, "Unknown (not reported)")) for name, source in (
                ("Queued at", "queued_at"), ("Queue position", "position"), ("Depth", "queue_depth"),
                ("Priority", "priority"), ("Wait ms", "queue_wait_ms"), ("Capacity", "capacity"),
                ("Slot owner", "slot_owner_job_id"), ("Slot acquired", "slot_acquired_at"),
                ("Slot released", "slot_released_at"), ("Release reason", "slot_released_reason"))]
        elif kind == "model":
            fields = [("Model", key), ("Runtime reference", key),
                      ("Availability", record.get("availability", "Unknown (not reported)")),
                      ("Residency", record.get("residency", "Unknown (not reported)")),
                      ("Configured context", record.get("context", "Unknown (not reported)")),
                      ("Runtime context", record.get("runtime_context", "Unknown (not reported)")),
                      ("Maximum context metadata", record.get("max_context", "Unknown (not reported)")),
                      ("Parameter count", "Unknown (not reported)"),
                      ("Quantization", record.get("quantization", "Unknown (not reported)")),
                      ("Size", record.get("size", "Unknown (not reported)"))]
        elif kind == "runtime":
            fields = [("Endpoint", key), ("Runtime state", record.get("state", "Unknown (not reported)")),
                      ("Slot owner", telemetry.get("slot_owner_job_id", "Unknown (not reported)")),
                      ("Queue depth", telemetry.get("queue_depth", "Unknown (not reported)"))]
        elif kind == "agent":
            fields = [("Agent", key), ("State", record.get("state", "Unknown (not reported)")),
                      ("Current job", record.get("job_id", "Unknown (not reported)")),
                      ("Model", record.get("model", "Unknown (not reported)"))]
        elif kind == "hardware":
            fields = [("Metric", key), ("Value", record.get("value", "Unknown (not reported)")),
                      ("Source", record.get("source", "Unknown (not reported)")),
                      ("Observed at", record.get("observed_at", "Unknown (not reported)"))]
        elif kind == "tests":
            selected = [(event, meta) for event, meta in zip(events, metadata) if event.get("event") == "tool.completed" and meta.get("tool_id") == "RUN_TESTS"]
            fields = []
            for n, (event, meta) in enumerate(selected, 1):
                try:
                    summary = json.loads(meta.get("result_summary", "{}"))
                except (TypeError, ValueError):
                    summary = {}
                fields.extend(((f"Test run {n} target", summary.get("target", "Unknown (not reported)")),
                               (f"Test run {n} project root", summary.get("project_root", "Unknown (not reported)")),
                               (f"Test run {n} command", summary.get("command_summary", "Unknown (not reported)")),
                               (f"Test run {n} started/completed", event.get("timestamp_utc", "Unknown (not reported)")),
                               (f"Test run {n} duration", summary.get("duration_s", "Unknown (not reported)")),
                               (f"Test run {n} exit code", summary.get("return_code", "Unknown (not reported)")),
                               (f"Test run {n} passed", summary.get("tests_passed", "Unknown (not reported)")),
                               (f"Test run {n} result", summary.get("test_result_summary", "Unknown (not reported)")),
                               (f"Test run {n} failed cases", summary.get("failed_tests", [])),
                               (f"Test run {n} output", f"Summary only · raw output not persisted · {summary.get('output_bytes', 'Unknown')} bytes · sha256 {summary.get('output_sha256', 'Unknown (not reported)')}")))
            if not fields:
                fields = [("Tests", development_test_status(events)), ("Project root", "Unknown (not reported)"),
                          ("Command", "Unknown (not reported)"), ("Bounded output", "No RUN_TESTS output persisted")]
        else:
            selected = []
            for event, meta in zip(events, metadata):
                try:
                    result = json.loads(meta.get("result_summary", "{}"))
                except (TypeError, ValueError):
                    result = {}
                code = result.get("tool_error_code") if isinstance(result, dict) else None
                if meta.get("status") not in {None, "success"} or meta.get("error_type") or code:
                    selected.append((event, meta, code))
            fields = [("FACT", f"{meta.get('tool_id', 'Unknown tool')} · {meta.get('status', 'Unknown status')} · {event.get('timestamp_utc', 'Unknown (not reported)')}")
                      for event, meta, _code in selected]
            fields.extend(("ERROR DATA", meta.get("error_type") or code or "Unknown (not reported)")
                          for _event, meta, code in selected)
            known_codes = {code for _event, _meta, code in selected if code}
            if "not_found" in known_codes:
                diagnosis = "Known: READ_FILE target did not exist in the worker workspace."
            elif "not_a_file" in known_codes:
                diagnosis = "Known: READ_FILE target resolved to a directory."
            elif known_codes & {"invalid_encoding", "too_large", "blocked_path", "read_failed"}:
                diagnosis = "Known: READ_FILE returned a safe structured error; see the tool code above."
            else:
                diagnosis = "Unknown (no deterministic diagnosis established by persisted events)."
            fields.extend((("Diagnosis", diagnosis), ("Evidence", "Persisted job events only; no LLM-generated diagnosis.")))
        timeline = events[-40:] if kind in {"queue", "failure", "tests"} else []
        related = [("Runtime", "runtime", OLLAMA_ENDPOINT)] if kind == "model" else []
        return {"kind": kind, "key": key, "title": f"{kind.replace('_', ' ').title()} · {key}",
                "summary": f"{len(fields)} factual details", "fields": fields,
                "events": timeline, "related": related}
    return {"kind": kind, "key": key, "title": str(kind).title(), "summary": "Unknown (not reported)",
            "fields": [("Value", str(key))], "events": [], "related": []}


class ControlCenterApp:
    POLL_SECONDS = 3

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dante AI Factory · Control Center")
        self.root.geometry("1440x900")
        self.root.minsize(1160, 740)
        self.root.configure(bg="#11151d")
        self.client_factory = Node0ControlClient
        self.snapshot = None
        self.status = {}
        self.health = {}
        self.jobs = []
        self.agents = []
        self.events = []
        self.ollama = {"online": False, "endpoint": OLLAMA_ENDPOINT, "installed": [], "resident": [], "errors": []}
        self._busy = False
        self._dev_worker = None
        self._dev_worker_lock = threading.Lock()
        self._dev_jobs = {}
        self._dev_job_started = {}
        self._dev_selected_job_id = None
        self._dev_poll_inflight = set()
        self._dev_submit_pending = 0
        self._dev_poll_after_id = None
        self._inspector_object = None
        self._build()
        self.refresh()

    def _build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#11151d")
        style.configure("Panel.TFrame", background="#1a202b")
        style.configure("TLabel", background="#11151d", foreground="#e6eaf0", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground="#98a4b5")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 21), foreground="#f4f6f9")
        style.configure("Metric.TLabel", font=("Segoe UI Semibold", 16), foreground="#ffffff", background="#1a202b")
        style.configure("PanelTitle.TLabel", font=("Segoe UI Semibold", 11), foreground="#aab6c8", background="#1a202b")
        style.configure("Inspect.TLabel", foreground="#9fc5ff", cursor="hand2")
        style.map("Inspect.TLabel", foreground=[("active", "#ffffff")])
        style.configure("TButton", padding=(12, 8), background="#263143", foreground="#eef2f7")
        style.map("TButton", background=[("active", "#35435a")])
        style.configure("Treeview", background="#171d27", fieldbackground="#171d27", foreground="#e6eaf0", rowheight=30)
        style.configure("Treeview.Heading", background="#263143", foreground="#c5cfdd", font=("Segoe UI Semibold", 9))

        shell = ttk.Frame(self.root, padding=18)
        shell.pack(fill="both", expand=True)
        top = ttk.Frame(shell)
        top.pack(fill="x", pady=(0, 15))
        ttk.Label(top, text="DANTE AI FACTORY", style="Muted.TLabel").pack(side="left")
        self.node_badge = tk.Label(top, text="NODE0 · CHECKING", bg="#333b49", fg="white", padx=12, pady=6, font=("Segoe UI Semibold", 9))
        self.node_badge.pack(side="right")
        self.updated = ttk.Label(top, text="Connecting to local Node 0…", style="Muted.TLabel")
        self.updated.pack(side="right", padx=12)

        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)
        nav = ttk.Frame(body, width=172, style="Panel.TFrame", padding=10)
        nav.pack(side="left", fill="y", padx=(0, 14))
        nav.pack_propagate(False)
        for page in PAGES:
            ttk.Button(nav, text=page, command=lambda p=page: self.show_page(p)).pack(fill="x", pady=3)
        ttk.Button(nav, text="Refresh now", command=self.refresh).pack(fill="x", pady=(16, 3))

        self.content = ttk.Frame(body)
        self.content.pack(side="left", fill="both", expand=True)
        self.inspector = ttk.Frame(body, style="Panel.TFrame", padding=12, width=360)
        self.inspector.pack(side="right", fill="y", padx=(12, 0))
        self.inspector.pack_propagate(False)
        self.inspector_title = ttk.Label(self.inspector, text="INSPECTOR", style="PanelTitle.TLabel", wraplength=330)
        self.inspector_title.pack(anchor="w")
        toolbar = ttk.Frame(self.inspector, style="Panel.TFrame")
        toolbar.pack(fill="x", pady=(8, 8))
        ttk.Button(toolbar, text="Copy details", command=self._copy_inspector).pack(side="left")
        ttk.Button(toolbar, text="Refresh", command=self._refresh_inspector).pack(side="left", padx=6)
        canvas = tk.Canvas(self.inspector, background="#1a202b", highlightthickness=0, width=336)
        inspector_scroll = ttk.Scrollbar(self.inspector, orient="vertical", command=canvas.yview)
        self.inspector_body = ttk.Frame(canvas, style="Panel.TFrame")
        self.inspector_body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inspector_body, anchor="nw", width=320)
        canvas.configure(yscrollcommand=inspector_scroll.set)
        inspector_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._inspector_object = None
        self._inspector_text = "Select an inspectable value to view its evidence."
        self.page_title = ttk.Label(self.content, text="Dashboard", style="Title.TLabel")
        self.page_title.pack(anchor="w", pady=(0, 14))
        self.page = ttk.Frame(self.content)
        self.page.pack(fill="both", expand=True)
        self.show_page("Dashboard")

    def _request(self, op, **kwargs):
        return self.client_factory().request({"op": op, **kwargs})

    def refresh(self):
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self):
        result = {"errors": []}
        for key, op in (("status", "status"), ("health", "health"), ("jobs", "workload_list"), ("agents", "agent_list")):
            try:
                result[key] = self._request(op, **({"limit": 100} if op == "workload_list" else {}))
            except Exception as exc:
                result["errors"].append(f"{op}: {getattr(exc, 'code', type(exc).__name__)}")
                LOG.exception("Control API request failed: %s", op)
        try:
            result["hardware"] = collect_snapshot()
        except Exception as exc:
            result["errors"].append(f"hardware: {type(exc).__name__}")
            LOG.exception("Hardware inventory collection failed")
        tags, tags_error = None, None
        try:
            tags = fetch_ollama("/api/tags")
        except Exception as exc:
            tags_error = f"ollama tags: {type(exc).__name__}"
            LOG.exception("Ollama /api/tags request failed")
        ps, ps_error = None, None
        try:
            ps = fetch_ollama("/api/ps")
        except Exception as exc:
            ps_error = f"ollama ps: {type(exc).__name__}"
            LOG.exception("Ollama /api/ps request failed")
        errors = []
        if tags_error:
            result["errors"].append(tags_error)
            errors.append(tags_error)
        if ps_error:
            result["errors"].append(ps_error)
            errors.append(ps_error)
        result["ollama"] = build_ollama_status(bool(tags), tags, ps, errors)
        self.root.after(0, lambda: self._apply(result))

    def _apply(self, result):
        self._busy = False
        self.status = result.get("status") or {}
        self.health = result.get("health") or {}
        self.jobs = result.get("jobs") or []
        self.agents = result.get("agents") or []
        self.snapshot = result.get("hardware")
        self.ollama = result.get("ollama") or {"online": False, "endpoint": OLLAMA_ENDPOINT, "installed": [], "resident": [], "errors": []}
        online = bool(self.status)
        label = "NODE0 · ONLINE" if online else "NODE0 · OFFLINE"
        color = "#1f6848" if online else "#8e303a"
        self.node_badge.configure(text=label, bg=color)
        self.updated.configure(text=datetime.now().astimezone().strftime("Updated %H:%M:%S") + (" · " + "; ".join(result["errors"]) if result["errors"] else ""))
        self.events = [f"{j.get('updated_at', '')}  {job_label(j)[0]}  {j.get('job_id', '')}" for j in self.jobs[:20]]
        if self.current_page != "Chat":
            self.show_page(self.current_page)
        self.root.after(self.POLL_SECONDS * 1000, self.refresh)

    def show_page(self, name):
        self.current_page = name
        self.page_title.configure(text=name)
        for child in self.page.winfo_children():
            child.destroy()
        getattr(self, f"page_{name.lower()}")()

    def _inspectable(self, parent, label, value, kind, key=None, record=None, events=()):
        text = f"{label}: {value}  ›"
        widget = ttk.Label(parent, text=text, style="Inspect.TLabel", wraplength=580)
        widget.pack(anchor="w", pady=2)
        payload = inspector_payload(kind, key if key is not None else value, record, events)
        widget.bind("<Button-1>", lambda _event, detail=payload: self._show_inspector(detail))
        return widget

    def _show_inspector(self, payload):
        self._inspector_object = payload
        for child in self.inspector_body.winfo_children():
            child.destroy()
        self.inspector_title.configure(text=payload["title"].upper())
        ttk.Label(self.inspector_body, text=payload.get("summary", "Unknown (not reported)"),
                  style="Muted.TLabel", wraplength=330).pack(anchor="w", pady=(0, 8))
        sections = payload.get("sections")
        if sections:
            for title, fields in sections:
                ttk.Label(self.inspector_body, text=title, style="PanelTitle.TLabel").pack(anchor="w", pady=(6, 2))
                for name, value in fields:
                    ttk.Label(self.inspector_body, text=f"{name}: {value}", style="Muted.TLabel",
                              wraplength=330, justify="left").pack(anchor="w", pady=1)
        else:
            for name, value in payload.get("fields", []):
                ttk.Label(self.inspector_body, text=f"{name}: {value}", style="Muted.TLabel",
                          wraplength=330, justify="left").pack(anchor="w", pady=2)
        related = payload.get("related", [])
        if related:
            ttk.Label(self.inspector_body, text="RELATED", style="PanelTitle.TLabel").pack(anchor="w", pady=(10, 3))
            for label, kind, key in related:
                ttk.Button(self.inspector_body, text=f"Inspect {label}: {key}",
                           command=lambda k=kind, v=key: self._show_inspector(inspector_payload(k, v))).pack(anchor="w", pady=2)
        timeline = payload.get("events", [])
        if timeline:
            ttk.Label(self.inspector_body, text="EVENT EVIDENCE", style="PanelTitle.TLabel").pack(anchor="w", pady=(10, 3))
            for event in timeline:
                ttk.Label(self.inspector_body,
                          text=f"{event.get('timestamp_utc', 'Unknown')} · {event.get('event', 'Unknown')} · {json.dumps(event.get('metadata') or {}, ensure_ascii=False, default=str)}",
                          style="Muted.TLabel", wraplength=330, justify="left").pack(anchor="w", pady=2)

    def _copy_inspector(self):
        if self._inspector_object is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(json.dumps(self._inspector_object, ensure_ascii=False, indent=2, default=str))

    def _refresh_inspector(self):
        payload = self._inspector_object
        if payload is None:
            self.refresh()
        elif payload.get("kind") == "job" and payload.get("key") in self._dev_jobs:
            job_id = payload["key"]
            self._dev_jobs[job_id]["details_loaded"] = False
            self._poll_dev_job()
        else:
            self.refresh()

    def _panel(self, parent, title, value, detail=""):
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        ttk.Label(frame, text=title.upper(), style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text=value, style="Metric.TLabel").pack(anchor="w", pady=(9, 4))
        if detail:
            ttk.Label(frame, text=detail, style="Muted.TLabel", wraplength=230).pack(anchor="w")
        return frame

    def page_dashboard(self):
        snap = self.snapshot
        row = ttk.Frame(self.page)
        row.pack(fill="x", pady=(0, 12))
        state = "OFFLINE" if not self.status else str(self.status.get("state", "UNKNOWN"))
        active = next((j for j in self.jobs if j.get("state") in {"running", "claimed", "retry_wait", "cancel_requested"}), None)
        active_label = job_label(active)[0] if active else "IDLE"
        model = self.status.get("model_reference") or self.health.get("model_reference") or "Unknown"
        dev_state = "ONLINE" if self.ollama.get("online") else "OFFLINE"
        dev_model = ", ".join(m.get("name", "Unknown") for m in self.ollama.get("resident", [])) or "No resident model reported"
        for title, value, detail in (("Node 0", state, "Local production supervisor"), ("Development runtime", dev_state, OLLAMA_ENDPOINT), ("Agent", active_label, active.get("job_id", "No active job") if active else "No active job"), ("Mode", "LOCAL", "No automatic cloud fallback"), ("Active model", model, "Qualification gate applies"), ("Development model", dev_model, "Resident model from 11434 /api/ps")):
            self._panel(row, title, value, detail).pack(side="left", fill="x", expand=True, padx=(0, 9))

        metrics = ttk.Frame(self.page)
        metrics.pack(fill="x", pady=(0, 12))
        cpu = display_fact(snap.cpu.utilization_percent, suffix="%") if snap else "Unknown (no snapshot)"
        ram = display_fact(snap.memory.available_bytes, scale=1 / (1024 ** 3), suffix=" GiB free") if snap else "Unknown (no snapshot)"
        gpu = snap.gpus[0] if snap and snap.gpus else None
        vals = [("GPU", display_fact(gpu.utilization_percent, suffix="%") if gpu else "Unknown (no GPU fact)"),
                ("VRAM", display_fact(gpu.vram_used_bytes, scale=1 / (1024 ** 3), suffix=" GiB used") if gpu else "Unknown (no GPU fact)"),
                ("Temperature / power", f"{display_fact(gpu.temperature_c, suffix=' °C')} / {display_fact(gpu.power_draw_w, suffix=' W')}" if gpu else "Unknown (no GPU fact)"),
                ("CPU / RAM", f"{cpu} / {ram}")]
        for title, value in vals:
            self._panel(metrics, title, value).pack(side="left", fill="x", expand=True, padx=(0, 9))

        queued_count, active_count = job_counts(self.jobs)
        counts = ttk.Frame(self.page)
        counts.pack(fill="x", pady=(0, 12))
        for title, value in (("Queued jobs", queued_count), ("Active jobs (running / claimed / retry / cancel)", active_count)):
            self._panel(counts, title, str(value)).pack(side="left", fill="x", expand=True, padx=(0, 9))

        ttk.Label(self.page, text="ACTIVE WORK", style="PanelTitle.TLabel").pack(anchor="w", pady=(4, 7))
        if active:
            status, color = job_label(active)
            box = ttk.Frame(self.page, style="Panel.TFrame", padding=14)
            box.pack(fill="x")
            tk.Label(box, text=status, bg=color, fg="#11151d", padx=9, pady=5, font=("Segoe UI Semibold", 10)).pack(side="left")
            ttk.Label(box, text=f"{active.get('job_id')}  ·  attempt {active.get('attempt_count', 0)}", style="Metric.TLabel").pack(side="left", padx=12)
            ttk.Label(box, text=self._operation_text(active), style="Muted.TLabel").pack(side="left", padx=10)
            ttk.Button(box, text="Details", command=lambda j=active: self.show_job_events(j)).pack(side="right")
            ttk.Button(box, text="Stop", command=lambda j=active: self.cancel_job(j)).pack(side="right", padx=8)
        else:
            self._panel(self.page, "Current operation", "IDLE", "No queued or running local job").pack(fill="x")

        lower = ttk.Frame(self.page)
        lower.pack(fill="both", expand=True, pady=(13, 0))
        self._panel(lower, "Agent state", "Tool activity is reported by workload events", "RUNNING · WAITING · TOOL · RETRY / BACKOFF · TIMEOUT · ERROR · DONE are separate job states.").pack(side="left", fill="both", expand=True, padx=(0, 9))
        self._panel(lower, "Last event", self.events[0] if self.events else "No job events yet", "Event details remain attached to the durable job record.").pack(side="left", fill="both", expand=True)

    def _tree(self, columns, headings, parent=None):
        tree = ttk.Treeview(parent or self.page, columns=columns, show="headings")
        for col, heading in zip(columns, headings):
            tree.heading(col, text=heading)
            tree.column(col, width=150, anchor="w")
        tree.pack(fill="both", expand=True)
        return tree

    def page_chat(self):
        ttk.Label(self.page, text="Local Qwen worker · 11434", style="PanelTitle.TLabel").pack(anchor="w")
        form = ttk.Frame(self.page, style="Panel.TFrame", padding=14)
        form.pack(fill="x", pady=10)
        ttk.Label(form, text="Model", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(form, text="LOCAL · dante-qwen-agent:latest · bounded RUN_TESTS available", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=10)
        ttk.Label(form, text="Objective", style="Muted.TLabel").grid(row=1, column=0, sticky="nw", pady=10)
        objective = tk.Text(form, height=5, bg="#11151d", fg="#e6eaf0", insertbackground="white", relief="flat", wrap="word")
        objective.grid(row=1, column=1, sticky="ew", padx=10, pady=10)
        form.columnconfigure(1, weight=1)
        buttons = ttk.Frame(form)
        buttons.grid(row=2, column=1, sticky="e")
        submit_button = ttk.Button(buttons, text="Submit local job", command=self._submit_dev_job)
        submit_button.pack(side="left", padx=(0, 8))
        cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel_dev_job)
        cancel_button.pack(side="left")
        self._chat_objective = objective
        self._chat_submit_button = submit_button
        self._chat_cancel_button = cancel_button
        jobs_frame = ttk.Frame(self.page)
        jobs_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(jobs_frame, text="RECENT LOCAL JOBS · select one to inspect its own telemetry", style="PanelTitle.TLabel").pack(anchor="w")
        self._chat_jobs_tree = self._tree(("state", "job_id", "queue"), ("State", "Job ID", "Queue"), jobs_frame)
        self._chat_jobs_tree.configure(height=5)
        self._chat_jobs_tree.column("state", width=260)
        self._chat_jobs_tree.column("job_id", width=300)
        self._chat_jobs_tree.column("queue", width=180)
        self._chat_jobs_tree.bind("<<TreeviewSelect>>", self._select_dev_job)
        output = tk.Text(self.page, bg="#171d27", fg="#e6eaf0", relief="flat", wrap="word", state="disabled")
        output.pack(fill="both", expand=True)
        self._chat_output = output
        actions = ttk.Frame(self.page)
        actions.pack(fill="x", pady=(4, 0))
        self._chat_inspect_actions = []
        self._chat_inspect_bar = actions
        ttk.Label(self.page, text="Workspace: artifacts/worker-sandbox · approved tools only · no shell · bounded RUN_TESTS · current-user permissions · not an OS sandbox", style="Muted.TLabel").pack(anchor="w", pady=8)
        submit_button.configure(state="normal")
        self._refresh_dev_cancel_button()
        self._render_selected_dev_job()
        threading.Thread(target=self._load_dev_jobs_background, daemon=True).start()

    def _get_dev_worker(self):
        if self._dev_worker is not None:
            return self._dev_worker
        with self._dev_worker_lock:
            if self._dev_worker is None:
                project_root = Path(__file__).resolve().parents[1]
                sandbox_parent = project_root / "artifacts"
                sandbox = sandbox_parent / "worker-sandbox"
                for path in (project_root, sandbox_parent, sandbox):
                    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                        raise ValueError("Local worker workspace cannot contain a link or junction")
                sandbox.mkdir(parents=True, exist_ok=True)
                config = load_config(project_root / "config" / "dante" / "base.yaml",
                                     overrides={"workspace": str(sandbox)})
                adapter = DevelopmentQwenAdapter()
                model = discover_development_qwen(adapter)
                db_path = project_root / config.database_path
                self._dev_worker = DevelopmentWorkerService(db_path, adapter, model, sandbox)
        return self._dev_worker

    def _render_dev_job(self, text):
        self._dev_last_display = text
        widget = getattr(self, "_chat_output", None)
        if widget is None or not widget.winfo_exists():
            return
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _render_dev_inspectables(self, job_id):
        info = self._dev_jobs.get(job_id) or {}
        events = info.get("events") or []
        record = {**info, "job": info.get("job")}
        bar = getattr(self, "_chat_inspect_bar", None)
        if bar is None or not bar.winfo_exists():
            return
        for child in bar.winfo_children():
            child.destroy()
        self._chat_inspect_actions = []
        options = (("Job", "job"), ("Model", "model"), ("Runtime", "runtime"), ("Queue", "queue"),
                   ("Model calls", "model_calls"), ("Tool calls", "tool_calls"),
                   ("Tests", "tests"), ("Failure", "failure"))
        for label, kind in options:
            button = ttk.Button(bar, text=f"Inspect {label}", command=lambda k=kind, i=info, e=events, j=job_id:
                                self._show_inspector(inspector_payload(k, j, i, e)))
            button.pack(side="left", padx=3, pady=3)
            self._chat_inspect_actions.append(button)
        self._show_inspector(inspector_payload("job", job_id, record, events))

    def _submit_dev_job(self):
        text = self._chat_objective.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Local Qwen", "Enter a bounded task.")
            return
        self._dev_submit_pending += 1
        if not self._dev_jobs:
            self._render_dev_job("SUBMITTING · LOCAL 11434 · existing single-slot queue\nTESTS: NOT RUN YET · bounded RUN_TESTS available")
        threading.Thread(target=self._submit_dev_job_background, args=(text,), daemon=True).start()

    def _submit_dev_job_background(self, objective):
        try:
            worker = self._get_dev_worker()
            job = worker.submit(objective)
            self.root.after(0, lambda: self._dev_job_submitted(job))
        except Exception as exc:
            message = f"SUBMIT ERROR · {type(exc).__name__}: {exc}"
            self.root.after(0, lambda msg=message: self._dev_submit_failed(msg))

    def _dev_job_submitted(self, job):
        self._dev_submit_pending = max(0, self._dev_submit_pending - 1)
        job_id = str(job.job_id)
        self._dev_jobs[job_id] = {"job": job, "state": job.state.value, "telemetry": None,
                                  "text": f"JOB: {job_id}\nSTATE: {job.state.value}\nQUEUE POSITION: {self._unknown()}\nTESTS: NOT RUN YET · bounded RUN_TESTS available"}
        self._dev_job_started[job_id] = time.monotonic()
        self._dev_selected_job_id = job_id
        self._upsert_dev_job_row(job_id)
        tree = getattr(self, "_chat_jobs_tree", None)
        if tree is not None and tree.winfo_exists():
            tree.selection_set(job_id)
            tree.focus(job_id)
        self._refresh_dev_cancel_button()
        self._render_selected_dev_job()
        self._schedule_dev_poll(0)

    def _dev_submit_failed(self, message):
        self._dev_submit_pending = max(0, self._dev_submit_pending - 1)
        if not self._dev_jobs:
            self._render_dev_job(message)

    def _job_is_live(self, job_id):
        info = self._dev_jobs.get(job_id) or {}
        return info.get("state") in {"queued", "claimed", "running", "cancel_requested", "retry_wait"}

    def _refresh_dev_cancel_button(self):
        widget = getattr(self, "_chat_cancel_button", None)
        if widget is not None and widget.winfo_exists():
            selected = self._dev_selected_job_id
            widget.configure(state="normal" if selected and self._job_is_live(selected) else "disabled")

    def _upsert_dev_job_row(self, job_id):
        tree = getattr(self, "_chat_jobs_tree", None)
        if tree is None or not tree.winfo_exists():
            return
        info = self._dev_jobs.get(job_id) or {}
        job = info.get("job")
        state = info.get("state") or getattr(getattr(job, "state", None), "value", "unknown")
        telemetry = info.get("telemetry") or {}
        state_label = development_job_state_label(state, telemetry, job_id)
        queue = telemetry.get("position")
        depth = telemetry.get("queue_depth")
        capacity = telemetry.get("capacity")
        queue_text = (f"{queue if queue is not None else self._unknown()} / {depth if depth is not None else self._unknown()} · cap {capacity if capacity is not None else self._unknown()}")
        values = (state_label, job_id, queue_text)
        if tree.exists(job_id):
            tree.item(job_id, values=values)
        else:
            tree.insert("", "end", iid=job_id, values=values)

    def _select_dev_job(self, _event=None):
        tree = getattr(self, "_chat_jobs_tree", None)
        selection = tree.selection() if tree is not None and tree.winfo_exists() else ()
        if selection:
            self._dev_selected_job_id = selection[0]
            self._refresh_dev_cancel_button()
            self._render_selected_dev_job()

    def _render_selected_dev_job(self):
        selected = self._dev_selected_job_id
        info = self._dev_jobs.get(selected) if selected else None
        if info:
            self._render_dev_job(info.get("text", f"JOB: {selected}\nSTATUS: {self._unknown()}"))
            self._render_dev_inspectables(selected)
        elif not self._dev_jobs:
            self._render_dev_job(self._dev_last_display if hasattr(self, "_dev_last_display") else "Ready · Ollama 11434")

    def _load_dev_jobs_background(self):
        try:
            jobs = self._get_dev_worker().list_jobs(limit=25)
            self.root.after(0, lambda: self._dev_jobs_loaded(jobs))
        except Exception:
            return

    def _dev_jobs_loaded(self, jobs):
        for job in reversed(jobs):
            job_id = str(job.job_id)
            self._dev_jobs.setdefault(job_id, {"job": job, "state": job.state.value, "telemetry": None,
                                               "details_loaded": False,
                                               "text": f"JOB: {job_id}\nSTATE: {job.state.value}\nLoading durable telemetry…"})
            self._upsert_dev_job_row(job_id)
        if self._dev_selected_job_id is None and self._dev_jobs:
            self._dev_selected_job_id = next(reversed(self._dev_jobs))
            tree = getattr(self, "_chat_jobs_tree", None)
            if tree is not None and tree.winfo_exists():
                tree.selection_set(self._dev_selected_job_id)
        self._render_selected_dev_job()
        self._schedule_dev_poll(0)

    @staticmethod
    def _unknown():
        return "Unknown (not reported)"

    def _schedule_dev_poll(self, delay_ms=1500):
        if any(self._job_is_live(job_id) or not info.get("details_loaded")
               for job_id, info in self._dev_jobs.items()) and self._dev_poll_after_id is None:
            self._dev_poll_after_id = self.root.after(delay_ms, self._poll_dev_job)

    def _poll_dev_job(self):
        self._dev_poll_after_id = None
        for job_id in tuple(self._dev_jobs):
            info = self._dev_jobs.get(job_id) or {}
            if (self._job_is_live(job_id) or not info.get("details_loaded")) and job_id not in self._dev_poll_inflight:
                self._dev_poll_inflight.add(job_id)
                threading.Thread(target=self._poll_dev_job_background, args=(job_id,), daemon=True).start()

    def _poll_dev_job_background(self, job_id):
        try:
            worker = self._get_dev_worker()
            job = worker.get(job_id)
            events = worker.events(job_id)
            telemetry = worker.queue_telemetry(job_id)
            started = self._dev_job_started.get(job_id)
            elapsed = time.monotonic() - started if started is not None else None
            metrics = development_job_metrics(events)
            state_label = development_job_state_label(job.state.value, telemetry, job_id)
            lines = [f"JOB: {job_id}", f"STATE: {state_label}",
                     *development_queue_lines(telemetry, job_id, job.state.value),
                     f"ELAPSED: {elapsed:.1f}s" if elapsed is not None else f"ELAPSED: {self._unknown()}",
                     f"CURRENT OPERATION: {metrics['current_operation']}",
                     f"MODEL CALLS: {metrics['model_calls']} · TOOL CALLS: {metrics['tool_calls']}",
                     f"INPUT TOKENS (SUM ACROSS CALLS): {metrics['input_tokens']}",
                     f"OUTPUT TOKENS (SUM ACROSS CALLS): {metrics['output_tokens']}",
                     "MODEL: dante-qwen-agent:latest · http://127.0.0.1:11434",
                     "TTFT / cached tokens / output tokens per second: Unknown (not reported)",
                     f"TESTS: {development_test_status(events)}", "", "TIMELINE"]
            for event in events[-40:]:
                name = event.get("event", self._unknown())
                metadata = event.get("metadata") or {}
                detail = ""
                if name == "tool.completed":
                    detail = f"  {metadata.get('tool_id', self._unknown())} · {metadata.get('status', self._unknown())}"
                elif name == "model.completed":
                    detail = (f"  call {metadata.get('model_calls', self._unknown())} · "
                              f"in {metadata.get('input_tokens', self._unknown())} · "
                              f"out {metadata.get('output_tokens', self._unknown())}")
                lines.append(f"{event.get('timestamp_utc', self._unknown())}  {name}{detail}")
            terminal = job.state.value in {"succeeded", "failed", "cancelled", "timed_out"}
            if job.result is not None:
                lines.extend(["", "RESULT", json.dumps(job.result, ensure_ascii=False, indent=2, default=str)])
            if job.failure is not None:
                lines.extend(["", "FAILURE", json.dumps(job.failure, ensure_ascii=False, indent=2, default=str)])
            self.root.after(0, lambda: self._dev_job_updated(job_id, job, telemetry, "\n".join(lines), terminal, events))
        except Exception as exc:
            message = f"JOB: {job_id}\nPOLL ERROR: {type(exc).__name__}: {exc}\nPolling will retry."
            self.root.after(0, lambda msg=message: self._dev_job_updated(job_id, None, None, msg, False))

    def _dev_job_updated(self, job_id, job, telemetry, text, terminal, events=None):
        self._dev_poll_inflight.discard(job_id)
        info = self._dev_jobs.setdefault(job_id, {})
        if job is not None:
            info["job"] = job
            info["state"] = job.state.value
        if telemetry is not None:
            info["telemetry"] = telemetry
        if events is not None:
            info["events"] = events
        if job is not None and telemetry is not None:
            info["details_loaded"] = True
        info["text"] = text
        self._upsert_dev_job_row(job_id)
        if self._dev_selected_job_id == job_id:
            self._render_dev_job(text)
            self._render_dev_inspectables(job_id)
            self._refresh_dev_cancel_button()
        if terminal:
            self._dev_job_started.pop(job_id, None)
        self._schedule_dev_poll()

    def _cancel_dev_job(self):
        job_id = self._dev_selected_job_id
        if not job_id or not self._job_is_live(job_id):
            return
        self._chat_cancel_button.configure(state="disabled")
        if self._dev_selected_job_id == job_id:
            self._render_dev_job(f"JOB: {job_id}\nCANCEL REQUESTED\nTESTS: current state remains in event timeline")
        def cancel_background():
            job = None
            try:
                job = self._get_dev_worker().cancel(job_id)
                message = f"JOB: {job_id}\nCANCEL REQUEST STATE: {job.state.value}"
            except Exception as exc:
                message = f"JOB: {job_id}\nCANCEL ERROR: {type(exc).__name__}: {exc}"
            self.root.after(0, lambda msg=message: self._dev_job_cancelled(job_id, job, msg))
            self.root.after(0, self._schedule_dev_poll)
        threading.Thread(target=cancel_background, daemon=True).start()

    def _dev_job_cancelled(self, job_id, job, message):
        info = self._dev_jobs.setdefault(job_id, {})
        if job is not None:
            info["job"] = job
            info["state"] = job.state.value
        info["text"] = message
        self._upsert_dev_job_row(job_id)
        if self._dev_selected_job_id == job_id:
            self._render_dev_job(message)
            self._refresh_dev_cancel_button()

    def page_agents(self):
        tree = self._tree(("name", "agent_id", "role", "model_target"), ("Name", "Agent ID", "Role", "Local model"))
        for a in self.agents:
            tree.insert("", "end", values=(a.get("name"), a.get("agent_id"), a.get("role"), a.get("model_target", {}).get("model_id", "")))
        tree.bind("<<TreeviewSelect>>", lambda _e: self._inspect_selected_row(tree, "agent"))
        tree.bind("<Double-1>", lambda _e: self._inspect_selected_row(tree, "agent"))

    def page_jobs(self):
        tree = self._tree(
            ("state", "job_id", "agent", "model", "endpoint", "priority", "queue", "elapsed", "attempts", "deadline", "failure"),
            ("State / phase", "Job ID", "Agent", "Model", "Endpoint", "Priority", "Queue position", "Elapsed wall", "Attempts", "Deadline", "Failure"))
        widths = {"state": 125, "job_id": 190, "agent": 135, "model": 150, "endpoint": 155,
                  "priority": 75, "queue": 100, "elapsed": 100, "attempts": 75, "deadline": 150, "failure": 120}
        for column, width in widths.items():
            tree.column(column, width=width, stretch=False)
        filter_row = ttk.Frame(self.page)
        filter_row.pack(fill="x", pady=(0, 8))
        ttk.Label(filter_row, text="Filter", style="Muted.TLabel").pack(side="left")
        filter_values = ("ALL", "RUNNING", "QUEUED", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN")
        current = getattr(self, "_jobs_filter", "ALL")
        if current not in filter_values:
            current = "ALL"
        filter_var = tk.StringVar(self.page, value=current)
        self._jobs_filter = current
        filter_box = ttk.Combobox(filter_row, textvariable=filter_var, state="readonly", values=list(filter_values), width=12)
        filter_box.pack(side="left", padx=8)
        def on_filter(_e):
            self._jobs_filter = filter_var.get()
            self.show_page("Jobs")
        filter_box.bind("<<ComboboxSelected>>", on_filter)
        for j in self.jobs:
            if self._jobs_filter != "ALL" and job_group(j) != self._jobs_filter:
                continue
            state = job_phase(j)
            tree.insert("", "end", iid=j.get("job_id"), values=(
                state, _known(j.get("job_id")), "Unknown (not reported)", "Unknown (not reported)",
                "Unknown (not reported)", _known(j.get("priority")), "Unknown (not reported)",
                elapsed_wall(j), _known(j.get("attempt_count")), _known(j.get("deadline_at")),
                _known((j.get("failure") or {}).get("code"))))
        tree.bind("<<TreeviewSelect>>", lambda _e: self._inspect_control_job(self._selected_job(tree)))
        tree.bind("<Double-1>", lambda _e: self._inspect_control_job(self._selected_job(tree)))
        scroll = ttk.Scrollbar(self.page, orient="horizontal", command=tree.xview)
        scroll.pack(fill="x")
        tree.configure(xscrollcommand=scroll.set)
        bar = ttk.Frame(self.page)
        bar.pack(fill="x", pady=8)
        ttk.Button(bar, text="Inspect job / events", command=lambda: self._inspect_control_job(self._selected_job(tree))).pack(side="left")
        ttk.Button(bar, text="Cancel / stop selected", command=lambda: self.cancel_job(self._selected_job(tree))).pack(side="left", padx=8)
        ttk.Label(bar, text="Retry unavailable: Node 0 API has no retry operation.", style="Muted.TLabel").pack(side="left", padx=8)

    def _inspect_control_job(self, job):
        if not job:
            return
        def fetch():
            return self._request("workload_events", job_id=job["job_id"])
        def run():
            try:
                events = fetch()
                payload = inspector_payload("job", job["job_id"], job, events)
                self.root.after(0, lambda: self._show_inspector(payload))
            except Exception as exc:
                payload = inspector_payload("failure", job["job_id"], {"error": type(exc).__name__})
                self.root.after(0, lambda: self._show_inspector(payload))
        threading.Thread(target=run, daemon=True).start()

    def _selected_job(self, tree):
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Jobs", "Select a job first.")
            return None
        return next((j for j in self.jobs if j.get("job_id") == sel[0]), None)

    def show_job_events(self, job):
        if not job:
            return
        def fetch():
            raw = self._request("workload_events", job_id=job["job_id"])
            return {"job": job, "events": raw}
        self._action(fetch, f"Events · {job['job_id']}")

    @staticmethod
    def _operation_text(job):
        events = job.get("events") or []
        if not events:
            state, _ = job_label(job)
            return f"Current state: {state}; detailed activity not yet available"
        latest = events[-1]
        name = latest.get("event", "unknown event")
        metadata = latest.get("metadata") or {}
        if "tool" in name.lower():
            phase = "TOOL"
        elif "wait" in name.lower() or "queue" in name.lower():
            phase = "WAITING"
        elif "retry" in name.lower():
            phase = "RETRY / BACKOFF"
        else:
            phase, _ = job_label(job)
        extra = metadata.get("tool_name") or metadata.get("operation") or metadata.get("code")
        return f"{phase} · {name}" + (f" · {extra}" if extra else "")

    def cancel_job(self, job):
        if not job:
            return
        self._action(lambda: self._request("workload_cancel", job_id=job["job_id"]), "Stop requested")

    def _action(self, action, title, output=None):
        def run():
            try:
                value = action()
                if title == "Job submitted":
                    text = job_submission_text(value)
                elif isinstance(value, dict) and isinstance(value.get("job"), dict):
                    text = job_details_text(value["job"], value.get("events") or [])
                else:
                    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
                if output:
                    self.root.after(0, lambda: (output.configure(state="normal"), output.delete("1.0", "end"), output.insert("end", text), output.configure(state="disabled")))
                else:
                    self.root.after(0, lambda: messagebox.showinfo(title, text[:5000]))
                LOG.info("%s: %s", title, text[:2000])
                self.root.after(0, self.refresh)
            except Exception as exc:
                msg = getattr(exc, "code", type(exc).__name__)
                LOG.exception("%s failed", title)
                self.root.after(0, lambda: messagebox.showerror(title, str(msg)))
        threading.Thread(target=run, daemon=True).start()

    def page_models(self):
        rows = build_model_rows(self.snapshot, self.ollama)
        tree = self._tree(
            ("model", "endpoint", "availability", "residency", "quantization", "max_context", "runtime_context", "vram", "size", "slot", "configured_context"),
            ("Model", "Runtime endpoint", "Availability", "Loaded", "Quantization",
             "Max context (metadata)", "Runtime context (loaded)", "VRAM", "Size", "Slot state", "OpenCode configured context"))
        for row in rows:
            tree.insert("", "end", values=row)
        tree.bind("<<TreeviewSelect>>", lambda _e: self._inspect_selected_row(tree, "model"))
        tree.bind("<Double-1>", lambda _e: self._inspect_selected_row(tree, "model"))
        if not rows:
            ttk.Label(self.page, text="No model inventory reported. Development runtime status is on Settings.", style="Muted.TLabel").pack(anchor="w", pady=10)

    def page_hardware(self):
        snap = self.snapshot
        tree = self._tree(("component", "fact", "value", "source"), ("Component", "Measurement", "Value", "Source"))
        for row in hardware_rows(snap, self.ollama):
            tree.insert("", "end", values=row)
        tree.bind("<<TreeviewSelect>>", lambda _e: self._inspect_selected_row(tree, "hardware"))
        tree.bind("<Double-1>", lambda _e: self._inspect_selected_row(tree, "hardware"))
        note = (f"Snapshot {snap.observed_at.isoformat()} · {snap.probe_version} · read-only · unknown facts are shown explicitly"
                if snap else "Hardware snapshot unavailable · Development Ollama status is read-only")
        ttk.Label(self.page, text=note, style="Muted.TLabel").pack(anchor="w", pady=8)

    def _inspect_selected_row(self, tree, kind):
        selection = tree.selection()
        if not selection:
            return
        values = tree.item(selection[0], "values")
        if kind == "model":
            record = {"availability": values[2], "residency": values[3], "quantization": values[4],
                      "max_context": values[5], "runtime_context": values[6],
                      "size": values[8], "context": values[10]}
            self._show_inspector(inspector_payload("model", values[0], record))
        elif kind == "agent":
            record = {"state": values[2], "job_id": "Unknown (not reported)", "model": values[3]}
            self._show_inspector(inspector_payload("agent", values[1], record))
        else:
            record = {"value": values[2], "source": values[3],
                      "observed_at": self.snapshot.observed_at.isoformat() if self.snapshot else "Unknown (not reported)"}
            self._show_inspector(inspector_payload("hardware", f"{values[0]} · {values[1]}", record))

    def page_logs(self):
        filter_row = ttk.Frame(self.page)
        filter_row.pack(fill="x", pady=(0, 8))
        id_box = ttk.Frame(filter_row)
        id_box.pack(side="left")
        ttk.Label(id_box, text="Job ID", style="Muted.TLabel").pack(side="left")
        group_box = ttk.Frame(filter_row)
        group_box.pack(side="left", padx=16, fill="x", expand=True)
        ttk.Label(group_box, text="Lifecycle group", style="Muted.TLabel").pack(side="left")
        groups = ("ALL", "RUNNING", "QUEUED", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN")
        entry_var = tk.StringVar(self.page)
        entry = tk.Entry(id_box, width=22, textvariable=entry_var)
        entry.pack(side="left", padx=8)
        filter_var = tk.StringVar(self.page, value="ALL")
        filter_box = ttk.Combobox(group_box, textvariable=filter_var, state="readonly", values=list(groups), width=12)
        filter_box.pack(side="left", padx=8)
        box = tk.Text(self.page, bg="#171d27", fg="#cdd6e2", relief="flat", wrap="none")
        box.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(self.page, orient="horizontal", command=box.xview)
        scroll.pack(fill="x")
        box.configure(xscrollcommand=scroll.set)
        def render(*_args):
            rows = filter_log_jobs(self.jobs, job_id=entry_var.get(), group=filter_var.get())
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("end", "Recent durable job activity (summary view of already-reported job records)\n\n")
            for job in rows:
                box.insert("end", f"{job.get('updated_at')}  {job_label(job)[0]}  {job.get('job_id')}\n")
            if not rows:
                box.insert("end", "(no matching job records)\n")
            box.insert("end", "\nAgent, severity, component, and event-time filters are unavailable in this summary view; the recent-job summary keeps only job_id / updated_at / state label.\n")
            box.insert("end", "Durable event details remain in Jobs → Details / events.\n")
            box.configure(state="disabled")
        entry_var.trace_add("write", render)
        filter_box.bind("<<ComboboxSelected>>", render)
        render()

    def page_settings(self):
        state = self.ollama
        limits = read_opencode_limits()
        row = ttk.Frame(self.page)
        row.pack(fill="x", pady=(0, 12))
        online = state.get("online")
        self._panel(row, "Development Ollama", "ONLINE" if online else "OFFLINE", f"Get-only checks of {state.get('endpoint', OLLAMA_ENDPOINT)}").pack(side="left", fill="x", expand=True, padx=(0, 9))
        self._panel(row, "Endpoint", state.get("endpoint", OLLAMA_ENDPOINT), "Local development runtime only. No cloud fallback.").pack(side="left", fill="x", expand=True, padx=(0, 9))
        self._panel(row, "Policy", "LOCAL ONLY", "Provider and endpoint are fixed. No cloud fallback, no load/unload actions. Read-only status: GET /api/tags and /api/ps only.").pack(side="left", fill="x", expand=True)

        ttk.Label(self.page, text="EXECUTION POLICY", style="PanelTitle.TLabel").pack(anchor="w", pady=(8, 7))
        policy_facts = (
            ("Local first", "ON", "Development worker only"),
            ("Cloud fallback", "OFF", "No remote inference fallback"),
            ("Production Node0", "127.0.0.1:11435", "Protected; no GUI requests or controls"),
            ("Qwen heavy slots", "1", "Single local inference at a time"),
            ("OpenCode timeout", f"{limits['timeout']} ms", f"Header timeout: {limits['header_timeout']} ms"),
            ("Qwen configured limits", f"{limits['context']} context", f"Output: {limits['output']} tokens"),
        )
        for offset in (0, 3):
            policy = ttk.Frame(self.page)
            policy.pack(fill="x", pady=(0, 8))
            for title, value, detail in policy_facts[offset:offset + 3]:
                self._panel(policy, title, value, detail).pack(side="left", fill="x", expand=True, padx=(0, 8))

        ttk.Label(self.page, text="INSTALLED MODELS", style="PanelTitle.TLabel").pack(anchor="w", pady=(8, 7))
        installed = self._tree(("name", "size"), ("Model", "Size"))
        for m in state.get("installed") or []:
            installed.insert("", "end", values=(m["name"], m["size"]))
        if not state.get("installed"):
            ttk.Label(self.page, text="No installed model reported (or Development Ollama offline).", style="Muted.TLabel").pack(anchor="w", pady=(0, 8))

        ttk.Label(self.page, text="RESIDENT MODELS (LOADED IN VRAM)", style="PanelTitle.TLabel").pack(anchor="w", pady=(10, 7))
        resident = self._tree(("name", "size_vram", "context_length"), ("Model", "Size VRAM", "Context length"))
        for m in state.get("resident") or []:
            resident.insert("", "end", values=(m["name"], m["size_vram"], m["context_length"]))
        if not state.get("resident"):
            ttk.Label(self.page, text="No resident model reported.", style="Muted.TLabel").pack(anchor="w", pady=(0, 8))
        errors = state.get("errors") or []
        ttk.Label(self.page, text="Last checks: " + ("; ".join(errors) if errors else f"{state.get('endpoint', OLLAMA_ENDPOINT)}/api/tags and /api/ps OK"), style="Muted.TLabel").pack(anchor="w", pady=10)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = tk.Tk()
    ControlCenterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
