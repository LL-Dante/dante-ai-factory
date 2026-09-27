"""Local desktop Control Center for the existing Node 0 APIs.

The UI is a presentation layer. Agent execution, job state, qualification and
hardware facts remain owned by Dante's existing control plane and contracts.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import tkinter as tk
from datetime import datetime, timezone
from tkinter import messagebox, ttk

from dante.hardware_inventory import collect_snapshot
from dante.node0_control import ControlError, Node0ControlClient


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


def build_model_rows(snapshot, ollama_state):
    """Assemble Models-page rows; metadata max context and resident runtime
    context are separate columns and missing facts stay explicitly unknown."""
    rows, seen = [], set()
    for runtime in (snapshot.runtimes if snapshot else ()):
        for model in runtime.installed_models:
            rows.append((model.runtime_reference, runtime.endpoint, "Unknown (not reported)", "Unknown (not reported)",
                         display_fact(model.quantization), "Unknown (not reported)", "Unknown (not reported)",
                         "Unknown (not reported)",
                         display_fact(model.size_bytes, scale=1 / (1024 ** 3), suffix=" GiB"), "Unknown (not reported)"))
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
        rows.append((name, OLLAMA_ENDPOINT, "AVAILABLE" if online else "OFFLINE",
                     "LOADED" if resident else ("NOT LOADED" if online else "Unknown (not reported)"),
                     model.get("quantization", "Unknown (not reported)"),
                     model.get("max_context", "Unknown (not reported)"),
                     runtime_context,
                     resident.get("size_vram", "Unknown (not reported)") if resident else "Unknown (not reported)",
                     model.get("size", "Unknown (not reported)"), "Unknown (not reported)"))
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


class ControlCenterApp:
    POLL_SECONDS = 3

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dante AI Factory · Control Center")
        self.root.geometry("1240x820")
        self.root.minsize(980, 680)
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
        self.show_page(self.current_page)
        self.root.after(self.POLL_SECONDS * 1000, self.refresh)

    def show_page(self, name):
        self.current_page = name
        self.page_title.configure(text=name)
        for child in self.page.winfo_children():
            child.destroy()
        getattr(self, f"page_{name.lower()}")()

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
        ttk.Label(self.page, text="Local agent chat", style="PanelTitle.TLabel").pack(anchor="w")
        form = ttk.Frame(self.page, style="Panel.TFrame", padding=14)
        form.pack(fill="x", pady=10)
        ttk.Label(form, text="Registered local agent", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        agent_var = tk.StringVar(value=self.agents[0].get("agent_id", "") if self.agents else "")
        ttk.Combobox(form, textvariable=agent_var, values=[a.get("agent_id", "") for a in self.agents], state="readonly", width=36).grid(row=0, column=1, sticky="w", padx=10)
        ttk.Label(form, text="Objective", style="Muted.TLabel").grid(row=1, column=0, sticky="nw", pady=10)
        objective = tk.Text(form, height=5, bg="#11151d", fg="#e6eaf0", insertbackground="white", relief="flat", wrap="word")
        objective.grid(row=1, column=1, sticky="ew", padx=10, pady=10)
        form.columnconfigure(1, weight=1)
        output = tk.Text(self.page, bg="#171d27", fg="#e6eaf0", relief="flat", wrap="word", state="disabled")
        output.pack(fill="both", expand=True)
        ttk.Label(self.page, text="Jobs execute LOCAL_ONLY through Node 0. No cloud fallback is available.", style="Muted.TLabel").pack(anchor="w", pady=8)
        def submit():
            text = objective.get("1.0", "end").strip()
            if not agent_var.get() or not text:
                messagebox.showinfo("Chat", "Choose a local agent and enter an objective.")
                return
            self._action(lambda: self._request("agent_submit", agent_id=agent_var.get(), objective=text), "Job submitted", output)
        ttk.Button(form, text="Submit local job", command=submit).grid(row=2, column=1, sticky="e")

    def page_agents(self):
        tree = self._tree(("name", "agent_id", "role", "model_target"), ("Name", "Agent ID", "Role", "Local model"))
        for a in self.agents:
            tree.insert("", "end", values=(a.get("name"), a.get("agent_id"), a.get("role"), a.get("model_target", {}).get("model_id", "")))

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
        tree.bind("<Double-1>", lambda _e: self.show_job_events(self._selected_job(tree)))
        scroll = ttk.Scrollbar(self.page, orient="horizontal", command=tree.xview)
        scroll.pack(fill="x")
        tree.configure(xscrollcommand=scroll.set)
        bar = ttk.Frame(self.page)
        bar.pack(fill="x", pady=8)
        ttk.Button(bar, text="Details / events", command=lambda: self.show_job_events(self._selected_job(tree))).pack(side="left")
        ttk.Button(bar, text="Cancel / stop selected", command=lambda: self.cancel_job(self._selected_job(tree))).pack(side="left", padx=8)
        ttk.Label(bar, text="Retry unavailable: Node 0 API has no retry operation.", style="Muted.TLabel").pack(side="left", padx=8)

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
            ("model", "endpoint", "availability", "residency", "quantization", "max_context", "runtime_context", "vram", "size", "slot"),
            ("Model", "Runtime endpoint", "Availability", "Loaded", "Quantization",
             "Max context (metadata)", "Runtime context (loaded)", "VRAM", "Size", "Slot state"))
        for row in rows:
            tree.insert("", "end", values=row)
        if not rows:
            ttk.Label(self.page, text="No model inventory reported. Development runtime status is on Settings.", style="Muted.TLabel").pack(anchor="w", pady=10)

    def page_hardware(self):
        snap = self.snapshot
        tree = self._tree(("component", "fact", "value", "source"), ("Component", "Measurement", "Value", "Source"))
        for row in hardware_rows(snap, self.ollama):
            tree.insert("", "end", values=row)
        note = (f"Snapshot {snap.observed_at.isoformat()} · {snap.probe_version} · read-only · unknown facts are shown explicitly"
                if snap else "Hardware snapshot unavailable · Development Ollama status is read-only")
        ttk.Label(self.page, text=note, style="Muted.TLabel").pack(anchor="w", pady=8)

    def page_logs(self):
        box = tk.Text(self.page, bg="#171d27", fg="#cdd6e2", relief="flat", wrap="none")
        box.pack(fill="both", expand=True)
        box.insert("end", "Recent durable job activity\n\n")
        for job in self.jobs:
            box.insert("end", f"{job.get('updated_at')}  {job_label(job)[0]}  {job.get('job_id')}\n")
        box.insert("end", "\nDetailed event metadata is available in Jobs → Details / events.\n")
        box.configure(state="disabled")

    def page_settings(self):
        state = self.ollama
        row = ttk.Frame(self.page)
        row.pack(fill="x", pady=(0, 12))
        online = state.get("online")
        self._panel(row, "Development Ollama", "ONLINE" if online else "OFFLINE", f"Get-only checks of {state.get('endpoint', OLLAMA_ENDPOINT)}").pack(side="left", fill="x", expand=True, padx=(0, 9))
        self._panel(row, "Endpoint", state.get("endpoint", OLLAMA_ENDPOINT), "Local development runtime only. No cloud fallback.").pack(side="left", fill="x", expand=True, padx=(0, 9))
        self._panel(row, "Policy", "LOCAL ONLY", "Provider and endpoint are fixed. No cloud fallback, no load/unload actions. Read-only status: GET /api/tags and /api/ps only.").pack(side="left", fill="x", expand=True)

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
