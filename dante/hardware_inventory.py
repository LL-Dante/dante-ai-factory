"""Read-only hardware and runtime inventory for the local node.

Nothing in this module mutates the machine. There is no clock, power-limit, BIOS,
service, task, model-store, qualification or configuration write path here, and no
code path can start or stop a runtime: runtimes are only ever asked for their
version and inventory over read-only HTTP.

Reuse is deliberate. Live sensor families (CPU/GPU utilization, temperature, power,
VRAM in use, memory totals) come from the existing monitoring subsystem instead of a
second poller. GPU identity comes from the existing NVIDIA probes. Runtime version
and model inventory come from the existing bounded Ollama probe. This module only
collects what those sources do not already provide.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dante.contracts.hardware import (CpuFacts, DiskFacts, Fact, GpuFacts, HardwareCapabilityProfile,
                                      HardwareSnapshot, MemoryFacts, ModelCapacityFact, ModelStoreFacts,
                                      RuntimeFacts, RuntimeModelFacts, SensorCoverage, SystemFacts,
                                      VolumeFacts)
from dante.contracts.runtime import RuntimeProfile
from dante.inference import (AdapterUnavailable, InferenceError, InferenceTimeout,
                             InvalidResponse, PolicyDenied, RateQuotaUnavailable)
from dante.local_runtime import OllamaAdapter
from dante.nvidia_probe import discover_gpu_uuids, discover_nvidia
from dante.recovery import digest

PROBE_VERSION = 'hardware-v1'
MONITORING_ENDPOINT = 'http://127.0.0.1:8765/api/status'
MONITORING_STATE = Path(r'C:\DanteAI\monitoring\state\current.json')
# Dynamic readings older than this are not presented as current conditions.
MAX_SENSOR_AGE_S = 300
MAX_OUTPUT_BYTES = 256 * 1024
MAX_MODEL_STORE_FILES = 20000
UNKNOWN = 'unknown'
NVIDIA = 'nvidia-smi'
WINDOWS = 'windows-cim'
POWERCFG = 'powercfg'
OLLAMA = 'ollama-http'
STORE = 'model-store'
DERIVED = 'derived'

_GUID = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
_LABELLED = re.compile(r'\(([^)]{1,80})\)')
# SMBIOS memory technology codes; anything else is reported as the raw code.
_SMBIS_MEMORY_TYPE = {26: 'ddr4', 34: 'ddr5', 24: 'ddr3', 20: 'ddr2', 17: 'ddr'}

_WMI_SCRIPT = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$os=Get-CimInstance Win32_OperatingSystem;"
    "$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1;"
    "$mem=@(Get-CimInstance Win32_PhysicalMemory);"
    "$disk=@(Get-CimInstance Win32_DiskDrive);"
    "$vol=@(Get-CimInstance Win32_LogicalDisk -Filter \"DriveType=3\");"
    "$plan=(powercfg /GETACTIVESCHEME);"
    "$guid='';$name='';"
    "if($plan){$m=[regex]::Match($plan,'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}');"
    "if($m.Success){$guid=$m.Value;$n=[regex]::Match($plan,'\\(([^)]{1,80})\\)');if($n.Success){$name=$n.Groups[1].Value}}};"
    "$boot='';if($os.LastBootUpTime){$boot=$os.LastBootUpTime.ToUniversalTime().ToString('o')};"
    "[pscustomobject]@{"
    "os_name=$os.Caption;os_version=$os.Version;os_build=$os.BuildNumber;architecture=$os.OSArchitecture;"
    "last_boot=$boot;"
    "cpu_name=$cpu.Name;cpu_cores=$cpu.NumberOfCores;cpu_threads=$cpu.NumberOfLogicalProcessors;"
    "cpu_max_mhz=$cpu.MaxClockSpeed;cpu_current_mhz=$cpu.CurrentClockSpeed;"
    "mem_modules=@($mem|ForEach-Object{[pscustomobject]@{locator=$_.DeviceLocator;bank=$_.BankLabel;"
    "bytes=$_.Capacity;speed=$_.Speed;configured=$_.ConfiguredClockSpeed;smbios=$_.SMBIOSMemoryType}});"
    "disks=@($disk|ForEach-Object{[pscustomobject]@{index=$_.Index;model=$_.Model;interface=$_.InterfaceType;"
    "media=$_.MediaType;size=$_.Size}});"
    "volumes=@($vol|ForEach-Object{[pscustomobject]@{mount=$_.DeviceID;fs=$_.FileSystem;size=$_.Size;free=$_.FreeSpace}});"
    "power_plan_guid=$guid;power_plan_name=$name} | ConvertTo-Json -Compress -Depth 5"
)

_NVIDIA_QUERY = ('index,memory.free,clocks.current.sm,clocks.current.memory,clocks.max.sm,'
                 'pcie.link.gen.current,pcie.link.width.current')
_NVIDIA_KEYS = ('memory_free_mib', 'sm_clock_mhz', 'memory_clock_mhz', 'max_sm_clock_mhz',
                'pcie_generation', 'pcie_link_width')


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text) if text.isdecimal() else float(text)
    except ValueError:
        return None


def _label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and len(value) <= 160 and all(ord(c) >= 32 for c in value) else None


def _key(value: Any) -> str | None:
    """Stringify a structural identifier such as a disk index, which WMI types as int."""
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    return text if text and len(text) <= 40 else None


def _items(value: Any) -> list[dict]:
    if isinstance(value, dict):
        return [value]
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _run_windows(script: str) -> dict | None:
    if os.name != 'nt' or shutil.which('powershell') is None:
        return None
    try:
        completed = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=20, check=False, shell=False,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or len(completed.stdout.encode('utf-8')) > MAX_OUTPUT_BYTES:
        return None
    try:
        parsed = json.loads(completed.stdout.strip() or 'null')
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _nvidia_dynamic(runner: Callable[[tuple[str, ...]], Any] | None = None) -> dict[str, dict[str, float | int | None]]:
    """Only the GPU fields the existing probes and monitoring do not already supply."""
    import csv
    import io

    def default(arguments: tuple[str, ...]):
        executable = shutil.which('nvidia-smi')
        if executable is None:
            return None
        return subprocess.run([executable, *arguments], stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=8, check=False, shell=False,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    call = runner or default
    try:
        output = call(('--query-gpu=' + _NVIDIA_QUERY, '--format=csv,noheader,nounits'))
    except (OSError, subprocess.SubprocessError):
        return {}
    if output is None or getattr(output, 'returncode', 1) != 0:
        return {}
    if len((output.stdout or '').encode('utf-8')) > MAX_OUTPUT_BYTES:
        return {}
    result: dict[str, dict[str, float | int | None]] = {}
    try:
        for row in csv.reader(io.StringIO(output.stdout), skipinitialspace=True):
            if len(row) != 7:
                return {}
            slot = row[0].strip()
            if not slot.isdecimal() or slot in result:
                return {}
            result[slot] = dict(zip(_NVIDIA_KEYS, (_number(cell) for cell in row[1:])))
    except (csv.Error, TypeError, ValueError):
        return {}
    return result


def _monitoring_live() -> dict | None:
    try:
        with urllib.request.urlopen(MONITORING_ENDPOINT, timeout=4) as response:
            if getattr(response, 'status', 200) != 200:
                return None
            body = response.read(MAX_OUTPUT_BYTES)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _monitoring_state() -> tuple[dict, str]:
    try:
        if not MONITORING_STATE.is_file():
            return {}, 'monitoring-state-absent'
        raw = MONITORING_STATE.read_text(encoding='utf-8')
    except (OSError, ValueError):
        return {}, 'monitoring-state-unreadable'
    if len(raw.encode('utf-8')) > MAX_OUTPUT_BYTES:
        return {}, 'monitoring-state-oversized'
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return {}, 'monitoring-state-invalid'
    return (payload, 'monitoring-state-file') if isinstance(payload, dict) else ({}, 'monitoring-state-invalid')


def _monitoring(fetch: Callable[[], tuple[dict, str]] | None = None) -> tuple[dict, str, datetime | None]:
    """Prefer the live monitoring service; otherwise use its persisted last sample."""
    getter = fetch or (lambda: (_monitoring_live(), 'monitoring-http') if _monitoring_live()
                       else _monitoring_state())
    try:
        payload, source = getter()
    except (OSError, ValueError, TypeError):
        return {}, UNKNOWN, None
    if not isinstance(payload, dict) or not payload:
        return {}, UNKNOWN, None
    observed = None
    stamp = payload.get('timestamp')
    if isinstance(stamp, str):
        try:
            observed = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
        except ValueError:
            observed = None
    return payload, source, observed


def _sensor(monitoring: dict | None, observed: datetime | None, section: str, key: str,
            scale: float = 1.0, unit: str | None = None, source: str = UNKNOWN) -> Fact:
    """Reuse a monitoring reading when it is present and still current."""
    block = monitoring.get(section) if isinstance(monitoring, dict) else None
    value = _number(block.get(key)) if isinstance(block, dict) else None
    if value is None or block is None:
        return Fact.unknown(f'{section}.{key} is not exposed by the monitoring subsystem', source=source)
    if observed is None:
        return Fact.unknown(f'{section}.{key} has no monitoring timestamp', source=source)
    age = (datetime.now(timezone.utc) - observed).total_seconds()
    if age > MAX_SENSOR_AGE_S:
        return Fact.unknown(f'{section}.{key} sample is {int(age)}s old, beyond the {MAX_SENSOR_AGE_S}s freshness bound',
                            source=source)
    scaled = value * scale if scale != 1.0 else value
    return Fact.measured(int(scaled) if float(scaled).is_integer() else round(scaled, 3),
                         source=source, unit=unit)


def _stale(monitoring: dict | None, observed: datetime | None) -> bool:
    if observed is None:
        return True
    return (datetime.now(timezone.utc) - observed).total_seconds() > MAX_SENSOR_AGE_S


def _system_facts(wmi: dict | None) -> SystemFacts:
    reason = 'Windows CIM inventory is unavailable on this host'
    source = WINDOWS

    def field(name: str, unit: str | None = None) -> Fact:
        value = _label(wmi.get(name)) if isinstance(wmi, dict) else None
        return Fact.measured(value, source=source) if value else Fact.unknown(reason, source=source)

    uptime = None
    boot = wmi.get('last_boot') if isinstance(wmi, dict) else None
    if isinstance(boot, str) and boot:
        try:
            started = datetime.fromisoformat(boot.replace('Z', '+00:00'))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            seconds = int((datetime.now(timezone.utc) - started).total_seconds())
            uptime = Fact.measured(max(seconds, 0), source=source, unit='s')
        except ValueError:
            uptime = None
    if uptime is None:
        uptime = Fact.unknown('Last boot time is unavailable', source=source)
    return SystemFacts(
        os_name=field('os_name'), os_version=field('os_version'), os_build=field('os_build'),
        architecture=field('architecture'), uptime_seconds=uptime,
        power_plan_guid=field('power_plan_guid'), power_plan_name=field('power_plan_name'))


def _cpu_facts(wmi: dict | None, monitoring: dict | None, observed: datetime | None) -> CpuFacts:
    reason = 'Windows CIM processor inventory is unavailable on this host'
    source = WINDOWS

    def field(name: str, unit: str | None = None) -> Fact:
        value = _number(wmi.get(name)) if isinstance(wmi, dict) else None
        return Fact.measured(value, source=source, unit=unit) if value is not None else Fact.unknown(reason, source=source)

    return CpuFacts(
        model=Fact.measured(_label(wmi.get('cpu_name')), source=source) if _label(wmi.get('cpu_name') if isinstance(wmi, dict) else None)
        else Fact.unknown(reason, source=source),
        physical_cores=field('cpu_cores'), logical_threads=field('cpu_threads'),
        base_clock_mhz=field('cpu_max_mhz', 'MHz'), current_clock_mhz=field('cpu_current_mhz', 'MHz'),
        utilization_percent=_sensor(monitoring, observed, 'cpu', 'utilization_percent', unit='%', source=UNKNOWN),
        temperature_c=_sensor(monitoring, observed, 'cpu', 'temperature_c', unit='C'))


def _memory_facts(wmi: dict | None, monitoring: dict | None, observed: datetime | None) -> MemoryFacts:
    source = WINDOWS
    reason = 'Windows CIM memory inventory is unavailable on this host'
    modules = _items(wmi.get('mem_modules')) if isinstance(wmi, dict) else []
    installed = sum(int(m['bytes']) for m in modules if _number(m.get('bytes')))
    if modules and installed:
        installed_fact = Fact.measured(installed, source=source, unit='B')
        module_count = Fact.measured(len(modules), source=source)
    else:
        installed_fact = Fact.unknown(reason, source=source)
        module_count = Fact.unknown(reason, source=source)
    speeds = {_number(m.get('configured')) or _number(m.get('speed')) for m in modules}
    speeds.discard(None)
    layout = _label('/'.join(sorted({_label(m.get('bank')) or _label(m.get('locator')) or '?' for m in modules}))) or None
    types = {_SMBIS_MEMORY_TYPE.get(int(m['smbios'])) for m in modules if _number(m.get('smbios')) is not None}
    types.discard(None)
    if len(types) == 1:
        memory_type = Fact.measured(next(iter(types)), source=source)
    else:
        memory_type = Fact.unknown('Mixed or unknown SMBIOS memory technology', source=source)
    return MemoryFacts(
        installed_bytes=installed_fact,
        available_bytes=_sensor(monitoring, observed, 'memory', 'available_mb', scale=1024 * 1024, unit='B'),
        speed_mhz=Fact.measured(next(iter(speeds)), source=source, unit='MHz') if len(speeds) == 1
        else Fact.unknown('Module speeds are mixed or unavailable', source=source),
        configured_speed_mhz=Fact.measured(next(iter(speeds)), source=source, unit='MHz') if len(speeds) == 1
        else Fact.unknown('Configured module speed is unavailable', source=source),
        module_count=module_count,
        channel_layout=Fact.measured(layout, source=source) if layout else Fact.unknown(reason, source=source),
        memory_type=memory_type)


def _gpu_facts(wmi_dynamic: dict[str, dict], monitoring: dict | None, observed: datetime | None,
               nvidia_probe=None, uuid_probe=None) -> tuple[GpuFacts, ...]:
    observation = (nvidia_probe or discover_nvidia)()
    uuids = (uuid_probe or discover_gpu_uuids)() if observation.gpus else {}
    reason = 'nvidia-smi did not report this field'
    profiles = []
    for gpu in observation.gpus or ():
        slot = gpu.slot
        extra = wmi_dynamic.get(slot) or {}
        free_mib = extra.get('memory_free_mib')
        sm = extra.get('sm_clock_mhz')
        mem_clock = extra.get('memory_clock_mhz')
        max_sm = extra.get('max_sm_clock_mhz')
        pcie_gen = extra.get('pcie_generation')
        pcie_width = extra.get('pcie_link_width')
        uuid_value = gpu.uuid or uuids.get(slot)
        profiles.append(GpuFacts(
            slot=slot, vendor=gpu.vendor, name=gpu.name,
            uuid=Fact.measured(uuid_value, source=NVIDIA) if uuid_value
            else Fact.unknown('GPU UUID is unavailable', source=NVIDIA),
            driver_version=Fact.measured(gpu.driver_version, source=NVIDIA) if gpu.driver_version
            else Fact.unknown(reason, source=NVIDIA),
            compute_capability=Fact.measured(gpu.compute_capability, source=NVIDIA) if gpu.compute_capability
            else Fact.unknown(reason, source=NVIDIA),
            cuda_driver_api=Fact.measured(observation.cuda_driver_api, source=NVIDIA) if observation.cuda_driver_api
            else Fact.unknown('CUDA driver API version is unavailable', source=NVIDIA),
            vram_total_bytes=Fact.measured(gpu.vram_bytes, source=NVIDIA, unit='B') if gpu.vram_bytes
            else Fact.unknown(reason, source=NVIDIA),
            vram_used_bytes=_sensor(monitoring, observed, 'gpu', 'vram_used_mb', scale=1024 * 1024, unit='B', source=NVIDIA),
            vram_free_bytes=Fact.measured(int(free_mib) * 1024 * 1024, source=NVIDIA, unit='B') if free_mib is not None
            else Fact.unknown(reason, source=NVIDIA),
            utilization_percent=_sensor(monitoring, observed, 'gpu', 'utilization_percent', unit='%', source=NVIDIA),
            temperature_c=_sensor(monitoring, observed, 'gpu', 'temperature_c', unit='C', source=NVIDIA),
            power_draw_w=_sensor(monitoring, observed, 'gpu', 'power_w', unit='W', source=NVIDIA),
            power_limit_w=_sensor(monitoring, observed, 'gpu', 'power_limit_w', unit='W', source=NVIDIA),
            sm_clock_mhz=Fact.measured(sm, source=NVIDIA, unit='MHz') if sm is not None else Fact.unknown(reason, source=NVIDIA),
            memory_clock_mhz=Fact.measured(mem_clock, source=NVIDIA, unit='MHz') if mem_clock is not None else Fact.unknown(reason, source=NVIDIA),
            max_sm_clock_mhz=Fact.measured(max_sm, source=NVIDIA, unit='MHz') if max_sm is not None else Fact.unknown(reason, source=NVIDIA),
            pcie_generation=Fact.measured(pcie_gen, source=NVIDIA) if pcie_gen is not None else Fact.unknown(reason, source=NVIDIA),
            pcie_link_width=Fact.measured(pcie_width, source=NVIDIA) if pcie_width is not None else Fact.unknown(reason, source=NVIDIA)))
    return tuple(profiles)


def _disks(wmi: dict | None) -> tuple[DiskFacts, ...]:
    source = WINDOWS
    reason = 'Windows CIM disk inventory is unavailable on this host'
    result = []
    for disk in _items(wmi.get('disks')) if isinstance(wmi, dict) else []:
        index = _key(disk.get('index'))
        if index is None:
            continue
        size = _number(disk.get('size'))
        model = _label(disk.get('model'))
        interface = _label(disk.get('interface'))
        media = _label(disk.get('media'))
        result.append(DiskFacts(
            index=index,
            model=Fact.measured(model, source=source) if model else Fact.unknown(reason, source=source),
            interface_type=Fact.measured(interface, source=source) if interface else Fact.unknown(reason, source=source),
            media_type=Fact.measured(media, source=source) if media else Fact.unknown(reason, source=source),
            size_bytes=Fact.measured(int(size), source=source, unit='B') if size else Fact.unknown(reason, source=source)))
    return tuple(result)


def _volumes(wmi: dict | None) -> tuple[VolumeFacts, ...]:
    source = WINDOWS
    reason = 'Windows CIM volume inventory is unavailable on this host'
    result = []
    for volume in _items(wmi.get('volumes')) if isinstance(wmi, dict) else []:
        mount = _label(volume.get('mount'))
        if mount is None or not mount.startswith(('C:', 'D:', 'E:', 'F:', 'G:')):
            continue
        size, free = _number(volume.get('size')), _number(volume.get('free'))
        filesystem = _label(volume.get('fs'))
        result.append(VolumeFacts(
            mount_point=mount,
            filesystem=Fact.measured(filesystem, source=source) if filesystem else Fact.unknown(reason, source=source),
            size_bytes=Fact.measured(int(size), source=source, unit='B') if size else Fact.unknown(reason, source=source),
            free_bytes=Fact.measured(int(free), source=source, unit='B') if free is not None
            else Fact.unknown(reason, source=source)))
    return tuple(result)


def _model_stores(paths: Sequence[Path]) -> tuple[ModelStoreFacts, ...]:
    """Bounded, read-only size accounting for explicitly configured stores only."""
    result = []
    for path in paths:
        name, exists, files, total = path.name, False, 0, 0
        if path.is_dir():
            exists = True
            for root, _dirs, names in os.walk(path):
                for entry in names:
                    files += 1
                    if files > MAX_MODEL_STORE_FILES:
                        break
                    try:
                        total += (Path(root) / entry).stat().st_size
                    except OSError:
                        continue
                if files > MAX_MODEL_STORE_FILES:
                    break
        if exists:
            result.append(ModelStoreFacts(name=name, exists=Fact.measured(True, source=STORE),
                file_count=Fact.measured(files, source=STORE),
                size_bytes=Fact.measured(total, source=STORE, unit='B')))
        else:
            result.append(ModelStoreFacts(name=name,
                exists=Fact.unknown(f'Model store {name} does not exist', source=STORE),
                file_count=Fact.unknown('Model store is absent', source=STORE),
                size_bytes=Fact.unknown('Model store is absent', source=STORE)))
    return tuple(result)


def _runtime_models(items: Sequence[dict], loaded: bool) -> tuple[RuntimeModelFacts, ...]:
    result = []
    for item in items:
        reference = _label(item.get('id') or item.get('name'))
        if reference is None:
            continue
        details = item.get('details') if isinstance(item.get('details'), dict) else {}
        size = _number(item.get('size_vram')) if loaded else _number(item.get('size'))
        quantization = _label(details.get('quantization_level'))
        digest_value = item.get('digest')
        result.append(RuntimeModelFacts(
            runtime_reference=reference,
            size_bytes=Fact.measured(int(size), source=OLLAMA, unit='B') if size
            else Fact.unknown('Model size is not reported by the runtime', source=OLLAMA),
            digest=Fact.measured(digest_value, source=OLLAMA)
            if isinstance(digest_value, str) and digest_value
            else Fact.unknown('Model digest is not reported by the runtime', source=OLLAMA),
            quantization=Fact.measured(quantization, source=OLLAMA) if quantization
            else Fact.unknown('Quantization is not reported by the runtime', source=OLLAMA)))
    return tuple(sorted(result, key=lambda model: model.runtime_reference))


def _runtime(endpoint: str, qualified_endpoint: str | None, adapter_factory=None) -> RuntimeFacts:
    """Version, installed inventory and resident models over read-only adapter calls."""
    profile = RuntimeProfile(runtime='ollama', base_url=endpoint)
    adapter = (adapter_factory or OllamaAdapter)(profile)
    version, installed, resident = None, (), ()
    reachable = False
    try:
        version = adapter.version()
        installed = adapter.inventory()
        reachable = True
    except (AdapterUnavailable, InferenceTimeout, InvalidResponse, InferenceError,
            PolicyDenied, RateQuotaUnavailable, KeyError, TypeError, ValueError, AttributeError):
        reachable = False
    if reachable:
        try:
            resident = adapter.loaded()
        except (AdapterUnavailable, InferenceTimeout, InvalidResponse, InferenceError,
                PolicyDenied, RateQuotaUnavailable, KeyError, TypeError, ValueError, AttributeError):
            resident = ()
    allocated = sum(int(_number(item.get('size_vram')) or 0) for item in resident)
    loaded_models = _runtime_models(resident, True)
    return RuntimeFacts(
        endpoint=endpoint, runtime='ollama',
        version=Fact.measured(version, source=OLLAMA) if reachable and _label(version)
        else Fact.unknown('Runtime version is unavailable', source=OLLAMA),
        reachable=Fact.measured(reachable, source=OLLAMA),
        qualified_node0=Fact.derived(endpoint == qualified_endpoint, source=DERIVED),
        installed_models=_runtime_models(installed, False) if reachable else (),
        loaded_models=loaded_models,
        vram_allocated_bytes=Fact.measured(allocated, source=OLLAMA, unit='B') if reachable
        else Fact.unknown('Runtime is unreachable', source=OLLAMA))


def collect_snapshot(*, wmi_runner: Callable[[str], dict | None] | None = None,
                     monitoring_fetch: Callable[[], tuple[dict, str]] | None = None,
                     runtime_endpoints: Sequence[str] = (),
                     qualified_endpoint: str | None = None,
                     model_stores: Sequence[Path] = (),
                     nvidia_runner: Callable[[tuple[str, ...]], Any] | None = None,
                     nvidia_probe=None, uuid_probe=None, adapter_factory=None) -> HardwareSnapshot:
    """Read-only inventory. Any unavailable input becomes an explicit UNKNOWN fact."""
    wmi = (wmi_runner or _run_windows)(_WMI_SCRIPT)
    monitoring, monitoring_source, observed = _monitoring(monitoring_fetch)
    dynamic = _nvidia_dynamic(nvidia_runner)
    return HardwareSnapshot(
        probe_version=PROBE_VERSION,
        system=_system_facts(wmi), cpu=_cpu_facts(wmi, monitoring, observed),
        memory=_memory_facts(wmi, monitoring, observed),
        gpus=_gpu_facts(dynamic, monitoring, observed, nvidia_probe, uuid_probe),
        disks=_disks(wmi), volumes=_volumes(wmi), model_stores=_model_stores(model_stores),
        runtimes=tuple(_runtime(endpoint, qualified_endpoint, adapter_factory)
                       for endpoint in runtime_endpoints),
        coverage=SensorCoverage(
            monitoring_reachable=bool(monitoring), monitoring_source=monitoring_source,
            monitoring_observed_at=observed))


def capability_profile(snapshot: HardwareSnapshot) -> HardwareCapabilityProfile:
    """Deterministic dimensions only. No fit, ranking or recommendation is produced."""
    facts: list[ModelCapacityFact] = []

    def add(dimension: str, fact: Fact, basis: tuple[str, ...] = ()) -> None:
        facts.append(ModelCapacityFact(dimension=dimension, kind=fact.kind, value=fact.value,
            unit=fact.unit, basis=basis or (dimension,), reason=fact.reason))

    add('logical_threads', snapshot.cpu.logical_threads)
    add('physical_cores', snapshot.cpu.physical_cores)
    add('base_clock_mhz', snapshot.cpu.base_clock_mhz)
    add('system_memory_bytes', snapshot.memory.installed_bytes)
    add('memory_speed_mhz', snapshot.memory.configured_speed_mhz)
    for gpu in snapshot.gpus:
        add(f'gpu[{gpu.slot}].vram_total_bytes', gpu.vram_total_bytes)
        add(f'gpu[{gpu.slot}].compute_capability', gpu.compute_capability)
        add(f'gpu[{gpu.slot}].sm_clock_mhz', gpu.max_sm_clock_mhz)
        add(f'gpu[{gpu.slot}].power_limit_w', gpu.power_limit_w)
    for volume in snapshot.volumes:
        add(f'volume[{volume.mount_point}].free_bytes', volume.free_bytes)
    for store in snapshot.model_stores:
        add(f'model_store[{store.name}].size_bytes', store.size_bytes)
    if not snapshot.gpus:
        facts.append(ModelCapacityFact(dimension='gpu.vram_total_bytes', kind='unknown',
            value=None, basis=('gpu.vram_total_bytes',),
            reason='No GPU inventory was observed'))
    return HardwareCapabilityProfile(
        source_snapshot_digest=digest(snapshot.digest_data()), facts=tuple(facts))
