"""Conservative host inventory and injected runtime discovery. No subprocesses or scans."""
from collections.abc import Callable, Mapping
import os
import platform
import re
from typing import Literal, Protocol
from uuid import UUID

from dante.contracts.qualification import EvidenceModel, MachineProfile, RuntimeObservation
from dante.nvidia_probe import NvidiaObservation, discover_nvidia


def _label(value: str) -> str | None:
    value = value.strip()
    return value if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,159}', value) else None


def _physical_memory() -> int | None:
    if os.name != 'nt':
        return None  # No platform-dependent estimate silently substituted.
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('physical', 'available', 'pagefile', 'available_pagefile', 'virtual', 'available_virtual', 'extended')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    return status.physical if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None


def probe_machine(node_id: UUID, *,
                  nvidia_probe: Callable[[], NvidiaObservation] | None = None) -> MachineProfile:
    """No hostname, username, serial numbers, paths, environment dumps or GPU guesses."""
    nvidia = (nvidia_probe or discover_nvidia)()
    return MachineProfile(node_id=node_id, os=_label(platform.system()) or 'unknown',
        os_version=_label(platform.version()) or 'unknown', architecture=_label(platform.machine()) or 'unknown',
        cpu=_label(platform.processor()), logical_cpus=os.cpu_count(), ram_bytes=_physical_memory(),
        gpus=nvidia.gpus, cuda_runtime=nvidia.cuda_driver_api)


class RuntimeInventory(Protocol):
    runtime: str

    def inventory(self) -> list[dict]: ...


class DiscoveryResult(EvidenceModel):
    state: Literal['discovered', 'unsupported', 'unavailable', 'invalid']
    observation: RuntimeObservation | None = None


class RuntimeDiscovery:
    """Explicit probe registration, not PATH scanning or auto-starting installed services."""
    def __init__(self, probes: Mapping[str, Callable[[], RuntimeObservation]] | None = None):
        self.probes = dict(probes or {})

    def discover(self, runtime: str) -> DiscoveryResult:
        probe = self.probes.get(runtime)
        if probe is None:
            return DiscoveryResult(state='unsupported')
        try:
            observation = RuntimeObservation.model_validate_json(probe().model_dump_json())
            if observation.runtime != runtime:
                return DiscoveryResult(state='invalid')
            return DiscoveryResult(state='discovered', observation=observation)
        except Exception:
            return DiscoveryResult(state='unavailable')


def adapter_inventory(adapter: RuntimeInventory, observation: RuntimeObservation,
                      aliases: Mapping[str, str]) -> RuntimeObservation:
    """Ollama/llama.cpp seam using existing bounded transports, only when explicitly called.

    Caller provides sanitized aliases. Raw provider IDs, filenames, paths and other
    inventory metadata are discarded. Reachability never establishes qualification.
    Versions/backends must be supplied by a future measured runtime-specific probe.
    """
    if adapter.runtime not in {'ollama', 'llamacpp'} or observation.runtime != adapter.runtime:
        raise ValueError('Unsupported inventory adapter')
    inventory = adapter.inventory()
    names = tuple(sorted({aliases[item['id']] for item in inventory if item['id'] in aliases}))
    return RuntimeObservation.model_validate({**observation.model_dump(), 'model_inventory': names})
