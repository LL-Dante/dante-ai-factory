"""Bounded NVIDIA inventory through nvidia-smi; absence and ambiguity stay unknown."""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from dante.contracts.qualification import EvidenceModel, GPUProfile, Label


_MAX_OUTPUT = 256 * 1024
_LABEL = re.compile(r'^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,159}$')


@dataclass(frozen=True)
class CommandOutput:
    returncode: int
    stdout: str
    stderr: str = ''


CommandRunner = Callable[[tuple[str, ...]], CommandOutput]


class NvidiaObservation(EvidenceModel):
    tool_available: bool
    gpus: tuple[GPUProfile, ...] | None = None
    cuda_driver_api: Label | None = None


def _run(executable: str, arguments: tuple[str, ...]) -> CommandOutput:
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    completed = subprocess.run([executable, *arguments], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=5, check=False, shell=False, creationflags=flags)
    return CommandOutput(completed.returncode, completed.stdout, completed.stderr)


def _bounded(output: CommandOutput) -> bool:
    return len(output.stdout.encode('utf-8')) + len(output.stderr.encode('utf-8')) <= _MAX_OUTPUT


def _no_devices(output: CommandOutput) -> bool:
    return 'no devices were found' in (output.stdout + '\n' + output.stderr).lower()


def _label(value: str) -> str | None:
    value = value.strip()
    return value if _LABEL.fullmatch(value) else None


def _base_inventory(output: CommandOutput) -> tuple[GPUProfile, ...] | None:
    if not _bounded(output):
        return None
    if _no_devices(output):
        return ()
    if output.returncode != 0:
        return None
    rows = list(csv.reader(io.StringIO(output.stdout), skipinitialspace=True))
    if not rows or any(len(row) != 4 for row in rows):
        return None
    profiles = []
    try:
        for index, name, memory_mib, driver in rows:
            index, name, driver = _label(index), _label(name), _label(driver)
            memory_mib = memory_mib.strip()
            if (index is None or not index.isascii() or not index.isdecimal()
                    or name is None or driver is None
                    or not memory_mib.isascii() or not memory_mib.isdecimal()):
                return None
            profiles.append(GPUProfile(slot=index, vendor='NVIDIA', name=name,
                vram_bytes=int(memory_mib) * 1024 * 1024, driver_version=driver))
        if len({profile.slot for profile in profiles}) != len(profiles):
            return None
        return tuple(profiles)
    except (ValueError, OverflowError):
        return None


def _compute_capabilities(output: CommandOutput) -> dict[str, str]:
    if output.returncode != 0 or not _bounded(output):
        return {}
    rows = list(csv.reader(io.StringIO(output.stdout), skipinitialspace=True))
    capabilities: dict[str, str] = {}
    for row in rows:
        if len(row) != 2:
            return {}
        slot, capability = _label(row[0]), _label(row[1])
        if (slot is None or not slot.isascii() or not slot.isdecimal()
                or capability is None or slot in capabilities):
            return {}
        capabilities[slot] = capability
    return capabilities


def _cuda_driver_api(output: CommandOutput) -> str | None:
    if output.returncode != 0 or not _bounded(output):
        return None
    match = re.search(r'\bCUDA Version:\s*([0-9]+\.[0-9]+)\b', output.stdout)
    return 'driver-supported-' + match.group(1) if match else None


def discover_nvidia(runner: CommandRunner | None = None) -> NvidiaObservation:
    """Return observed facts only. CUDA value is driver API support, not toolkit/runtime proof."""
    if runner is None:
        executable = shutil.which('nvidia-smi')
        if executable is None:
            return NvidiaObservation(tool_available=False)
        runner = lambda arguments: _run(executable, arguments)
    try:
        base = runner(('--query-gpu=index,name,memory.total,driver_version',
                       '--format=csv,noheader,nounits'))
        gpus = _base_inventory(base)
        if gpus is None:
            return NvidiaObservation(tool_available=True)
        if not gpus:
            return NvidiaObservation(tool_available=True, gpus=())
        compute = _compute_capabilities(runner(('--query-gpu=index,compute_cap',
                                                '--format=csv,noheader,nounits')))
        enriched = tuple(profile.model_copy(update={'compute_capability': compute.get(profile.slot)})
                         for profile in gpus)
        cuda = _cuda_driver_api(runner(tuple()))
        return NvidiaObservation(tool_available=True, gpus=enriched, cuda_driver_api=cuda)
    except subprocess.TimeoutExpired:
        return NvidiaObservation(tool_available=True)
    except (OSError, subprocess.SubprocessError, csv.Error, ValueError, TypeError, AttributeError):
        return NvidiaObservation(tool_available=False)
