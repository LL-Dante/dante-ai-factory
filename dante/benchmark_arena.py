"""Stage B2 measurement arena.

The arena turns one prompt into one recorded sample, one model into one run, and a
run into a comparison that refuses to invent an overall winner. Two rules are
enforced in code rather than trusted to convention: the benchmark runtime is never
the production runtime, and production residency is re-read after every batch.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dante.contracts.benchmarks import (
    BenchmarkComparison, BenchmarkIsolation, BenchmarkParameters, BenchmarkPrompt,
    BenchmarkResult, BenchmarkRun, BenchmarkSample, DimensionComparison,
    ModelBenchmarkProfile, RuntimeResidency, VramBudget, text_digest,
)
from dante.benchmark_runtime import BenchmarkRuntimeError
from dante.contracts.hardware import Fact
from dante.nvidia_probe import CommandOutput, CommandRunner

PROBE = 'benchmark-arena-v1'
GENERATE = 'ollama-generate'
LOAD = 'ollama-load_duration'
PS = 'ollama-ps'
NVIDIA = 'nvidia-smi-sampled'
RAM = 'win32-process-working-set'
LOCAL = 'local-monotonic-clock'
DERIVED = 'derived-arena'
UNKNOWN_SENSORS = 'GPU sensors did not return usable readings'
UNKNOWN_RAM = 'Per-process RAM attribution was unavailable'

# A deliberately small GPU allowance. The desktop, the driver and the qualified
# production model must all keep working while a benchmark model is resident.
VRAM_RESERVE_BYTES = 2 * 1024 ** 3
VRAM_BUDGET_BYTES = 6 * 1024 ** 3
VRAM_KV_RESERVE_BYTES = 1 * 1024 ** 3
# Measured on this host: 4B artifact 2,497,293,931 B reported 3,178,149,969 B of VRAM.
VRAM_OVERHEAD_RATIO = 3_178_149_969 / 2_497_293_931
NO_ANSWER = 'No comparable measurement is available'

CommandRunnerFactory = Callable[[], CommandRunner]


def _default_runner() -> CommandRunner:
    executable = shutil.which('nvidia-smi')
    if executable is None:
        raise FileNotFoundError('nvidia-smi is unavailable')
    def run(arguments: tuple[str, ...]) -> CommandOutput:
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        completed = subprocess.run([executable, *arguments], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=5, check=False, shell=False, creationflags=flags)
        return CommandOutput(completed.returncode, completed.stdout, completed.stderr)
    return run


def _powershell(script: str) -> CommandOutput:
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    completed = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=30, check=False, shell=False, creationflags=flags)
    return CommandOutput(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    parent_pid: int
    name: str
    working_set_bytes: int


def read_processes() -> tuple[ProcessRow, ...]:
    """One bounded process query. Failure yields nothing, never a zero."""
    script = ("Get-CimInstance Win32_Process | "
              "Select-Object ProcessId,ParentProcessId,Name,WorkingSetSize | "
              "ConvertTo-Json -Compress")
    try:
        output = _powershell(script)
    except (OSError, subprocess.SubprocessError):
        return ()
    if output.returncode != 0 or not output.stdout.strip():
        return ()
    try:
        payload = json.loads(output.stdout)
    except ValueError:
        return ()
    rows = payload if isinstance(payload, list) else [payload]
    parsed: list[ProcessRow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            pid = int(row.get('ProcessId'))
            parent = int(row.get('ParentProcessId'))
            name = str(row.get('Name') or '')
            working = int(row.get('WorkingSetSize'))
        except (TypeError, ValueError):
            continue
        if pid > 0 and name:
            parsed.append(ProcessRow(pid, parent, name, working))
    return tuple(parsed)


def listening_pid(port: int, runner: Callable[..., CommandOutput] = _powershell) -> int | None:
    script = (f"(Get-NetTCPConnection -State Listen -LocalPort {int(port)} -ErrorAction SilentlyContinue | "
              "Select-Object -First 1 -ExpandProperty OwningProcess)")
    try:
        output = runner(script)
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode != 0:
        return None
    text = output.stdout.strip()
    return int(text) if text.isdecimal() else None


class RamProbe:
    """Host RAM attributable to one runtime's inference servers.

    Reads are injected so a test never shells out, and an unavailable reading is
    reported as unavailable rather than as zero bytes.
    """

    def __init__(self, port: int, *, process_reader=read_processes, owner_reader=listening_pid):
        self.port = port
        self._processes = process_reader
        self._owner = owner_reader

    def working_set(self) -> int | None:
        try:
            owner = self._owner(self.port)
        except Exception:
            return None
        if owner is None:
            return None
        try:
            processes = self._processes()
        except Exception:
            return None
        servers = [row for row in processes
                   if row.parent_pid == owner and row.name.lower().startswith('llama-server')]
        return sum(row.working_set_bytes for row in servers) if servers else None


def server_working_set(port: int, processes: Sequence[ProcessRow]) -> int | None:
    """Sum the working set of the inference servers owned by one runtime's port."""
    owner = listening_pid(port)
    if owner is None:
        return None
    children = [row for row in processes if row.parent_pid == owner]
    servers = [row for row in children if row.name.lower().startswith('llama-server')]
    if not servers:
        return None
    return sum(row.working_set_bytes for row in servers)


def _gpu_row(output: CommandOutput) -> dict | None:
    if output.returncode != 0 or not output.stdout.strip():
        return None
    rows = list(csv.reader(io.StringIO(output.stdout), skipinitialspace=True))
    if not rows or any(len(row) != 5 for row in rows):
        return None
    util, used, free, temperature, power = [], [], [], [], []
    for row in rows:
        try:
            util.append(int(row[0].strip()))
            used.append(int(row[1].strip()))
            free.append(int(row[2].strip()))
            temperature.append(int(row[3].strip()))
            power.append(float(row[4].strip()))
        except (TypeError, ValueError):
            return None
    return {'util': max(util), 'used': sum(used), 'free': sum(free),
            'temperature': max(temperature), 'power': max(power)}


def read_gpu(runner: CommandRunner | None = None) -> dict | None:
    query = ('--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits')
    try:
        output = (runner or _default_runner())(query)
    except (OSError, subprocess.SubprocessError, FileNotFoundError):
        return None
    return _gpu_row(output)


class GpuSampler:
    """Poll the GPU while a generation runs, so peaks are observed not inferred."""

    def __init__(self, runner: CommandRunner | None = None, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._runner = runner
        self._stop = threading.Event()
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        runner = self._runner
        while not self._stop.is_set():
            try:
                row = read_gpu(runner)
            except Exception:
                row = None
            if row is not None:
                with self._lock:
                    self._rows.append(row)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> 'GpuSampler':
        self._thread = threading.Thread(target=self._poll, name='gpu-sampler', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exception) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False

    def facts(self) -> dict[str, Fact]:
        with self._lock:
            rows = list(self._rows)
        if not rows:
            unknown = Fact.unknown(UNKNOWN_SENSORS, source=NVIDIA)
            return {name: unknown for name in ('gpu_utilization_peak_percent',
                                               'gpu_utilization_mean_percent',
                                               'gpu_temperature_peak_c', 'gpu_power_peak_w',
                                               'gpu_free_vram_min_bytes')}
        utils = [row['util'] for row in rows]
        return {
            'gpu_utilization_peak_percent': Fact.measured(max(utils), source=NVIDIA, unit='percent'),
            'gpu_utilization_mean_percent': Fact.derived(round(statistics.fmean(utils), 2),
                                                        source=NVIDIA, unit='percent'),
            'gpu_temperature_peak_c': Fact.measured(max(row['temperature'] for row in rows),
                                                    source=NVIDIA, unit='celsius'),
            'gpu_power_peak_w': Fact.derived(round(max(row['power'] for row in rows), 2),
                                             source=NVIDIA, unit='watt'),
            'gpu_free_vram_min_bytes': Fact.measured(min(row['free'] for row in rows) * 1024 * 1024,
                                                     source=NVIDIA, unit='bytes'),
        }


def plan_budget(*, free_vram_bytes: int, artifact_bytes: int | None, block_count: int | None,
                budget_bytes: int = VRAM_BUDGET_BYTES, reserve_bytes: int = VRAM_RESERVE_BYTES,
                kv_reserve_bytes: int = VRAM_KV_RESERVE_BYTES) -> VramBudget:
    """Cap GPU residency so a benchmark model can never crowd out production."""
    if free_vram_bytes <= reserve_bytes + 1024 ** 3:
        raise ValueError(f'Only {free_vram_bytes} B of free VRAM; refusing to place a benchmark model')
    # The reserve protects production, so it is kept and the budget yields to it.
    budget_bytes = min(budget_bytes, free_vram_bytes - reserve_bytes)
    if budget_bytes <= kv_reserve_bytes:
        raise ValueError(f'A {budget_bytes} B budget leaves no room for weights beside a '
                         f'{kv_reserve_bytes} B KV reserve')
    if artifact_bytes is None or block_count is None or block_count <= 0:
        return VramBudget(free_vram_bytes=free_vram_bytes, budget_bytes=budget_bytes,
            reserve_bytes=min(reserve_bytes, max(free_vram_bytes - 1, 0)),
            kv_reserve_bytes=min(kv_reserve_bytes, budget_bytes - 1), num_gpu_layers=0,
            block_count=block_count, full_gpu_possible=False,
            basis='Artifact size or block count is unknown, so no layers are placed on the GPU')
    estimate = int(artifact_bytes * VRAM_OVERHEAD_RATIO)
    usable = budget_bytes - kv_reserve_bytes
    fits = estimate <= usable
    layers = block_count if fits else max(int(block_count * usable / estimate), 1)
    basis = (f'estimate={estimate} B at overhead ratio {VRAM_OVERHEAD_RATIO:.4f}; '
             f'budget={budget_bytes} B minus {kv_reserve_bytes} B KV reserve leaves {usable} B for weights; '
             f'free VRAM={free_vram_bytes} B with {reserve_bytes} B reserve; '
             + ('full residency fits inside the budget' if fits else
                f'partial offload: {layers} of {block_count} layers'))
    return VramBudget(free_vram_bytes=free_vram_bytes, budget_bytes=budget_bytes,
        reserve_bytes=min(reserve_bytes, max(free_vram_bytes - 1, 0)), kv_reserve_bytes=kv_reserve_bytes,
        full_gpu_estimate_bytes=estimate, overhead_ratio=round(VRAM_OVERHEAD_RATIO, 4),
        num_gpu_layers=layers, block_count=block_count, full_gpu_possible=fits, basis=basis)


def _rate(count: int | None, duration_ns: int | None) -> Fact:
    if count is None or duration_ns is None or duration_ns <= 0:
        return Fact.unknown('The runtime did not report token counts or durations', source=GENERATE)
    return Fact.derived(round(count / (duration_ns / 1e9), 3), source=DERIVED, unit='tokens_per_second')


def _seconds(nanoseconds: int | None, reason: str) -> Fact:
    if nanoseconds is None:
        return Fact.unknown(reason, source=GENERATE)
    return Fact.measured(round(nanoseconds / 1e9, 6), source=GENERATE, unit='seconds')


def _median(facts: Sequence[Fact]) -> float | None:
    values = [fact.value for fact in facts if fact.kind != 'unknown' and fact.value is not None]
    return round(statistics.median(values), 4) if values else None


@dataclass
class ModelTarget:
    """One model the caller already proved is installed locally."""

    model_id: str
    digest_sha256: str
    artifact_bytes: int | None
    block_count: int | None
    quantization: str | None
    parameter_size: str | None
    parameter_count: int | None = None
    architecture: str | None = None
    native_context_tokens: int | None = None


class BenchmarkArena:
    """Runs bounded local benchmarks and returns evidence, never conclusions."""

    ISOLATED_BENCHMARK = True

    def __init__(self, *, production_endpoint: str, benchmark_endpoint: str,
                 hardware_digest: str | None = None, gpu_runner: CommandRunner | None = None,
                 sampler_interval_s: float = 0.25, process_reader=read_processes,
                 owner_reader=listening_pid):
        if production_endpoint.rstrip('/') == benchmark_endpoint.rstrip('/'):
            raise ValueError('Benchmark runtime must differ from the production runtime')
        self.production_endpoint = production_endpoint.rstrip('/')
        self.benchmark_endpoint = benchmark_endpoint.rstrip('/')
        self.hardware_digest = hardware_digest
        self._gpu_runner = gpu_runner
        self._sampler_interval = sampler_interval_s
        self._port = int(self.benchmark_endpoint.rsplit(':', 1)[-1])
        self._ram = RamProbe(self._port, process_reader=process_reader, owner_reader=owner_reader)
        self._version: str | None = None
        self._production_last: RuntimeResidency | None = None
        self._loaded: list[str] = []
        self._disturbances: list[str] = []
        self._violations: list[str] = []
        self._batches = 0

    # -- observation helpers -------------------------------------------------
    def _residency(self, runtime) -> RuntimeResidency:
        gpu = read_gpu(self._gpu_runner)
        return RuntimeResidency(
            endpoint=runtime.endpoint,
            models=tuple(sorted((entry['name'], entry['size_vram'], entry['context_length'])
                                for entry in runtime.residency())),
            gpu_used_bytes=(gpu['used'] * 1024 * 1024) if gpu else None,
            gpu_free_bytes=(gpu['free'] * 1024 * 1024) if gpu else None)

    def _ram_facts(self, baseline: int | None) -> dict[str, Fact]:
        current = self._ram.working_set()
        if current is None:
            return {'ram_working_set_bytes': Fact.unknown(UNKNOWN_RAM, source=RAM),
                    'ram_delta_bytes': Fact.unknown(UNKNOWN_RAM, source=RAM)}
        facts = {'ram_working_set_bytes': Fact.measured(current, source=RAM, unit='bytes')}
        facts['ram_delta_bytes'] = (Fact.derived(current - baseline, source=DERIVED, unit='bytes')
                                    if baseline is not None else
                                    Fact.unknown('No pre-load RAM baseline was captured', source=RAM))
        return facts

    def _check_production(self, production, expected: RuntimeResidency, label: str) -> None:
        self._batches += 1
        current = self._residency(production)
        if not current.same_models_as(expected):
            self._disturbances.append(
                f'{label}: production residency changed from {list(expected.names)} to {list(current.names)}')
        self._production_last = current

    # -- measurement ---------------------------------------------------------
    def _measure(self, runtime, target: ModelTarget, prompt: BenchmarkPrompt, *,
                 parameters: BenchmarkParameters, repetition: int, phase: str,
                 production, production_expected: RuntimeResidency, label: str,
                 baseline_ram: int | None) -> BenchmarkSample:
        started = time.time()
        with GpuSampler(self._gpu_runner, self._sampler_interval) as sampler:
            raw = runtime.generate(target.model_id, prompt=prompt.text, parameters=parameters)
        completed = time.time()
        measurements: dict[str, Fact] = {
            'time_to_first_token_s': (Fact.measured(round(raw.time_to_first_token_s, 6),
                                                    source=GENERATE, unit='seconds')
                                       if raw.time_to_first_token_s is not None else
                                       Fact.unknown('The stream produced no first token', source=GENERATE)),
            'wall_clock_s': Fact.measured(round(raw.wall_clock_s, 6), source=LOCAL, unit='seconds'),
            'total_latency_s': _seconds(raw.total_duration_ns, 'The runtime reported no total duration'),
            'prompt_tokens': (Fact.measured(raw.prompt_eval_count, source=GENERATE, unit='tokens')
                              if raw.prompt_eval_count is not None else
                              Fact.unknown('The runtime reported no prompt token count', source=GENERATE)),
            'output_tokens': (Fact.measured(raw.eval_count, source=GENERATE, unit='tokens')
                              if raw.eval_count is not None else
                              Fact.unknown('The runtime reported no output token count', source=GENERATE)),
            'prompt_tokens_per_second': _rate(raw.prompt_eval_count, raw.prompt_eval_duration_ns),
            'generation_tokens_per_second': _rate(raw.eval_count, raw.eval_duration_ns),
        }
        measurements.update(sampler.facts())
        if phase == 'cold':
            measurements['cold_load_s'] = _seconds(raw.load_duration_ns,
                                                   'The runtime reported no load duration')
        else:
            measurements['warm_latency_s'] = _seconds(raw.total_duration_ns,
                                                      'The runtime reported no total duration')
        resident = runtime.residency()
        entry = next((item for item in resident if item['name'] == target.model_id), None)
        if entry is not None:
            measurements['vram_resident_bytes'] = Fact.measured(entry['size_vram'], source=PS, unit='bytes')
        else:
            measurements['vram_resident_bytes'] = Fact.unknown(
                'The model was not resident after the request', source=PS)
        measurements['vram_delta_bytes'] = (Fact.derived(entry['size_vram'], source=DERIVED, unit='bytes')
                                            if entry is not None else
                                            Fact.unknown('Resident VRAM is unknown', source=PS))
        measurements.update(self._ram_facts(baseline_ram))
        parsed: bool | None = None
        if parameters.structured_output:
            try:
                json.loads(raw.response)
                parsed = True
            except ValueError:
                parsed = False
        self._check_production(production, production_expected, label)
        return BenchmarkSample(
            model_id=target.model_id, digest_sha256=target.digest_sha256,
            quantization=target.quantization, parameter_size_label=target.parameter_size,
            runtime_endpoint=self.benchmark_endpoint, runtime_version=self._version_of(runtime),
            hardware_snapshot_digest=self.hardware_digest, prompt_id=prompt.prompt_id,
            prompt_digest=prompt.digest, prompt_kind=prompt.kind, parameters=parameters,
            repetition=repetition, phase=phase,
            gpu_execution=self._gpu_execution(measurements),
            started_at=_stamp(started), completed_at=_stamp(completed),
            measurements=measurements, done_reason=raw.done_reason,
            output_excerpt=raw.response[:600], output_sha256=text_digest(raw.response),
            output_parsed=parsed, disposition='completed', sensor_source=NVIDIA)

    def _version_of(self, runtime) -> str | None:
        if self._version is None:
            try:
                self._version = runtime.version()
            except BenchmarkRuntimeError:
                self._version = None
        return self._version

    @staticmethod
    def _gpu_execution(measurements: dict[str, Fact]) -> str:
        fact = measurements.get('gpu_utilization_peak_percent')
        if fact is None or fact.kind == 'unknown':
            return 'unknown'
        if isinstance(fact.value, (int, float)) and fact.value > 0:
            return 'active'
        return 'inactive'

    # -- orchestration -------------------------------------------------------
    def run(self, runtime, production, targets: Sequence[ModelTarget], prompts: Sequence[BenchmarkPrompt],
            parameters: BenchmarkParameters, *, repetitions: int = 2) -> BenchmarkResult:
        ordered = sorted(targets, key=lambda item: item.model_id)
        production_before = self._residency(production)
        benchmark_before = self._residency(runtime)
        base_ram = self._ram.working_set()
        gpu = read_gpu(self._gpu_runner)
        free_vram = (gpu['free'] * 1024 * 1024) if gpu else VRAM_BUDGET_BYTES + VRAM_RESERVE_BYTES
        runs: list[BenchmarkRun] = []
        try:
            for target in ordered:
                runs.append(self._run_model(runtime, production, target, prompts, parameters,
                                            repetitions=repetitions, production_before=production_before,
                                            base_ram=base_ram, free_vram=free_vram))
        finally:
            self._cleanup(runtime)
        self._check_production(production, production_before, 'final')
        benchmark_after = self._residency(runtime)
        isolation = BenchmarkIsolation(
            production_endpoint=self.production_endpoint, benchmark_endpoint=self.benchmark_endpoint,
            production_before=production_before,
            production_after=getattr(self, '_production_last', production_before),
            production_unchanged=not self._disturbances,
            production_checked_batches=self._batches,
            production_disturbances=tuple(self._disturbances),
            benchmark_before=benchmark_before, benchmark_after=benchmark_after,
            benchmark_restored=benchmark_after.same_models_as(benchmark_before),
            cleanup_attempted=True, models_loaded=tuple(dict.fromkeys(self._loaded)),
            models_unloaded=tuple(dict.fromkeys(self._loaded)), leaked_residency=tuple(benchmark_after.names),
            policy=(f'benchmark runtime {self.benchmark_endpoint} is separate from production '
                    f'{self.production_endpoint}; GPU allowance capped at {VRAM_BUDGET_BYTES} B with '
                    f'{VRAM_RESERVE_BYTES} B reserve; production residency re-read after every batch'),
            violations=tuple(self._violations))
        return BenchmarkResult(
            isolation=isolation, hardware_snapshot_digest=self.hardware_digest, runs=tuple(runs),
            comparison=self._compare(runs, parameters), smoke_only=True)

    def _run_model(self, runtime, production, target: ModelTarget, prompts: Sequence[BenchmarkPrompt],
                   parameters: BenchmarkParameters, *, repetitions: int,
                   production_before: RuntimeResidency, base_ram: int | None,
                   free_vram: int) -> BenchmarkRun:
        started = time.time()
        budget = plan_budget(free_vram_bytes=free_vram, artifact_bytes=target.artifact_bytes,
                             block_count=target.block_count)
        samples: list[BenchmarkSample] = []
        failures: list[str] = []
        disposition: str | None = None
        reason: str | None = None
        try:
            for repetition in range(1, max(repetitions, 1) + 1):
                phase = 'cold' if repetition == 1 else 'warm'
                batch = sorted(prompts, key=lambda item: item.prompt_id)
                for prompt in batch:
                    sample_parameters = parameters.model_copy(update={
                        'num_gpu_layers': budget.num_gpu_layers,
                        'structured_output': prompt.kind == 'structured_json'})
                    try:
                        sample = self._measure(runtime, target, prompt, parameters=sample_parameters,
                            repetition=repetition, phase=phase, production=production,
                            production_expected=production_before, label=f'{target.model_id}/{prompt.prompt_id}',
                            baseline_ram=base_ram)
                    except Exception as error:
                        reason = f'{type(error).__name__}: {error}'
                        disposition = ('resource_exhausted' if 'residency' in reason or 'status 500' in reason
                                       else 'failed')
                        failures.append(f'{target.model_id}/{prompt.prompt_id}/{phase}: {reason}')
                        samples.append(self._failed_sample(target, prompt, sample_parameters, repetition,
                                                          phase, reason, disposition))
                        break
                    samples.append(sample)
                    self._loaded.append(target.model_id)
                    vram = sample.measurements.get('vram_resident_bytes')
                    if vram is not None and vram.kind != 'unknown' and vram.value > budget.budget_bytes:
                        reason = (f'Resident VRAM {vram.value} B exceeded the {budget.budget_bytes} B budget')
                        disposition = 'resource_exhausted'
                        failures.append(f'{target.model_id}: {reason}')
                        break
                if disposition is not None:
                    break
        finally:
            if target.model_id in self._loaded or samples:
                try:
                    runtime.unload(target.model_id)
                except BenchmarkRuntimeError as error:
                    self._violations.append(f'{target.model_id}: unload failed: {error}')
        tested = any(sample.disposition == 'completed' for sample in samples)
        load_failed = disposition is not None and not tested
        completed = [s for s in samples if s.disposition == 'completed']
        # Speed medians use plain-text samples only, so JSON-mode measurement never
        # mixes into a throughput number.
        plain = [s for s in completed if not s.parameters.structured_output] or completed
        timed_out = sum(1 for s in samples if s.disposition == 'timeout')
        profile = ModelBenchmarkProfile(
            model_id=target.model_id, digest_sha256=target.digest_sha256,
            quantization=target.quantization, parameter_size_label=target.parameter_size,
            parameter_count=target.parameter_count, artifact_bytes=target.artifact_bytes,
            architecture=target.architecture, native_context_tokens=target.native_context_tokens,
            runtime_endpoint=self.benchmark_endpoint, tested=tested or not load_failed,
            skip_reason=(reason if load_failed else None),
            load_succeeded=(None if load_failed else tested),
            load_failure=(failures[0] if load_failed else None),
            vram_budget=budget, observed_vram_resident_bytes=_max_value(completed, 'vram_resident_bytes'),
            observed_ram_working_set_bytes=_max_value(completed, 'ram_working_set_bytes'),
            gpu_execution=(completed[0].gpu_execution if completed else 'unknown'),
            sample_count=len(samples), completed_samples=len(completed),
            failed_samples=len(samples) - len(completed) - timed_out, timed_out_samples=timed_out,
            malformed_outputs=sum(1 for s in samples if s.output_parsed is False),
            median_generation_tokens_per_second=_median([s.measurements.get('generation_tokens_per_second', _no()) for s in plain]),
            median_prompt_tokens_per_second=_median([s.measurements.get('prompt_tokens_per_second', _no()) for s in plain]),
            median_cold_load_s=_median([s.measurements.get('cold_load_s', _no()) for s in plain]),
            median_warm_latency_s=_median([s.measurements.get('warm_latency_s', _no()) for s in plain]),
            median_time_to_first_token_s=_median([s.measurements.get('time_to_first_token_s', _no()) for s in plain]))
        run = BenchmarkRun(profile=profile, started_at=_stamp(started), completed_at=_stamp(time.time()),
                           samples=tuple(sorted(samples, key=lambda s: (s.repetition, s.prompt_id, s.phase))),
                           batch_count=repetitions, production_unchanged_through_run=not self._disturbances,
                           abort_reason=reason if disposition == 'resource_exhausted' else None)
        self._check_production(production, production_before, f'{target.model_id}/model-end')
        return run

    def _failed_sample(self, target: ModelTarget, prompt: BenchmarkPrompt, parameters: BenchmarkParameters,
                       repetition: int, phase: str, reason: str, disposition: str) -> BenchmarkSample:
        now = _stamp(time.time())
        measurements: dict[str, Fact] = {}
        if phase == 'cold':
            # A cold sample that never loaded has no load duration. Say so.
            measurements['cold_load_s'] = Fact.unknown(reason[:200], source=GENERATE)
        return BenchmarkSample(
            model_id=target.model_id, digest_sha256=target.digest_sha256,
            quantization=target.quantization, runtime_endpoint=self.benchmark_endpoint,
            hardware_snapshot_digest=self.hardware_digest, prompt_id=prompt.prompt_id,
            prompt_digest=prompt.digest, prompt_kind=prompt.kind, parameters=parameters,
            repetition=repetition, phase=phase, started_at=now, completed_at=now,
            measurements=measurements, disposition=disposition, failure=reason[:500],
            sensor_source=NVIDIA)

    def _cleanup(self, runtime) -> None:
        for model_id in list(dict.fromkeys(self._loaded)):
            try:
                runtime.unload(model_id)
            except BenchmarkRuntimeError as error:
                self._violations.append(f'cleanup: {model_id}: {error}')

    def _compare(self, runs: Sequence[BenchmarkRun], parameters: BenchmarkParameters) -> BenchmarkComparison:
        tested = [run for run in runs if run.profile.tested]
        models = tuple(run.profile.model_id for run in tested)
        keys = {sample.parameters.sampling_key() for run in tested for sample in run.samples
                if sample.disposition == 'completed'}
        shared = len(keys) <= 1
        differences: list[str] = []
        if not shared:
            differences.append('Sampling parameters differed between models')
        plans = {run.profile.model_id: run.profile.vram_budget for run in tested}
        for model_id, budget in plans.items():
            if budget is not None and not budget.full_gpu_possible:
                differences.append(f'{model_id} ran partially offloaded ({budget.num_gpu_layers} GPU layers), '
                                   'so its speed is not directly comparable to a full-GPU model')
        offloaded = {model_id for model_id, budget in plans.items()
                     if budget is not None and not budget.full_gpu_possible}
        speed_comparable = shared and not offloaded
        dimensions: list[DimensionComparison] = []
        for name, attribute, unit in (
                ('cold_load_s', 'median_cold_load_s', 'seconds'),
                ('generation_tokens_per_second', 'median_generation_tokens_per_second', 'tokens_per_second'),
                ('prompt_tokens_per_second', 'median_prompt_tokens_per_second', 'tokens_per_second'),
                ('time_to_first_token_s', 'median_time_to_first_token_s', 'seconds'),
                ('vram_resident_bytes', 'observed_vram_resident_bytes', 'bytes'),
                ('ram_working_set_bytes', 'observed_ram_working_set_bytes', 'bytes')):
            values = {}
            for run in tested:
                value = getattr(run.profile, attribute)
                values[run.profile.model_id] = (Fact.measured(value, source=DERIVED, unit=unit)
                                                 if value is not None else
                                                 Fact.unknown(NO_ANSWER, source=DERIVED))
            is_speed = 'tokens_per_second' in name
            comparable = not (is_speed and not speed_comparable)
            blocking = tuple(sorted(offloaded)) if (is_speed and not speed_comparable) else ()
            best = None
            if comparable and all(fact.kind != 'unknown' for fact in values.values()) and len(values) > 1:
                best = max(values, key=lambda key: values[key].value if name.endswith(
                    'tokens_per_second') else -values[key].value)
            if comparable:
                basis = 'median over completed samples'
            elif blocking:
                basis = 'not comparable: partial offload on ' + ', '.join(blocking)
            else:
                basis = 'not comparable: shared sampling parameters could not be confirmed'
            dimensions.append(DimensionComparison(
                dimension=name, unit=unit, comparable=comparable, values=values, best_model_id=best,
                basis=basis, blocking_differences=blocking))
        parsed = {}
        for run in tested:
            checks = [sample.output_parsed for sample in run.samples
                      if sample.parameters.structured_output and sample.output_parsed is not None]
            parsed[run.profile.model_id] = (Fact.measured(
                f'{sum(1 for value in checks if value)}/{len(checks)} structured outputs parsed',
                source=DERIVED, unit='ratio') if checks else Fact.unknown(NO_ANSWER, source=DERIVED))
        dimensions.append(DimensionComparison(
            dimension='structured_output_validity', comparable=True, values=parsed,
            basis='one fixed JSON-mode prompt per model', blocking_differences=()))
        return BenchmarkComparison(
            models=models, dimensions=tuple(sorted(dimensions, key=lambda item: item.dimension)),
            shared_parameters_identical=shared, recorded_differences=tuple(differences),
            policy=('Dimensions are reported separately. No universal score, no overall winner, '
                    'and no role assignment is derived from a smoke benchmark.'))


def _stamp(seconds: float):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _no() -> Fact:
    return Fact.unknown(NO_ANSWER, source=DERIVED)


def _max_value(samples: Sequence[BenchmarkSample], metric: str) -> int | None:
    values = [sample.measurements[metric].value for sample in samples
              if metric in sample.measurements and sample.measurements[metric].kind != 'unknown'
              and isinstance(sample.measurements[metric].value, (int, float))]
    return int(max(values)) if values else None


__all__ = ['BenchmarkArena', 'GpuSampler', 'ModelTarget', 'RamProbe',
           'ProcessRow', 'listening_pid', 'plan_budget', 'read_gpu', 'read_processes',
           'server_working_set']
