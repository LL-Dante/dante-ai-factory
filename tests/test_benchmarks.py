"""Stage B2 benchmark arena tests.

The recurring theme is refusal: a benchmark may not target production, may not
publish a universal score, may not invent a metric, and may not leave a model
resident. Every test here asserts an absence as firmly as a measurement.
"""
from __future__ import annotations

import json
import threading
import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from dante.agent_runner import (BENCHMARK_ENDPOINT, DanteBenchmarkAgent, build_agent_registry,
                                benchmark_parameters, benchmark_prompts, definition_digest)
from dante.benchmark_arena import (VRAM_BUDGET_BYTES, VRAM_OVERHEAD_RATIO, VRAM_RESERVE_BYTES,
                                   BenchmarkArena, ModelTarget, ProcessRow, RamProbe, plan_budget,
                                   read_gpu)
from dante.benchmark_runtime import BenchmarkRuntime, BenchmarkRuntimeError, RawGeneration
from dante.contracts.agents import (BENCHMARK_CAPABILITIES, READ_ONLY_CAPABILITIES, AgentDefinition,
                                    AgentModelTarget, AgentTaskPayload)
from dante.contracts.benchmarks import (BenchmarkComparison, BenchmarkIsolation, BenchmarkParameters,
                                        BenchmarkPrompt, BenchmarkResult, BenchmarkSample,
                                        DimensionComparison, ModelBenchmarkProfile, RuntimeResidency,
                                        VramBudget, text_digest)
from dante.contracts.hardware import Fact
from dante.control_center import benchmark_overview, render_benchmark_text
from dante.model_scout import hardware_snapshot_digest
from dante.nvidia_probe import CommandOutput
from dante.workload import ModelRequirement, QualificationRejected, WorkloadSpec

PRODUCTION = 'http://127.0.0.1:11435'
BENCH = 'http://127.0.0.1:11434'
SMALL = 'qwen3:4b'
LARGE = 'hf.co/KikoCis/Qwen3.8-27B-GGUF:Q3_K_M'
DERIVED = 'dante-qwen-agent:latest'
SMALL_DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'
LARGE_DIGEST = '4ae69a30b61946aa270ab86eaf5ffbcac7e7977129c62f99503e42c4be946a46'
HARDWARE = 'a' * 64
PROD_RESIDENT = (('qwen3:4b', 3178149969, 4096),)
GPU_CSV = '42, 4537, 11441, 43, 148.82\n'


def gpu_runner(*_arguments, **_keywords):
    return CommandOutput(0, GPU_CSV, '')


def gpu_runner_busy(*_arguments, **_keywords):
    return CommandOutput(0, '94, 7862, 8116, 43, 148.82\n', '')


def gpu_runner_idle(*_arguments, **_keywords):
    return CommandOutput(0, '0, 4537, 11441, 32, 56.90\n', '')


def no_processes():
    return ()


def owner_none(_port):
    return None


def ram_processes():
    return (ProcessRow(100, 1, 'ollama.exe', 40 * 1024 * 1024),
            ProcessRow(200, 100, 'llama-server.exe', 800 * 1024 * 1024))


def owner_present(port):
    return 100 if port == 11434 else None


def tag(name, digest, size, *, family='qwen3', quant='Q4_K_M', params='4.0B', parent=''):
    return {'name': name, 'digest': digest, 'size': size, 'modified_at': '2026-09-01T00:00:00Z',
            'details': {'family': family, 'parameter_size': params, 'quantization_level': quant,
                        'format': 'gguf', 'context_length': 262144, 'parent_model': parent}}


def show(*, family='qwen3', blocks=36):
    return {'details': {'family': family, 'quantization_level': 'Q4_K_M'},
            'model_info': {f'{family}.block_count': blocks, f'{family}.context_length': 262144},
            'capabilities': ['completion', 'thinking', 'tools']}


class FakeAdapter:
    """Stands in for the read-only Ollama adapter."""

    def __init__(self, profile, *, tags=(), resident=(), shown=None, version='0.34.4'):
        self.profile = profile
        self._tags = list(tags)
        self._resident = list(resident)
        self._shown = dict(shown or {})
        self._version = version

    def version(self):
        return self._version

    def inventory(self):
        return self._tags

    def loaded(self):
        return self._resident

    def show(self, reference):
        payload = self._shown.get(reference)
        if payload is None:
            raise ValueError('not found')
        return payload


class FakeRuntime:
    """Stands in for BenchmarkRuntime. Records every call and every unload."""

    def __init__(self, endpoint, *, tags=(), resident=(), shown=None, vram_by_model=None,
                 generate_error=None, unload_error=None, vram_total=3178149969):
        self.endpoint = endpoint
        self.tags = [dict(item) for item in tags]
        self.resident_models = {item['name']: dict(item) for item in resident}
        self.shown = dict(shown or {})
        self.vram_by_model = dict(vram_by_model or {})
        self.generate_error = generate_error
        self.unload_error = unload_error
        self.vram_total = vram_total
        self.generated: list[tuple[str, str, BenchmarkParameters]] = []
        self.unloaded: list[str] = []

    def version(self):
        return '0.34.4'

    def installed(self):
        # Same flat, normalized shape the real adapter produces.
        normalized = []
        for item in self.tags:
            details = item.get('details') or {}
            normalized.append({'name': item['name'], 'digest': item['digest'], 'size': item['size'],
                'family': details.get('family'), 'quantization': details.get('quantization_level'),
                'parameter_size': details.get('parameter_size'),
                'parent_model': details.get('parent_model'),
                'context_length': details.get('context_length'), 'capabilities': ()})
        return tuple(sorted(normalized, key=lambda item: item['name']))

    def block_count(self, model_id):
        for key, value in (self.shown.get(model_id) or {}).get('model_info', {}).items():
            if key.endswith('.block_count'):
                return value
        return None

    def show(self, model_id):
        return dict(self.shown.get(model_id) or {})

    def residency(self):
        return tuple(self.resident_models[name] for name in sorted(self.resident_models))

    def generate(self, model_id, *, prompt, parameters):
        self.generated.append((model_id, prompt, parameters))
        if model_id not in self.resident_models:
            self.resident_models[model_id] = {
                'name': model_id, 'digest': SMALL_DIGEST, 'size_vram': self.vram_by_model.get(
                    model_id, self.vram_total), 'context_length': 4096, 'quantization': 'Q4_K_M',
                'parameter_size': '4.0B'}
        if self.generate_error is not None:
            raise self.generate_error
        return RawGeneration(response='{"fact": "ok"}' if parameters.structured_output else 'answer text',
            thinking=None, load_duration_ns=1_642_000_000 if parameters is not None else 0,
            prompt_eval_count=27, prompt_eval_duration_ns=69_540_000, eval_count=48,
            eval_duration_ns=217_656_000, total_duration_ns=1_760_000_000,
            done_reason='length', time_to_first_token_s=0.0393, wall_clock_s=1.81, chunk_count=9)

    def unload(self, model_id):
        if self.unload_error is not None:
            raise self.unload_error
        self.resident_models.pop(model_id, None)
        self.unloaded.append(model_id)


def target(model_id=SMALL, digest=SMALL_DIGEST, *, size=2497293931, blocks=36, quant='Q4_K_M',
           params='4.0B', family='qwen3'):
    return ModelTarget(model_id=model_id, digest_sha256=digest, artifact_bytes=size,
                       block_count=blocks, quantization=quant, parameter_size=params,
                       architecture=family, native_context_tokens=262144)


def production_runtime(**keywords):
    return FakeRuntime(PRODUCTION, resident=[{'name': SMALL, 'digest': SMALL_DIGEST,
        'size_vram': 3178149969, 'context_length': 4096, 'quantization': 'Q4_K_M',
        'parameter_size': '4.0B'}], **keywords)


def benchmark_runtime(**keywords):
    return FakeRuntime(BENCH, **keywords)


def arena(**keywords):
    options = {'production_endpoint': PRODUCTION, 'benchmark_endpoint': BENCH,
               'hardware_digest': HARDWARE, 'gpu_runner': gpu_runner,
               'process_reader': no_processes, 'owner_reader': owner_none,
               'sampler_interval_s': 0.01}
    options.update(keywords)
    return BenchmarkArena(**options)


def run_arena(arena_object, runtime, production, targets, **keywords):
    return arena_object.run(runtime, production, targets, benchmark_prompts(),
                            benchmark_parameters(), **keywords)


class BenchmarkContractTests(unittest.TestCase):
    def test_benchmark_runtime_may_not_be_production(self):
        with self.assertRaises(ValidationError):
            BenchmarkIsolation(production_endpoint=PRODUCTION, benchmark_endpoint=PRODUCTION,
                production_before=RuntimeResidency(endpoint=PRODUCTION),
                benchmark_before=RuntimeResidency(endpoint=PRODUCTION), policy='x')

    def test_result_rejects_production_target(self):
        with self.assertRaises(ValidationError):
            BenchmarkResult(isolation=BenchmarkIsolation(
                production_endpoint=PRODUCTION, benchmark_endpoint=PRODUCTION,
                production_before=RuntimeResidency(endpoint=PRODUCTION),
                benchmark_before=RuntimeResidency(endpoint=PRODUCTION), policy='x'),
                comparison=BenchmarkComparison(models=(SMALL,), shared_parameters_identical=True,
                                               policy='x'))

    def test_no_universal_score_or_winner(self):
        for field in ('universal_score', 'overall_winner'):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    BenchmarkComparison(models=(SMALL,), shared_parameters_identical=True,
                                        policy='x', **{field: 1.0 if field == 'universal_score' else SMALL})

    def test_role_assignments_are_impossible(self):
        with self.assertRaises(ValidationError):
            BenchmarkComparison(models=(SMALL,), shared_parameters_identical=True, policy='x',
                                role_assignments=('writer',))

    def test_blocked_dimension_may_not_name_a_best_model(self):
        with self.assertRaises(ValidationError):
            DimensionComparison(dimension='generation_tokens_per_second', comparable=False,
                values={SMALL: Fact.measured(1.0, source='s')}, best_model_id=SMALL, basis='x')

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(ValidationError):
            self._sample(measurements={'vibes': Fact.measured(1.0, source='s')})

    def test_prompt_digest_must_match_text(self):
        with self.assertRaises(ValidationError):
            BenchmarkPrompt(prompt_id='P1', kind='latency', digest='0' * 64, characters=4,
                            text='real text')

    def test_cold_sample_requires_load_duration(self):
        with self.assertRaises(ValidationError):
            self._sample(phase='cold')

    def test_incomplete_sample_must_say_why(self):
        with self.assertRaises(ValidationError):
            self._sample(disposition='failed')

    def test_unknown_fact_carries_no_value(self):
        with self.assertRaises(ValidationError):
            Fact(kind='unknown', value=1, source='s', reason='r')

    def test_budget_cannot_exceed_free_vram(self):
        with self.assertRaises(ValidationError):
            VramBudget(free_vram_bytes=1000, budget_bytes=900, reserve_bytes=900,
                       kv_reserve_bytes=10, num_gpu_layers=1, full_gpu_possible=False, basis='x')

    def test_budget_reserve_must_leave_room_for_weights(self):
        with self.assertRaises(ValidationError):
            VramBudget(free_vram_bytes=10_000_000_000, budget_bytes=1024, reserve_bytes=0,
                       kv_reserve_bytes=1024, num_gpu_layers=1, full_gpu_possible=False, basis='x')

    def _sample(self, **keywords):
        now = datetime.now(timezone.utc)
        options = {'model_id': SMALL, 'digest_sha256': SMALL_DIGEST, 'runtime_endpoint': BENCH,
                   'hardware_snapshot_digest': HARDWARE, 'prompt_id': 'LAT_1',
                   'prompt_digest': 'b' * 64, 'prompt_kind': 'latency',
                   'parameters': benchmark_parameters(), 'repetition': 2, 'phase': 'warm',
                   'started_at': now, 'completed_at': now, 'measurements': {}}
        options.update(keywords)
        return BenchmarkSample(**options)


class VramBudgetTests(unittest.TestCase):
    FREE = 11995709440

    def test_small_model_fits_fully_on_gpu(self):
        budget = plan_budget(free_vram_bytes=self.FREE, artifact_bytes=2497293931, block_count=36)
        self.assertTrue(budget.full_gpu_possible)
        self.assertEqual(budget.num_gpu_layers, 36)

    def test_large_model_is_partially_offloaded(self):
        budget = plan_budget(free_vram_bytes=self.FREE, artifact_bytes=13500737704, block_count=65)
        self.assertFalse(budget.full_gpu_possible)
        self.assertLess(budget.num_gpu_layers, 65)
        self.assertGreater(budget.num_gpu_layers, 0)

    def test_budget_never_exceeds_free_vram_minus_reserve(self):
        budget = plan_budget(free_vram_bytes=self.FREE, artifact_bytes=13500737704, block_count=65)
        self.assertLessEqual(budget.budget_bytes + budget.reserve_bytes, self.FREE)
        self.assertEqual(budget.budget_bytes, VRAM_BUDGET_BYTES)
        self.assertEqual(budget.reserve_bytes, VRAM_RESERVE_BYTES)

    def test_estimate_matches_the_observed_four_b_ratio(self):
        budget = plan_budget(free_vram_bytes=self.FREE, artifact_bytes=2497293931, block_count=36)
        self.assertEqual(budget.full_gpu_estimate_bytes, 3178149969)
        self.assertAlmostEqual(VRAM_OVERHEAD_RATIO, 1.2726, places=3)

    def test_unknown_artifact_places_nothing_on_gpu(self):
        budget = plan_budget(free_vram_bytes=self.FREE, artifact_bytes=None, block_count=None)
        self.assertEqual(budget.num_gpu_layers, 0)
        self.assertFalse(budget.full_gpu_possible)

    def test_tiny_free_vram_is_refused(self):
        with self.assertRaises(ValueError):
            plan_budget(free_vram_bytes=1024, artifact_bytes=2497293931, block_count=36)


class SensorTests(unittest.TestCase):
    def test_gpu_row_is_parsed(self):
        row = read_gpu(gpu_runner)
        self.assertEqual(row['util'], 42)
        self.assertEqual(row['free'], 11441)
        self.assertAlmostEqual(row['power'], 148.82, places=2)

    def test_malformed_gpu_output_grants_nothing(self):
        self.assertIsNone(read_gpu(lambda *_a, **_k: CommandOutput(0, 'not,csv\n', '')))

    def test_failed_gpu_query_grants_nothing(self):
        self.assertIsNone(read_gpu(lambda *_a, **_k: CommandOutput(1, '', 'boom')))

    def test_ram_probe_attributes_by_parent_process(self):
        probe = RamProbe(11434, process_reader=ram_processes, owner_reader=owner_present)
        self.assertEqual(probe.working_set(), 800 * 1024 * 1024)

    def test_ram_probe_reports_absence_rather_than_zero(self):
        self.assertIsNone(RamProbe(11434, process_reader=no_processes,
                                   owner_reader=owner_present).working_set())
        self.assertIsNone(RamProbe(11434, process_reader=ram_processes,
                                   owner_reader=owner_none).working_set())

    def test_ram_probe_survives_a_failing_reader(self):
        def explode():
            raise OSError('denied')
        self.assertIsNone(RamProbe(11434, process_reader=explode,
                                   owner_reader=owner_present).working_set())


class ArenaIsolationTests(unittest.TestCase):
    def test_arena_refuses_production_as_benchmark_runtime(self):
        with self.assertRaises(ValueError):
            BenchmarkArena(production_endpoint=PRODUCTION, benchmark_endpoint=PRODUCTION,
                           hardware_digest=HARDWARE)

    def test_production_is_re_read_after_every_batch(self):
        production = production_runtime()
        result = run_arena(arena(), benchmark_runtime(), production, [target()], repetitions=2)
        self.assertTrue(result.isolation.production_unchanged)
        self.assertGreaterEqual(result.isolation.production_checked_batches, 4)
        self.assertTrue(result.isolation.verified())

    def test_a_disturbed_production_is_reported(self):
        # Production is the runtime that drifts, once the benchmark has run a batch.
        state = {'batches': 0}

        class Drifting(FakeRuntime):
            def residency(self):
                if state['batches'] > 1:
                    return ({'name': SMALL, 'digest': SMALL_DIGEST, 'size_vram': 3178149969,
                             'context_length': 4096, 'quantization': 'Q4_K_M',
                             'parameter_size': '4.0B'},
                            {'name': 'other:1b', 'digest': 'c' * 64, 'size_vram': 1,
                             'context_length': 512, 'quantization': 'Q4_K_M',
                             'parameter_size': '1.0B'})
                return super().residency()

        class Counting(FakeRuntime):
            def generate(self, model_id, *, prompt, parameters):
                state['batches'] += 1
                return super().generate(model_id, prompt=prompt, parameters=parameters)

        runtime = Counting(BENCH)
        result = run_arena(arena(), runtime, Drifting(PRODUCTION, resident=[
            {'name': SMALL, 'digest': SMALL_DIGEST, 'size_vram': 3178149969, 'context_length': 4096,
             'quantization': 'Q4_K_M', 'parameter_size': '4.0B'}]), [target()], repetitions=2)
        self.assertFalse(result.isolation.production_unchanged)
        self.assertTrue(result.isolation.production_disturbances)
        self.assertFalse(result.isolation.verified())

    def test_benchmark_residency_is_restored(self):
        runtime = benchmark_runtime()
        result = run_arena(arena(), runtime, production_runtime(), [target()], repetitions=2)
        self.assertTrue(result.isolation.benchmark_restored)
        self.assertEqual(result.isolation.leaked_residency, ())
        self.assertEqual(runtime.residency(), ())
        self.assertIn(SMALL, runtime.unloaded)

    def test_pre_existing_benchmark_residency_is_left_alone(self):
        runtime = benchmark_runtime(resident=[{'name': 'other:1b', 'digest': 'c' * 64,
            'size_vram': 1, 'context_length': 512, 'quantization': 'Q4_K_M',
            'parameter_size': '1.0B'}])
        result = run_arena(arena(), runtime, production_runtime(), [target()], repetitions=2)
        self.assertTrue(result.isolation.benchmark_restored)
        self.assertEqual(runtime.residency()[0]['name'], 'other:1b')

    def test_a_failing_sample_still_unloads(self):
        runtime = benchmark_runtime(generate_error=BenchmarkRuntimeError('runtime returned 500'))
        result = run_arena(arena(), runtime, production_runtime(), [target()], repetitions=2)
        self.assertEqual(runtime.residency(), ())
        self.assertIn(SMALL, runtime.unloaded)
        self.assertFalse(result.runs[0].profile.tested)

    def test_a_failing_unload_is_recorded_as_a_violation(self):
        runtime = benchmark_runtime(unload_error=BenchmarkRuntimeError('unload refused'))
        result = run_arena(arena(), runtime, production_runtime(), [target()], repetitions=2)
        self.assertTrue(result.isolation.violations)
        self.assertFalse(result.isolation.verified())


class ArenaMeasurementTests(unittest.TestCase):
    def test_cold_and_warm_phases_are_recorded_separately(self):
        result = run_arena(arena(gpu_runner=gpu_runner_busy), benchmark_runtime(),
                           production_runtime(), [target()], repetitions=2)
        samples = result.runs[0].samples
        self.assertEqual(samples[0].phase, 'cold')
        self.assertIn('cold_load_s', samples[0].measurements)
        self.assertNotIn('warm_latency_s', samples[0].measurements)
        warm = [item for item in samples if item.phase == 'warm']
        self.assertTrue(warm)
        for sample in warm:
            self.assertIn('warm_latency_s', sample.measurements)
            self.assertNotIn('cold_load_s', sample.measurements)

    def test_every_sample_keeps_its_identity_and_provenance(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(), [target()],
                           repetitions=1)
        sample = result.runs[0].samples[0]
        self.assertEqual(sample.model_id, SMALL)
        self.assertEqual(sample.digest_sha256, SMALL_DIGEST)
        self.assertEqual(sample.runtime_endpoint, BENCH)
        self.assertEqual(sample.hardware_snapshot_digest, HARDWARE)
        self.assertEqual(sample.prompt_digest, text_digest(
            next(item.text for item in benchmark_prompts() if item.prompt_id == sample.prompt_id)))
        self.assertEqual(sample.sensor_source, 'nvidia-smi-sampled')
        for name, fact in sample.measurements.items():
            self.assertIsNotNone(fact.source, name)
            if fact.kind == 'unknown':
                self.assertIsNone(fact.value)
                self.assertTrue(fact.reason)

    def test_gpu_execution_is_observed(self):
        busy = run_arena(arena(gpu_runner=gpu_runner_busy), benchmark_runtime(),
                         production_runtime(), [target()], repetitions=1)
        self.assertEqual(busy.runs[0].profile.gpu_execution, 'active')
        idle = run_arena(arena(gpu_runner=gpu_runner_idle), benchmark_runtime(),
                         production_runtime(), [target()], repetitions=1)
        self.assertEqual(idle.runs[0].profile.gpu_execution, 'inactive')

    def test_missing_sensors_become_unknown_not_zero(self):
        def broken(*_arguments, **_keywords):
            raise OSError('nvidia-smi missing')
        result = run_arena(arena(gpu_runner=broken), benchmark_runtime(), production_runtime(),
                           [target()], repetitions=1)
        measurements = result.runs[0].samples[0].measurements
        self.assertEqual(measurements['gpu_utilization_peak_percent'].kind, 'unknown')
        self.assertIsNone(measurements['gpu_power_peak_w'].value)
        self.assertEqual(result.runs[0].profile.gpu_execution, 'unknown')

    def test_vram_above_budget_aborts_the_model(self):
        result = run_arena(arena(), benchmark_runtime(vram_total=VRAM_BUDGET_BYTES + 1),
                           production_runtime(), [target()], repetitions=2)
        profile = result.runs[0].profile
        self.assertEqual(len(result.runs[0].samples), 1)
        self.assertIn('exceeded', result.runs[0].abort_reason or '')
        self.assertIsNotNone(profile.observed_vram_resident_bytes)
        self.assertEqual(result.isolation.benchmark_restored, True)

    def test_structured_output_is_checked(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(), [target()],
                           repetitions=2)
        parsed = {sample.prompt_id: sample.output_parsed
                  for sample in result.runs[0].samples}
        self.assertIs(parsed['JSON_1'], True)
        self.assertIsNone(parsed['LAT_1'])

    def test_malformed_structured_output_is_counted(self):
        class Malformed(FakeRuntime):
            def generate(self, model_id, *, prompt, parameters):
                sample = super().generate(model_id, prompt=prompt, parameters=parameters)
                return RawGeneration(**{**sample.__dict__, 'response': 'not json at all'})

        result = run_arena(arena(), Malformed(BENCH), production_runtime(), [target()],
                           repetitions=2)
        # JSON_1 runs once per repetition, so both of its samples are malformed.
        self.assertEqual(result.runs[0].profile.malformed_outputs, 2)

    def test_ram_is_recorded_when_attributable(self):
        result = run_arena(arena(process_reader=ram_processes, owner_reader=owner_present),
                           benchmark_runtime(), production_runtime(), [target()], repetitions=1)
        measurements = result.runs[0].samples[0].measurements
        self.assertEqual(measurements['ram_working_set_bytes'].kind, 'measured')
        self.assertEqual(measurements['ram_working_set_bytes'].value, 800 * 1024 * 1024)

    def test_samples_are_deterministic_in_order(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(), [target()],
                           repetitions=2)
        keys = [(item.repetition, item.prompt_id, item.phase) for item in result.runs[0].samples]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), 6)

    def test_runs_are_ordered_by_model_id(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(),
                           [target(LARGE, LARGE_DIGEST, size=13500737704, blocks=65, quant='Q3_K_M',
                                   params='26.9B', family='qwen35'), target()], repetitions=1)
        self.assertEqual([item.profile.model_id for item in result.runs], sorted([SMALL, LARGE]))


class ComparisonTests(unittest.TestCase):
    def test_partial_offload_blocks_a_speed_winner(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(),
                           [target(LARGE, LARGE_DIGEST, size=13500737704, blocks=65, quant='Q3_K_M',
                                   params='26.9B', family='qwen35'), target()], repetitions=2)
        speed = result.comparison.by_dimension('generation_tokens_per_second')
        self.assertFalse(speed.comparable)
        self.assertIsNone(speed.best_model_id)
        self.assertTrue(speed.blocking_differences)
        self.assertTrue(any('offloaded' in item for item in result.comparison.recorded_differences))

    def test_shared_parameters_are_reported_identical(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(), [target()],
                           repetitions=2)
        self.assertTrue(result.comparison.shared_parameters_identical)
        keys = {item.parameters.sampling_key() for item in result.runs[0].samples}
        self.assertEqual(len(keys), 1)

    def test_json_mode_does_not_make_speed_incomparable(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(),
                           [target(LARGE, LARGE_DIGEST, size=13500737704, blocks=65, quant='Q3_K_M',
                                   params='26.9B', family='qwen35'), target()], repetitions=2)
        self.assertTrue(result.comparison.shared_parameters_identical)
        speed = result.comparison.by_dimension('generation_tokens_per_second')
        # Still blocked, but only by the offload difference, never by JSON mode.
        self.assertEqual(speed.blocking_differences, (LARGE,))

    def test_comparison_publishes_no_score_and_no_winner(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(), [target()],
                           repetitions=2)
        self.assertIsNone(result.comparison.universal_score)
        self.assertIsNone(result.comparison.overall_winner)
        self.assertEqual(result.comparison.role_assignments, ())
        self.assertEqual(result.comparison.verdict, 'dimension_only')

    def test_dimensions_are_separate_and_sorted(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(), [target()],
                           repetitions=2)
        names = [item.dimension for item in result.comparison.dimensions]
        self.assertEqual(names, sorted(names))
        self.assertIn('vram_resident_bytes', names)
        self.assertIn('generation_tokens_per_second', names)
        self.assertIn('structured_output_validity', names)

    def test_no_download_delete_or_role_fields_exist(self):
        result = run_arena(arena(), benchmark_runtime(), production_runtime(), [target()],
                           repetitions=2)
        self.assertEqual(result.models_downloaded, 0)
        self.assertEqual(result.models_deleted, 0)
        self.assertEqual(result.role_assignments_made, 0)
        self.assertFalse(result.qualification_changed)
        self.assertFalse(result.campaign_exhausted)


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_malformed_endpoint_is_refused(self):
        for endpoint in ('', '11434', 'ftp://x', 'http://', 'http://a b'):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(BenchmarkRuntimeError):
                    BenchmarkRuntime(endpoint)

    def test_model_reference_characters_are_checked(self):
        runtime = BenchmarkRuntime(BENCH)
        for reference in ('', '../../etc', 'a b', 'x' * 200):
            with self.subTest(reference=reference):
                with self.assertRaises(BenchmarkRuntimeError):
                    runtime.generate(reference, prompt='hi', parameters=benchmark_parameters())

    def test_modules_never_call_a_download_or_delete_endpoint(self):
        import dante.benchmark_arena as arena_module
        import dante.benchmark_runtime as runtime_module
        for module in (arena_module, runtime_module):
            with open(module.__file__, encoding='utf-8') as handle:
                source = handle.read()
            for forbidden in ('/api/pull', '/api/delete', '/api/create', '/api/copy',
                              'huggingface.co', 'ollama.com', 'api.openai', 'api.anthropic'):
                with self.subTest(module=module.__name__, token=forbidden):
                    self.assertNotIn(forbidden, source)

    def test_think_false_and_num_gpu_are_always_sent(self):
        runtime = BenchmarkRuntime(BENCH)
        body = runtime._request_body(SMALL, prompt='hi',
                                     parameters=benchmark_parameters().model_copy(
                                         update={'num_gpu_layers': 20}), keep_alive=None)
        self.assertIs(body['think'], False)
        self.assertEqual(body['options']['num_gpu'], 20)
        self.assertEqual(body['options']['seed'], 42)
        self.assertTrue(body['stream'])

    def test_json_mode_is_requested_only_when_asked(self):
        runtime = BenchmarkRuntime(BENCH)
        plain = runtime._request_body(SMALL, prompt='hi', parameters=benchmark_parameters(),
                                      keep_alive=None)
        structured = runtime._request_body(SMALL, prompt='hi', parameters=benchmark_parameters()
            .model_copy(update={'structured_output': True}), keep_alive=None)
        self.assertNotIn('format', plain)
        self.assertEqual(structured['format'], 'json')

    def test_adapter_metadata_is_normalized(self):
        adapter = FakeAdapter(None, tags=[tag(SMALL, SMALL_DIGEST, 2497293931)],
                              shown={SMALL: show()})
        runtime = BenchmarkRuntime(BENCH, adapter_factory=lambda _profile: adapter)
        installed = runtime.installed()
        self.assertEqual(installed[0]['name'], SMALL)
        self.assertEqual(installed[0]['size'], 2497293931)
        self.assertEqual(runtime.block_count(SMALL), 36)

    def test_entries_without_a_digest_are_dropped(self):
        adapter = FakeAdapter(None, tags=[{'name': 'x', 'size': 1}, tag(SMALL, SMALL_DIGEST, 2)])
        runtime = BenchmarkRuntime(BENCH, adapter_factory=lambda _profile: adapter)
        self.assertEqual([item['name'] for item in runtime.installed()], [SMALL])

    def test_unreachable_metadata_raises_a_bounded_error(self):
        class Broken(FakeAdapter):
            def inventory(self):
                raise OSError('refused')
        runtime = BenchmarkRuntime(BENCH, adapter_factory=lambda _profile: Broken(None))
        with self.assertRaises(BenchmarkRuntimeError) as caught:
            runtime.installed()
        self.assertNotIn('Traceback', str(caught.exception))


TARGET_MODEL = AgentModelTarget(model_id=SMALL, runtime_reference=SMALL,
                                digest_sha256=SMALL_DIGEST, context_tokens=4096)


class BenchmarkAgentTests(unittest.TestCase):
    TARGET = TARGET_MODEL

    def setUp(self):
        self.definition = DanteBenchmarkAgent.definition(self.TARGET)
        self.agents = DanteBenchmarkAgent(
            collector=None,
            runtime_factory=lambda endpoint: (production_runtime() if endpoint == PRODUCTION
                                              else benchmark_runtime(
                tags=[tag(SMALL, SMALL_DIGEST, 2497293931), tag(LARGE, LARGE_DIGEST, 13500737704,
                       family='qwen35', quant='Q3_K_M', params='26.9B'),
                      tag(DERIVED, 'b' * 64, 13500737325, family='qwen35', quant='Q3_K_M',
                          params='26.9B', parent=LARGE)],
                shown={SMALL: show(), LARGE: show(family='qwen35', blocks=65)})))

    def test_definition_declares_the_benchmark_capability(self):
        self.assertEqual(self.definition.agent_id, 'dante-benchmark')
        self.assertEqual(self.definition.capabilities, BENCHMARK_CAPABILITIES)
        self.assertNotIn('READ_ONLY_INVENTORY', self.definition.capabilities)
        self.assertEqual(self.definition.output_token_budget, 64)
        self.assertLessEqual(self.definition.timeout_s, 600)

    def test_capability_literals_stay_exclusive(self):
        for capabilities in (('LOCAL_INFERENCE', 'LOCAL_BENCHMARK'), ('UNKNOWN',),
                             ('READ_ONLY_INVENTORY', 'LOCAL_BENCHMARK')):
            with self.subTest(capabilities=capabilities):
                with self.assertRaises(ValidationError):
                    AgentDefinition(**{**self.definition.model_dump(), 'capabilities': capabilities})

    def test_read_only_capability_is_unchanged(self):
        self.assertEqual(READ_ONLY_CAPABILITIES, ('READ_ONLY_INVENTORY',))

    def test_derivative_is_excluded_with_its_declared_parent(self):
        runtime = benchmark_runtime(tags=[
            tag(SMALL, SMALL_DIGEST, 2497293931),
            tag(LARGE, LARGE_DIGEST, 13500737704, family='qwen35', quant='Q3_K_M', params='26.9B'),
            tag(DERIVED, 'b' * 64, 13500737325, family='qwen35', quant='Q3_K_M', params='26.9B',
                parent=LARGE)], shown={SMALL: show(), LARGE: show(family='qwen35', blocks=65)})
        targets, excluded = DanteBenchmarkAgent()._targets(runtime)
        self.assertEqual([item.model_id for item in targets], sorted([SMALL, LARGE]))
        reasons = {item['model_id']: item['reason'] for item in excluded}
        self.assertIn('Declared derivative of', reasons[DERIVED])
        self.assertIn(LARGE, reasons[DERIVED])

    def test_a_model_that_is_not_installed_is_recorded_as_skipped(self):
        runtime = benchmark_runtime(tags=[tag(SMALL, SMALL_DIGEST, 2497293931)], shown={SMALL: show()})
        targets, excluded = DanteBenchmarkAgent()._targets(runtime)
        self.assertEqual([item.model_id for item in targets], [SMALL])
        self.assertEqual(excluded[0]['model_id'], LARGE)
        self.assertIn('Not installed', excluded[0]['reason'])

    def test_duplicate_signature_is_not_benchmarked_twice(self):
        twin = tag('qwen3:4b:copy', 'c' * 64, 2497293931)
        runtime = benchmark_runtime(tags=[tag(SMALL, SMALL_DIGEST, 2497293931), twin],
                                    shown={SMALL: show()})
        agent = DanteBenchmarkAgent(models=(SMALL, 'qwen3:4b:copy'))
        targets, excluded = agent._targets(runtime)
        self.assertEqual(len(targets), 1)
        self.assertIn('already selected', excluded[0]['reason'])

    def test_targets_carry_measured_identity(self):
        targets, _ = DanteBenchmarkAgent()._targets(self.agents._runtime_factory(BENCH))
        large = next(item for item in targets if item.model_id == LARGE)
        self.assertEqual(large.digest_sha256, LARGE_DIGEST)
        self.assertEqual(large.artifact_bytes, 13500737704)
        self.assertEqual(large.block_count, 65)
        self.assertEqual(large.quantization, 'Q3_K_M')

    def test_agent_refuses_when_benchmark_endpoint_equals_production(self):
        agent = DanteBenchmarkAgent(benchmark_endpoint=PRODUCTION,
                                    runtime_factory=self.agents._runtime_factory)
        with self.assertRaises(QualificationRejected):
            agent.execute(_inference(PRODUCTION), _job(), self.definition, _payload(self.definition),
                          _spec(self.definition), threading.Event(), lambda *_a, **_k: None)

    def test_agent_performs_no_inference_on_production(self):
        production = production_runtime()
        runtime = benchmark_runtime(tags=[tag(SMALL, SMALL_DIGEST, 2497293931)],
                                    shown={SMALL: show()})
        agent = DanteBenchmarkAgent(models=(SMALL,), collector=None,
                                    runtime_factory=lambda endpoint: production if endpoint == PRODUCTION else runtime)
        inference = _inference(PRODUCTION)
        result = agent.execute(inference, _job(), self.definition, _payload(self.definition),
                               _spec(self.definition), threading.Event(), lambda *_a, **_k: None)
        self.assertEqual(result.metadata['inference_calls'], 0)
        self.assertFalse(result.metadata['inference_performed'])
        # Three fixed prompts over two repetitions, all on the benchmark runtime.
        self.assertEqual(len(runtime.generated), 6)
        self.assertEqual(production.generated, [])

    def test_agent_reports_isolation_and_withholds_rankings(self):
        production = production_runtime()
        runtime = benchmark_runtime(tags=[tag(SMALL, SMALL_DIGEST, 2497293931)],
                                    shown={SMALL: show()})
        agent = DanteBenchmarkAgent(models=(SMALL,),
                                    runtime_factory=lambda endpoint: production if endpoint == PRODUCTION else runtime)
        result = agent.execute(_inference(PRODUCTION), _job(), self.definition,
                               _payload(self.definition), _spec(self.definition),
                               threading.Event(), lambda *_a, **_k: None)
        metadata = result.metadata
        self.assertTrue(metadata['production_unchanged'])
        self.assertTrue(metadata['benchmark_restored'])
        self.assertTrue(metadata['rankings_withheld'])
        self.assertFalse(metadata['winners_declared'])
        self.assertEqual(metadata['role_assignments'], 0)
        self.assertTrue(metadata['smoke_only'])
        self.assertIsNone(result.structured_result['benchmark']['comparison']['overall_winner'])

    def test_agent_fails_when_isolation_cannot_be_verified(self):
        production = production_runtime()

        class Drifting(FakeRuntime):
            def residency(self):
                if self.generated:
                    return ({'name': 'other:1b', 'digest': 'c' * 64, 'size_vram': 1,
                             'context_length': 512, 'quantization': 'Q4_K_M',
                             'parameter_size': '1.0B'},)
                return super().residency()

        runtime = Drifting(BENCH, tags=[tag(SMALL, SMALL_DIGEST, 2497293931)], shown={SMALL: show()})
        agent = DanteBenchmarkAgent(models=(SMALL,),
                                    runtime_factory=lambda endpoint: production if endpoint == PRODUCTION else runtime)
        with self.assertRaises(QualificationRejected):
            agent.execute(_inference(PRODUCTION), _job(), self.definition, _payload(self.definition),
                          _spec(self.definition), threading.Event(), lambda *_a, **_k: None)

    def test_prompts_are_fixed_and_digest_proven(self):
        prompts = benchmark_prompts()
        self.assertEqual([item.prompt_id for item in prompts], ['LAT_1', 'INSTR_1', 'JSON_1'])
        self.assertEqual({item.kind for item in prompts},
                         {'latency', 'instruction', 'structured_json'})
        for item in prompts:
            self.assertEqual(item.digest, text_digest(item.text))
            self.assertEqual(item.characters, len(item.text))
        self.assertEqual(benchmark_prompts(), prompts)

    def test_samples_are_bound_to_a_hardware_snapshot_when_one_is_available(self):
        # A stand-in for the snapshot: the digest function itself is covered by the
        # Stage B1 tests, so what matters here is that the digest reaches every sample.
        class _Snapshot:
            def model_dump(self, **_keywords):
                return {'system': {'os': 'windows'}, 'coverage': {'sensors': 1}}

        expected = hardware_snapshot_digest(_Snapshot())
        production = production_runtime()
        runtime = benchmark_runtime(tags=[tag(SMALL, SMALL_DIGEST, 2497293931)],
                                    shown={SMALL: show()})
        agent = DanteBenchmarkAgent(models=(SMALL,), collector=_Snapshot,
            runtime_factory=lambda endpoint: production if endpoint == PRODUCTION else runtime)
        result = agent.execute(_inference(PRODUCTION), _job(), self.definition,
                               _payload(self.definition), _spec(self.definition),
                               threading.Event(), lambda *_a, **_k: None)
        benchmark = result.structured_result['benchmark']
        self.assertEqual(benchmark['hardware_snapshot_digest'], expected)
        for sample in benchmark['runs'][0]['samples']:
            self.assertEqual(sample['hardware_snapshot_digest'], expected)

    def test_an_unavailable_snapshot_is_recorded_as_absent(self):
        def explode():
            raise OSError('probe refused')
        production = production_runtime()
        runtime = benchmark_runtime(tags=[tag(SMALL, SMALL_DIGEST, 2497293931)],
                                    shown={SMALL: show()})
        agent = DanteBenchmarkAgent(models=(SMALL,), collector=explode,
            runtime_factory=lambda endpoint: production if endpoint == PRODUCTION else runtime)
        result = agent.execute(_inference(PRODUCTION), _job(), self.definition,
                               _payload(self.definition), _spec(self.definition),
                               threading.Event(), lambda *_a, **_k: None)
        self.assertIsNone(result.structured_result['benchmark']['hardware_snapshot_digest'])

    def test_registry_registers_the_benchmark_agent(self):
        registry = build_agent_registry(self.TARGET)
        self.assertIn('dante-benchmark', [item.agent_id for item in registry.list()])

    def test_benchmark_capability_requires_an_isolated_implementation(self):
        from dante.agent_runner import AgentRegistry, AgentRunner
        definition = self.definition
        registry = AgentRegistry()

        class Unisolated:
            READ_ONLY = False

            def execute(self, *_arguments, **_keywords):
                raise AssertionError('must not run')

        registry.register(definition, Unisolated())
        runner = AgentRunner(registry, _inference(PRODUCTION), _Store())
        with self.assertRaises(Exception) as caught:
            runner.execute(_job(), _spec(definition), threading.Event())
        self.assertIn('isolated benchmark implementation', str(caught.exception))


class ControlCenterBenchmarkViewTests(unittest.TestCase):
    def _structured(self):
        production = production_runtime()
        runtime = benchmark_runtime(tags=[tag(SMALL, SMALL_DIGEST, 2497293931)],
                                    shown={SMALL: show()})
        agent = DanteBenchmarkAgent(models=(SMALL,),
                                    runtime_factory=lambda endpoint: production if endpoint == PRODUCTION else runtime)
        definition = DanteBenchmarkAgent.definition(TARGET_MODEL)
        result = agent.execute(_inference(PRODUCTION), _job(), definition, _payload(definition),
                               _spec(definition), threading.Event(), lambda *_a, **_k: None)
        return result.structured_result

    def test_overview_exposes_isolation_and_dimensions(self):
        overview = benchmark_overview(self._structured())
        self.assertTrue(overview['isolation']['verified'])
        self.assertEqual(overview['isolation']['production_endpoint'], PRODUCTION)
        self.assertEqual(overview['isolation']['benchmark_endpoint'], BENCH)
        self.assertEqual(overview['models_tested'], [SMALL])
        self.assertTrue(overview['comparison']['dimensions'])
        self.assertIsNone(overview['comparison']['universal_score'])
        self.assertIsNone(overview['comparison']['overall_winner'])
        self.assertEqual(overview['role_assignments_made'], 0)

    def test_overview_shows_unknown_with_a_reason(self):
        overview = benchmark_overview(self._structured())
        samples = overview['models'][0]['samples']
        unknown = [cell for sample in samples for cell in sample['measurements'].values()
                   if cell['kind'] == 'unknown']
        for cell in unknown:
            self.assertIsNone(cell['value'])
            self.assertTrue(cell['reason'])

    def test_render_is_bounded_and_declares_no_winner(self):
        text = render_benchmark_text(benchmark_overview(self._structured()))
        self.assertIn('BENCHMARK', text)
        self.assertIn('production_unchanged=true', text)
        self.assertIn('overall_winner=None', text)
        self.assertIn('role_assignments=0', text)
        self.assertLess(len(text), 8000)

    def test_missing_payload_is_refused(self):
        with self.assertRaises(ValueError):
            benchmark_overview({'benchmark': None})


class BenchmarkTransportTests(unittest.TestCase):
    """Exercises the real HTTP path, including streaming decode.

    A fake runtime cannot catch a decoding mistake in the streaming reader, so
    every branch of `generate` is driven through a mock transport here.
    """

    def _runtime(self, handler):
        import httpx
        return BenchmarkRuntime(BENCH, transport=httpx.MockTransport(handler),
                                adapter_factory=lambda _profile: FakeAdapter(None))

    @staticmethod
    def _ndjson(*messages):
        return ''.join(json.dumps(item) + '\n' for item in messages).encode('utf-8')

    def test_stream_is_decoded_and_counters_kept(self):
        import httpx

        def handler(request):
            body = json.loads(request.content)
            self_check.append(body)
            return httpx.Response(200, content=self._ndjson(
                {'response': 'Hel', 'done': False},
                {'response': 'lo', 'done': False},
                {'response': '', 'done': True, 'done_reason': 'length',
                 'load_duration': 1_642_000_000, 'prompt_eval_count': 27,
                 'prompt_eval_duration': 69_540_000, 'eval_count': 5,
                 'eval_duration': 22_000_000, 'total_duration': 1_760_000_000}))

        self_check: list = []
        raw = self._runtime(handler).generate(SMALL, prompt='hi', parameters=benchmark_parameters())
        self.assertEqual(raw.response, 'Hello')
        self.assertEqual(raw.eval_count, 5)
        self.assertEqual(raw.prompt_eval_count, 27)
        self.assertEqual(raw.done_reason, 'length')
        self.assertIsNotNone(raw.time_to_first_token_s)
        self.assertEqual(raw.chunk_count, 3)
        self.assertIs(self_check[0]['think'], False)
        self.assertEqual(self_check[0]['options']['num_predict'], 48)

    def test_stream_without_a_final_message_is_refused(self):
        import httpx
        def handler(_request):
            return httpx.Response(200, content=self._ndjson({'response': 'x', 'done': False}))
        with self.assertRaises(BenchmarkRuntimeError):
            self._runtime(handler).generate(SMALL, prompt='hi', parameters=benchmark_parameters())

    def test_malformed_json_line_is_refused(self):
        import httpx
        def handler(_request):
            return httpx.Response(200, content=b'{not json\n')
        with self.assertRaises(BenchmarkRuntimeError) as caught:
            self._runtime(handler).generate(SMALL, prompt='hi', parameters=benchmark_parameters())
        self.assertIn('malformed', str(caught.exception))

    def test_error_envelope_is_refused(self):
        import httpx
        def handler(_request):
            return httpx.Response(200, content=self._ndjson({'error': 'model not found'}))
        with self.assertRaises(BenchmarkRuntimeError):
            self._runtime(handler).generate(SMALL, prompt='hi', parameters=benchmark_parameters())

    def test_non_200_is_refused(self):
        import httpx
        def handler(_request):
            return httpx.Response(500, content=b'overloaded')
        with self.assertRaises(BenchmarkRuntimeError) as caught:
            self._runtime(handler).generate(SMALL, prompt='hi', parameters=benchmark_parameters())
        self.assertIn('500', str(caught.exception))

    def test_unload_confirms_residency_is_gone(self):
        import httpx
        state = {'resident': True}

        class Unloading(FakeAdapter):
            def loaded(self):
                return ([{'name': SMALL, 'size_vram': 1, 'context_length': 4096}]
                        if state['resident'] else [])

        def handler(_request):
            state['resident'] = False
            return httpx.Response(200, content=b'{}')

        runtime = BenchmarkRuntime(BENCH, transport=httpx.MockTransport(handler),
                                   adapter_factory=lambda _profile: Unloading(None))
        runtime.unload(SMALL)
        self.assertFalse(state['resident'])

    def test_unload_that_never_takes_effect_is_refused(self):
        import httpx
        runtime = BenchmarkRuntime(BENCH, transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b'{}')),
            adapter_factory=lambda _profile: FakeAdapter(
                None, resident=[{'name': SMALL, 'size_vram': 1, 'context_length': 4096}]))
        with self.assertRaises(BenchmarkRuntimeError) as caught:
            runtime.unload(SMALL)
        self.assertIn('still resident', str(caught.exception))


class _Store:
    def __init__(self):
        self.events: list = []

    def record_event(self, job_id, event, attempt_id=None, **metadata):
        self.events.append((job_id, event, metadata))


class _Supervisor:
    def __init__(self, endpoint):
        self.config = type('Config', (), {'qualification': type('Qualification', (), {
            'profile': type('Profile', (), {'base_url': endpoint})()})()})()
        self.qualification_id = 'qual-fixture'


class _Inference:
    def __init__(self, endpoint):
        self.supervisor = _Supervisor(endpoint)


def _inference(endpoint=PRODUCTION):
    return _Inference(endpoint)


def _spec(definition):
    return WorkloadSpec(job_type='AGENT_TASK', prompt='run the bounded benchmark',
                        model=ModelRequirement(model_id=SMALL, runtime_reference=SMALL,
                                               digest_sha256=SMALL_DIGEST, context_tokens=4096),
                        max_output_tokens=64,
                        agent_task=AgentTaskPayload(agent_id=definition.agent_id, objective='x',
                                                    definition_version=definition.version,
                                                    definition_digest=definition_digest(definition)))


def _payload(definition):
    return AgentTaskPayload(agent_id=definition.agent_id, objective='x',
                            definition_version=definition.version,
                            definition_digest=definition_digest(definition))


def _job():
    from dante.workload import JobRecord
    now = datetime.now(timezone.utc)
    return JobRecord(job_id='job-benchmark-fixture', state='running', priority=0,
                     created_at=now, updated_at=now, attempt_count=1)


if __name__ == '__main__':
    unittest.main()
